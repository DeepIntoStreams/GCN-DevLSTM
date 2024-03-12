import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import import_class
from model.gcn import unit_gcn1
from model.dev_layer import *


class GCN_developRNN_basic(nn.Module):
    """This class is used to build the GCN-DevLSTM block
    Args:
        num_gcn_scales (int): the number of hops used in GCN 
        c_in (int): the channel of the input 
        c_out (int): the channel of the output 
        A_binary (np.array): the binary adjacency matrix
        choose_model (str): the type of the RNN model, only supports LSTM
        dilation (list): the dilation factor 
        stride (int): the stride size 
        kernel (int): the kernel size
        hidden_size (list): the hidden size of the Lie algbra matrix
        residual (bool, optional): If residual == True: use residual connection
                                   If residual == False: don't use residual connection
                                   Defaults to True.
        use_sp (bool, optional): If use_sp == True: use the first point of each partition as the start point
                                If use_sp == False: Don't use the start point of each partition
                                Defaults to True.
    """
    def __init__(self,
                num_gcn_scales,
                c_in,
                c_out,
                A_binary,
                choose_model,
                dilation,
                stride,
                kernel,
                hidden_size,
                residual = True,
                use_sp = True,
                ):
                
        super(GCN_developRNN_basic, self).__init__()
        self.gcn = unit_gcn1(c_in, c_out, A_binary,num_gcn_scales)
        self.choose_model = choose_model
        param = se # use se Lie algbera 
        h_size = hidden_size
        self.out = c_out
        self.use_sp = use_sp

        self.kernel_size = kernel
        self.stride = stride

        num_branch = len(dilation)
        self.num_branch = num_branch
        self.dev_layers = nn.ModuleList()
        
        for i in range(self.num_branch):
            if stride == 1 and dilation[i]==2:
                use_sp = False
                num_branch = 1

            self.dev_layers.append(dilation_dev(
                    dilation=dilation[i],
                    h_size=h_size,
                    param=param,
                    input_channel=c_out,
                    kernel_size=kernel,
                    stride=stride,
                    return_sequence=False,
                    use_sp=use_sp))
            
        channel_num = int(num_branch*c_out + 2*h_size * h_size) 
 

        if choose_model == 'LSTM':
            self.model = nn.LSTM(
                input_size=channel_num,
                hidden_size=c_out,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
        else: 
            raise NotImplementedError('model type only supports LSTM')
 
        self.pooling = nn.ModuleList()
        self.pooling.append(nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(c_out)  
        ))

        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(nn.Conv2d(c_out, int(c_out/2), kernel_size=1, padding=0),
            nn.BatchNorm2d(int(c_out/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(c_out/2),int(c_out),kernel_size=(5, 1),padding=(2, 0),stride=(stride, 1)),
            nn.BatchNorm2d(int(c_out))
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (c_in == c_out) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in,c_out,kernel_size=1,stride=(stride,1)),
                nn.BatchNorm2d(c_out)
            )

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(c_out)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        '''forward
        Args:  
            x (torch.tensor): A batch of skeleton data, shape (B,c_in,T,V)
        Returns:
            torch.tensor: the result of one GCN-DevLSTM block, shape (B,c_out,T',V)
                  where T' is the length of the output sequence
        '''
        N_M,C,T,V = x.shape
        
        x_in = x.clone()
        x = self.gcn(x)

        for module in self.pooling:
            x_pooling = module(x)

        for module in self.conv:
            x_conv = module(x)

        x = x.permute(0, 3, 2, 1).contiguous().view(
            N_M * V, T, self.out).contiguous()

        x_dev = []
        for i in range(self.num_branch):
            x_dev.append(self.dev_layers[i](x).type_as(x))
        x_dev = torch.cat(x_dev, axis=-1)
        _, T_segments, _ = x_dev.shape

        x, _ = self.model(x_dev)
        x = self.bn(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = x.view(
        N_M, V, T_segments, self.out).permute(0, 3, 2, 1).contiguous()

        x = x_pooling + x_conv+ x 
            
        if hasattr(self.residual, '__iter__'):
            for module in self.residual:
                x_in = module(x_in)
        else:
            x_in = self.residual(x_in)

        out = self.gelu(x + x_in)
        return out


class Model(nn.Module):
    """This class is used to build the GCN-DevLSTM model
    Args:
        num_class (int): the number of classes
        num_point (int): the number of joints
        num_person (int): the number of people
        num_gcn_scales (int): the number of hops used in GCN 
        graph (str): the type of the graph, only supports 'graph.ntu_rgb_d.Graph'
        labeling_mode (str): the type of the labeling mode, support 'None/spatial/dual'
        choose_model (str): the type of the RNN model, only supports LSTM
        hidden_size (list): the hidden size of the Lie algbra matrix
        kernel_size (list): kernel size
        stride (list): stride size 
        dilation (list): the dilation factor 
        in_channels (int, optional): the channel of the input data, Defaults to 3.
    """
    
    def __init__(self,
                num_class,
                num_point,
                num_person,
                num_gcn_scales,
                graph,
                labeling_mode,
                choose_model,
                hidden_size,
                kernel_size,
                stride,
                dilation,
                in_channels=3,
                ):

        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph(labeling_mode = labeling_mode).A_binary
        self.data_bn = nn.BatchNorm1d([num_person * in_channels * num_point])
        self.channels = [in_channels, 64, 64, 64, 64, 128, 128, 128, 256, 256]  
        self.developRNN_blocks = nn.ModuleList()

        for i in range(len(self.channels)-1):
            c_in = self.channels[i]
            c_out = self.channels[i+1]
            developRNN = GCN_developRNN_basic(
                num_gcn_scales,
                c_in,
                c_out,
                A_binary,
                choose_model,
                dilation[i],
                stride[i],
                kernel_size[i],
                hidden_size[i]
                )
            self.developRNN_blocks.append(developRNN)
        
        self.fc = nn.Linear(c_out, num_class)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """forward
        Args:  
            x (torch.tensor): A batch of skeleton data, shape (B,C,T,V,M)
        Returns:
            torch.tensor: the result of the GCN-DevLSTM model, shape (B,num_class)
        """

        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        
        x = self.data_bn(x)
        
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        for i in range(len(self.channels)-1):
            x = self.developRNN_blocks[i](x)
        
        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   
        out = out.mean(1)   
        out = self.fc(out)

        return out

