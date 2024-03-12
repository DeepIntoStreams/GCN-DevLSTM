import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, './DevNet')
from development.nn import development_layer
from development.so import so
from development.se import se
import torch.nn.functional as F

class dilation_dev(nn.Module):
    """this class is used to partition the time sequence and calculate the path development of each partition.
    Args:
        dilation (int): the dilation factor
        h_size (int): size of the hidden Lie algbra matrix
        param (method): parametrisation method to map the GL matrix to required matrix Lie algebra.
        input_channel (int): the channel of the input sequence
        kernel_size (int): the length of each partition
        stride (int): stride size 
        use_sp (bool, optional): If use_sp == True: use the first point of each partition as the start point
                                 If use_sp == False: Don't use the start point of each partition
                                 Defaults to True.
        return_sequence (bool, optional): If return_sequence == True: return whole development path
                                          If return_sequence == False, only return the matrix Lie group element at the last time step
                                          Defaults to False.
    """
    def __init__(self, dilation, h_size, param, input_channel, kernel_size, stride, use_sp = True, return_sequence=False):
        super(dilation_dev, self).__init__()

        self.param = param
        self.h_size = h_size
        self.return_sequence = return_sequence
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_sp = use_sp
        self.pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2 
        self.unfold = nn.Unfold([kernel_size,1], dilation=[dilation,1], padding=[0,0], stride=[stride,1])
        
        self.development_layers = development_layer(
            input_size=input_channel, hidden_size=self.h_size, channels=1, param=self.param,
            return_sequence=return_sequence, complexification=False)
    

    def forward(self, inp):    
        """forward 
        Args:
            inp (torch.tensor): A batch of matrix time series, shape (B,T,C)
        Returns:
            torch.tensor: path developement result with start point of each partition, shape (B,T',C+m^2) if use_sp == True
                          path developement result of each partition, shape (B,T',m^2) if use_sp == False
        """
        # padding the input sequence
        if self.dilation == 1 and self.stride == 1:
            p3d = (0, 0, 0, 1) 
        else:
            p3d = (0, 0, self.pad, self.pad) 
        inp = F.pad(inp, p3d, "replicate")

        inp = inp.permute(0,2,1).unsqueeze(-1) # N,C,T

        segment = int((inp.shape[2] - (self.kernel_size-1)*self.dilation-1)/self.stride+1) # number of partitions
        
        # partition the input sequence according to the kernel, dilation and stride
        x = self.unfold(inp).contiguous().view(inp.shape[0],inp.shape[1],self.kernel_size,segment).permute(0,3,2,1).contiguous()\
        .view(inp.shape[0]*segment,self.kernel_size,inp.shape[1])  

        dev_out = self.development_layers(x).view(inp.shape[0],segment,-1) # Output dimension size: N,segment,m^2

        # use the first point of each partition as the start point
        if self.use_sp: 
            sp = x[:,0,:].view(inp.shape[0],segment,-1)
            return torch.cat([dev_out,sp],axis=-1) 
        
        return dev_out
 

