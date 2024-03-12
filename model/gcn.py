import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph.tools import k_adjacency, normalize_adjacency_matrix
from model.activation import activation_factory
from torch.autograd import Variable

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         conv_init(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_gcn1(nn.Module):
    #this class is for CTR-GC used in CTR-GCN 
    def __init__(self, in_channels, out_channels, A, num_gcn_scales, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn1, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        # num_gcn_scales = 3 for three hops setting. comment belowing line 52 when labeling_mode == 'spatial'
        A = np.array([k_adjacency(A, k, with_self=True) for k in range(num_gcn_scales)])   
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        # self.fc = nn.Conv2d(3*out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         conv_init(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         bn_init(m, 1)
        # bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())

        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class unit_gcn(nn.Module): 
    # this class is for adaptive graph convolution in 2s-agcn
    def __init__(self, in_channels, out_channels, A, num_gcn_scales, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        #num_gcn_scales = 3 for three hops setting. comment belowing line 111 when labeling_mode == 'spatial'
        A = np.array([k_adjacency(A, k, with_self=True) for k in range(num_gcn_scales)])
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class fixed_graph(nn.Module):
    # this class is for fixed graph without learnable parameters for altering adjacency matrix
    def __init__(self, in_channels, out_channels, A, num_gcn_scales):
        super(fixed_graph, self).__init__()
        
        #num_gcn_scales = 3 for three hops setting. comment belowing line 163 when labeling_mode == 'spatial'
        A = np.array([k_adjacency(A, k, with_self=True) for k in range(num_gcn_scales)])
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        
        self.num_subset = num_gcn_scales

        self.conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)
