import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import math
import collections
from itertools import repeat
from net.utils.graph import Graph
from .attentions import ST_Joint_Att
from .attentions import *

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

A_25 = np.array([[
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],  # 5
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],  # 9
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],  # 13
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],  # 17
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],  # 21
    [0, 0, 1, 0, 0],  # 22
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],  # 24
    [0, 1, 0, 0, 0]
]], dtype=float)





def conv_dw(inp, oup, kernel_size, stride, padding, dilation, bias):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, dilation=dilation, bias=bias),

        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),

        nn.BatchNorm2d(oup),
        # nn.LeakyReLU(),
        # nn.BatchNorm2d(oup)

    )

class SPGConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, A=None, bias=False,**kwargs):
        super().__init__()

        A_size = A.shape
        self.register_buffer('A', A*A)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.dw_gcn_weight = nn.Parameter(torch.Tensor(in_channels, A_size[1], A_size[2]))
        self.pw_gcn_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bn = nn.BatchNorm2d(in_channels)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.dw_gcn_weight.data.uniform_(-stdv, stdv)
        self.pw_gcn_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x,**kwargs):
        self.A = np.multiply(self.A,self.A)
        dw_gcn_weight = self.dw_gcn_weight.mul(self.A)
        x = torch.einsum('nctv,cvw->nctw', (x, dw_gcn_weight))
        x = self.bn(x)
        x = torch.einsum('nctw,cd->ndtw', (x, self.pw_gcn_weight))

        return x

class TemporalConv(nn.Module):
    def __init__(self, out_channels, t_kernel_size, stride, padding, dilation, bias, dropout):
        super().__init__()

        layers = []
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            conv_dw(out_channels,
                    out_channels,
                    (t_kernel_size, 1),
                    (stride, 1),
                    padding,
                    dilation=(dilation, 1),
                    bias=bias,)
        )

        if dropout:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        return self.conv(x)






class STCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, dropout=0, residual=True, bias=False, A=None,**kwargs):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (((kernel_size[0] - 1) * dilation) // 2, 0)

        self.gcn = SPGConv(in_channels, out_channels, kernel_size[1], A=A, bias=bias)
        self.dilation = dilation
        self.t_kernel_size = kernel_size[0]
        self.tcn = TemporalConv(out_channels, self.t_kernel_size, stride, padding, dilation, bias, dropout)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                conv_dw(
                    in_channels,
                    out_channels,
                    kernel_size=(self.t_kernel_size, 1),   # init: 1, my: kernel_size
                    stride=(stride, 1),
                    padding=padding,
                    dilation=(self.dilation, 1),
                    bias=bias
                )
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,**kwargs):

        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res

        return  self.relu(x)






class HPGNet(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, group_num=5, in_plane=16, dilation=1, topology='physical', **kwargs):
        super().__init__()

        # load graph
        self.target_group_num = group_num
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        # config graph
        if topology == 'complete':
            A = torch.ones(self.graph.A.shape)
        elif topology == 'complement':
            graph_path = 'complement_graph_2.npz'
            print('load graph_path: %s' % graph_path)
            seleted_graph = np.load(graph_path)
            A = seleted_graph['arr_1']
            A = torch.tensor(A, dtype=torch.float32, requires_grad=False).unsqueeze(0)
            A_embed = torch.tensor(1 - A_25, dtype=torch.float32, requires_grad=False)
            print('complement graph.')
        else:
            A_embed = torch.tensor(A_25, dtype=torch.float32, requires_grad=False)
            # A_embed = torch.ones([A.size(0), A.size(1), group_num])
            print('physical graph.')

        A_group = torch.ones([A.size(0), group_num, group_num])
        A_config = [A, A, A, A_embed, A_group, A_group]

        self.register_buffer('A', A)
        self.register_buffer('A_group', A_group)
        self.register_buffer('A_embed', A_embed)
        # graph_size = A.shape

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.in_plane = in_plane
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        # self.motion_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.skeleton_net = nn.Sequential(
            STCBlock(in_channels, self.in_plane, kernel_size, 2, dilation, A=A_config[0], residual=False, **kwargs),
            STCBlock(self.in_plane, self.in_plane, kernel_size, 2, dilation, A=A_config[1], **kwargs),
            STCBlock(self.in_plane, self.in_plane * 2 ** 1, kernel_size, 2, dilation, A=A_config[2], **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A_config[3], residual=False, **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 2, kernel_size, 2, dilation, A=A_config[4], **kwargs),
            STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 1, dilation, A=A_config[5], **kwargs),


        )

        # fcn for prediction
        self.fcn = nn.Conv2d(self.in_plane * 2 ** 2, num_class, kernel_size=1)



    def forward(self, x,**kwargs):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        x = self.skeleton_net(x)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])

        # multi-person
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

class HPGNetFusion(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, group_num=5, in_plane=16, dilation=1, topology='physical',
                 **kwargs):
        super().__init__()

        # load graph
        self.target_group_num = group_num
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = np.multiply(A,A)
        # config graph
        if topology == 'complete':
            A = torch.ones(self.graph.A.shape)
        elif topology == 'complement':
            graph_path = 'complement_graph_2.npz'
            print('load graph_path: %s' % graph_path)
            seleted_graph = np.load(graph_path)
            A = seleted_graph['arr_1']
            A = torch.tensor(A, dtype=torch.float32, requires_grad=False).unsqueeze(0)
            A_embed = torch.tensor(1 - A_25, dtype=torch.float32, requires_grad=False)
            print('complement graph.')
        else:
            A_embed = torch.tensor(A_25, dtype=torch.float32, requires_grad=False)
            # A_embed = torch.ones([A.size(0), A.size(1), group_num])
            print('physical graph.')

        A_group = torch.ones([A.size(0), group_num, group_num])
        A_config = [A*A, A*A, A_embed, A_group, A_group, A_group]
        # A_config = [A, A, A, A, A, A]
        self.register_buffer('A2',A*A)
        self.register_buffer('A', A)
        self.register_buffer('A_group', A_group)
        self.register_buffer('A_embed', A_embed)
        # graph_size = A.shape

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.in_plane = in_plane
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.motion_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.skeleton_net = nn.Sequential(
            STCBlock(in_channels, self.in_plane, kernel_size, 2, dilation, A=A_config[0], residual=False, **kwargs),
            STCBlock(self.in_plane, self.in_plane, kernel_size, 2, dilation, A=A_config[1], **kwargs),
            STCBlock(self.in_plane, self.in_plane * 2 ** 1, kernel_size, 2, dilation, A=A_config[2], residual=False, **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A_config[3], **kwargs),
            # STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A_config[3], **kwargs),

        )
        # self.add_module('att1',ST_Joint_Att(self.in_plane * 2, 1, bias=None, **kwargs))
        # self.add_module('att2',selfAtt(self.in_plane*2**2 , **kwargs) )
        # ST_Joint_Att(self.in_plane * 2, 1, bias=None, **kwargs)
        # selfAtt(self.in_plane*2**2 , **kwargs)
        # self.add_module('ya',att_self(self.in_plane,None,**kwargs))
        # self.add_module('attt',Part_Att(in_channels,parts=A_25 , bias=None,**kwargs))
        # self.add_module('m',Spatio_Temporal_Att(self.in_plane,None,**kwargs))

        self.motion_net = nn.Sequential(
            STCBlock(in_channels, self.in_plane, kernel_size, 2, dilation, A=A_config[0], residual=False, **kwargs),
            STCBlock(self.in_plane, self.in_plane, kernel_size, 2, dilation, A=A_config[1], **kwargs),
            STCBlock(self.in_plane, self.in_plane * 2 ** 1, kernel_size, 2, dilation, A=A_config[2], residual=False, **kwargs),
            STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A_config[3], **kwargs),
            # STCBlock(self.in_plane * 2 ** 1, self.in_plane * 2 ** 1, kernel_size, 1, dilation, A=A_config[3], **kwargs),

        )
        # self.add_module('att3',Frame_Att(**kwargs))
        # Frame_Att(**kwargs)
        # selfAtt(in_channels * 2 ** 4, **kwargs)
        # self.add_module('m', Spatio_Temporal_Att(self.in_plane, None, **kwargs))
        # self.add_module('ya', att_self(self.in_plane, None, **kwargs))
        self.fusion = nn.Sequential(
            STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 2, dilation, A=A_config[4], **kwargs),
            STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 1, dilation, A=A_config[5], **kwargs),
            # STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 1, dilation, A=A_config[5], **kwargs),
            # STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 1, dilation, A=A_config[5], **kwargs),
            # STCBlock(self.in_plane * 2 ** 2, self.in_plane * 2 ** 2, kernel_size, 1, dilation, A=A_config[5], **kwargs),
        )
        # self.add_module('att4',selfAtt(in_channels*2**4 , **kwargs))
        # selfAtt(in_channels*2**4 , **kwargs)
        # self.add_module('m', Spatio_Temporal_Att(self.in_plane, None, **kwargs))
        # fcn for prediction
        self.fcn = nn.Conv2d(self.in_plane * 2 ** 2, num_class, kernel_size=1)


    def forward(self, x,**kwargs):
        motion = x[1]
        x = x[0]

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # motion nornalization
        motion = motion.permute(0, 4, 3, 1, 2).contiguous()
        motion = motion.view(N * M, V * C, T)
        motion = self.motion_bn(motion)
        motion = motion.view(N, M, V, C, T)
        motion = motion.permute(0, 1, 3, 4, 2).contiguous()
        motion = motion.view(N * M, C, T, V)

        # forwad

        x = torch.cat([self.skeleton_net(x), self.motion_net(motion)], 1).contiguous()

        x = self.fusion(x)

        # global pooling
        #x = F.avg_pool2d(x, x.size()[2:])
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        # multi-person
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x








class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, group_num, in_plane, dilation, topology, **kwargs):
        super().__init__()
        self.position_net = \
            HPGNet(in_channels, num_class, graph_args, group_num, in_plane, dilation, topology, **kwargs)

    def forward(self, x,**kwargs):
        position = x[0]

        out = self.position_net(position)


        return out


class ModelFusion(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, group_num, in_plane, dilation, topology, **kwargs):
        super().__init__()
        self.fusion_net = HPGNetFusion(in_channels, num_class, graph_args, group_num, in_plane, dilation, topology, **kwargs)

    def forward(self, x):

        return self.fusion_net(x)


