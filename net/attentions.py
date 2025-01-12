import torch
from torch import nn
import torch.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter



class Attention_Layer(nn.Module):
    def __init__(self, out_channel, att_type, **kwargs):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
        }

        self.att = __attention[att_type](channel=out_channel, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        #self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return (self.bn(x) + res)

class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att

class Part_Att(nn.Module):
    def __init__(self, channel, parts, bias, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        inner_channel = channel

        self.softmax = nn.Softmax(dim=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, channel*len(self.parts), kernel_size=1, bias=bias),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_att = self.softmax(self.fcn(x).view(N, C, 1, len(self.parts)))
        x_att = x_att.index_select(3, self.joints).expand_as(x)
        return x_att

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)

class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)

class mohammad_att(nn.Module):
    def __init__(self,channel,**kwargs):
        super(mohammad_att, self).__init__()
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel*4,channel*4,kernel_size=1),
            nn.BatchNorm2d(channel*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel*4,channel*4,kernel_size=1),
            nn.Sigmoid()
                                 )
    def forward(self, x):
        return self.fcn(x)

class Frame_Att(nn.Module):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9,1), padding=(4,0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        return self.conv(x)

class Joint_Att(nn.Module):
    def __init__(self, parts, **kwargs):
        super(Joint_Att, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joint, num_joint//2, kernel_size=1),
            nn.BatchNorm2d(num_joint//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joint//2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
       #return self.fcn(x.transpose(1, 3)).transpose(1, 3)
        return self.fcn

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=32,**kwargs):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

class TemporalAttention(nn.Module):
    r"""An implementation of the Temporal Attention Module( i.e. compute temporal attention scores). For details see this paper:
    `"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow
    Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(TemporalAttention, self).__init__()

        self._U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))  # for example 307
        self._U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))  # for example (1, 307)
        self._U3 = nn.Parameter(torch.FloatTensor(in_channels))  # for example (1)
        self._be = nn.Parameter(
            torch.FloatTensor(1, num_of_timesteps, num_of_timesteps)
        )
        self._Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))  # for example (12, 12)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the temporal attention layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).

        Return types:
            * **E** (PyTorch FloatTensor) - Temporal attention score matrices, with shape (B, T_in, T_in).
        """

        LHS = torch.matmul(torch.matmul(X.permute(0, 3, 2, 1), self._U1), self._U2)


        RHS = torch.matmul(self._U3, X)


        E = torch.matmul(self._Ve, torch.sigmoid(torch.matmul(LHS, RHS) + self._be))
        E = F.softmax(E, dim=1)
        return E

class selfAtt(nn.Module):
    def __init__(self, ch , **kwargs):
        super(selfAtt, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(ch , ch , kernel_size=1),
            nn.BatchNorm2d(ch),
            nn.Tanh(),
            nn.Conv2d(ch, ch , kernel_size=1),
            nn.Softmax(dim=1)

        )
    def forward(self,x):
        return self.fcn(x)


class Spatio_Temporal_Att(nn.Module):
    def __init__(self , in_channel , bias , **kwargs):
        super(Spatio_Temporal_Att, self).__init__()

        self.FC_Net = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=1,bias=bias),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel,in_channel,kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )

        self.Temporal_Conv = nn.Conv2d(in_channel,in_channel,kernel_size=1)
        self.Spatial_Conv = nn.Conv2d(in_channel,in_channel , kernel_size=1)

    def forward (self, x):
        T , V , C = x.size()

        spatial_x = x.mean(1 , keepdims = True).transpose(2 , 3)
        temporal_x = x. mean(0 , keepdims = True)
        attention_x = self.FC_Net(torch.cat([temporal_x,spatial_x] , dim = 2))
        temporal_x , spatial_x = torch.split(attention_x, [T,V], dim=2)
        x_s = self.Spatial_Conv(spatial_x.transpose( 2 , 3 )).sigmoid()
        x_t = self.Temporal_Conv(temporal_x).sigmoid()
        attention_x = x_t * x_s
        return attention_x

class att_self(nn.Module):
    def __init__(self, in_channel, bias, **kwargs):
        super(att_self, self).__init__()
        self.net = nn.Conv2d(in_channel,in_channel,kernel_size=1)
    def forward(self,x):
        C,T,V,N = x.size()
        x1= x.mean(0,keepdims = True)
        x2 = x.mean(3 , keepdims = True)
        attention_x0 = self.net(torch.cat([x1, x2], dim=2))
        attention_x1 = self.net(torch.cat([x2, x1], dim=2))
        attention_x = attention_x0*attention_x1
        return attention_x

   # if self.inter_channels is None:
   #          self.inter_channels = in_channels // 2
   #          if self.inter_channels == 0:
   #              self.inter_channels = 1


# if dimension == 3:
#     conv_nd = nn.Conv3d
#     max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
#     bn = nn.BatchNorm3d
# elif dimension == 2:
#     conv_nd = nn.Conv2d
#     max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
#     bn = nn.BatchNorm2d
# else:
#     conv_nd = nn.Conv1d
#     max_pool_layer = nn.MaxPool1d(kernel_size=(2))
#     bn = nn.BatchNorm1