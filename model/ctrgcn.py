import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)
    


def  conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        conv_init(self.conv_q_right)
        conv_init(self.conv_v_right)
        conv_init(self.conv_q_left)
        conv_init(self.conv_v_left)

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class geom_conv(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(geom_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels==6 or in_channels == 9:
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=-1, keepdim=True) + 1e-5  # N,C,V,1
        A = A / A_norm
        return A

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class multimodal_conv(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(multimodal_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels==6 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x1, x2, A1=None, A2=None, alpha=1):
        x1, x2, x3, x4 = self.conv1(x1).mean(-2), self.conv2(x2).mean(-2), self.conv3(x1), self.conv4(x2)
        rel = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1, x2 = self.conv5(rel) * alpha, self.conv6(rel)
        x1 = x1 + (A1 if A1 is not None else 0)  # N,C,V,V
        x2 = x2 + (A2 if A2 is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x2 = torch.einsum('ncuv,nctv->nctu', x2, x4)
        return x1, x2

class AMA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(AMA, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_v = nn.Conv2d(self.inplanes, self.planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_k = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        
        self.conv_c = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_down = nn.Conv2d(self.planes+self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self._init_modules()

    def _init_modules(self):
        conv_init(self.conv_v)
        conv_init(self.conv_k)
        conv_init(self.conv_q)
        conv_init(self.conv_c)
        conv_init(self.conv_down)
        bn_init(self.bn, 1)


    def forward(self, x):
        
        value, key, query = self.conv_v(x), self.conv_k(x), self.conv_q(x)

        batch, channel, T, V = key.size()
        # [N, C, T*V]
        key = key.view(batch, channel, T * V)

        # [N, 1, T*V]
        query = query.view(batch, 1, T * V)

        # [N, 1, T*V]
        query = self.softmax(query)

        # [N, C, 1, 1]
        interaction = torch.matmul(key, query.transpose(1,2)).unsqueeze(-1)

        # [N, \hat{C}, 1, 1]
        interaction = self.conv_c(interaction)

        # [N, \hat{C}, 1, 1]
        attention = self.sigmoid(interaction)

        attended_emb = value * attention
        
        out = self.bn(self.conv_down(torch.cat([attended_emb, x], dim=1)))

        return out

class unit_mm_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_mm_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        if A is not None:
            self.num_subset = A.shape[0]
        else:
            self.num_subset = 1
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(multimodal_conv(in_channels, out_channels))

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
        if A is not None:
            if self.adaptive:
                self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            else:
                self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        else:
            self.A = None
            self.PA = None
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn1, 1e-6)
        bn_init(self.bn2, 1e-6)

    def forward(self, x1, x2):
        y1 = None
        y2 = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x1.get_device()) if self.A is not None else self.A
        for i in range(self.num_subset):
            z1, z2 = self.convs[i](x1, x2, A[i] if A is not None else A, self.alpha)
            y1 = z1 + y1 if y1 is not None else z1
            y2 = z2 + y2 if y2 is not None else z2
        y1 = self.bn1(y1)
        y1 += self.down(x1)
        y1 = self.relu(y1)
        y2 = self.bn2(y2)
        y2 += self.down(x2)
        y2 = self.relu(y2)
        
        return y1, y2

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(geom_conv(in_channels, out_channels))

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
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

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

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        # self.psa = PSA_p(out_channels, out_channels)
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # y = self.relu(self.tcn1(self.psa(self.gcn1(x))) + self.residual(x))
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class MM_TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(MM_TCN_GCN_unit, self).__init__()
        self.conv1 = nn.Conv2d(2*out_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mm_gcn1 = unit_mm_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.mm_gcn2 = unit_mm_gcn(in_channels, out_channels, None, adaptive=adaptive)
        self.tcn = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.conv2 = nn.Conv2d(2*out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        conv_init(self.conv1)
        bn_init(self.bn1, 1)
        conv_init(self.conv2)
        bn_init(self.bn2, 1)

    def forward(self, x1, x2):
        x = self.relu(self.bn1(self.conv1(torch.cat([x1, x2], dim=1))))
        m11, m21 = self.mm_gcn1(x1, x2)
        m12, m22 = self.mm_gcn2(x1.permute(0, 1, 3, 2), x2.permute(0, 1, 3, 2))
        y1 = self.relu(self.tcn(m11) + self.tcn(m12.permute(0, 1, 3, 2)) + self.residual(x1))
        y2 = self.relu(self.tcn(m21) + self.tcn(m22.permute(0, 1, 3, 2)) + self.residual(x2))
        y = self.relu(x + self.bn2(self.conv2(torch.cat([y1, y2], dim=1))))
        return y

class Multimodal_unit(nn.Module):
    def __init__(self,  model_dims=60, num_heads=1):
        super(Multimodal_unit, self).__init__()
        # Joint encoder
        self.ske_encoder_sa = nn.MultiheadAttention(model_dims, num_heads)
        self.ske_encoder_linear1 = nn.Linear(model_dims, model_dims)
        self.ske_encoder_linear2 = nn.Linear(model_dims, model_dims)
        self.ske_encoder_norm1   = nn.LayerNorm(model_dims)
        self.ske_encoder_norm2   = nn.LayerNorm(model_dims)
        # Bone encoder
        self.act_encoder_sa = nn.MultiheadAttention(model_dims, num_heads)
        self.act_encoder_linear1 = nn.Linear(model_dims, model_dims)
        self.act_encoder_linear2 = nn.Linear(model_dims, model_dims)
        self.act_encoder_norm1   = nn.LayerNorm(model_dims)
        self.act_encoder_norm2   = nn.LayerNorm(model_dims)
        # Modal fusion
        self.act_fusion_linear = nn.Linear(2 * model_dims, model_dims)
        self.act_decoder_sa = nn.MultiheadAttention(model_dims, num_heads)
        self.act_decoder_linear1 = nn.Linear(model_dims, model_dims)
        self.act_decoder_linear2 = nn.Linear(model_dims, model_dims)
        self.act_decoder_norm1   = nn.LayerNorm(model_dims)
        self.act_decoder_norm2   = nn.LayerNorm(model_dims)
        # 
        self.act_fn  = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def transformer_encoding(self, x, query, key, value, att_fn, linear1=None, linear2=None, norm1=None, norm2=None):
        attended_emb, att_scores = att_fn(query, key, value)
        x = x + attended_emb
        if norm1 is not None:
            x = norm1(x)
        src_x = x
        if linear1 is not None:
            x = linear1(x)
        x = self.act_fn(x)
        if linear2 is not None:
            x = linear2(x)
        x = x + src_x
        if norm2 is not None:
            x = norm2(x)
        return self.act_fn(x)

    def forward(self, x1, x2):
        bs = x1.size(0)
        x1 = x1.permute(2, 0, 1).contiguous()
        x2 = x2.permute(2, 0, 1).contiguous()
        
        # Joint encoder
        encoded_m1_feat = self.transformer_encoding(x1, query=x2, key=x1, value=x1, 
                                            att_fn=self.ske_encoder_sa,
                                            linear1=self.ske_encoder_linear1,
                                            linear2=self.ske_encoder_linear2,
                                            norm1=self.ske_encoder_norm1,
                                            norm2=self.ske_encoder_norm2)
        
        # action semantics encdoer
        encoded_m2_feat = self.transformer_encoding(x2, query=x1, key=x2, value=x2, 
                                            att_fn=self.act_encoder_sa,
                                            linear1=self.act_encoder_linear1,
                                            linear2=self.act_encoder_linear2,
                                            norm1=self.act_encoder_norm1,
                                            norm2=self.act_encoder_norm2)
        
        # multimodal decoder
        x = self.act_fusion_linear(torch.cat([encoded_m1_feat, encoded_m2_feat], dim=-1))
        decoded_emb = self.transformer_encoding(x, query=x, key=x, value=x, 
                                            att_fn=self.act_decoder_sa,
                                            linear1=self.act_decoder_linear1,
                                            linear2=self.act_decoder_linear2,
                                            norm1=self.act_decoder_norm1,
                                            norm2=self.act_decoder_norm2)
        decoded_emb = decoded_emb.permute(1, 2, 0).contiguous()
        return decoded_emb.squeeze()


class global_fusion(nn.Module):
    def __init__(self, in_channels=256, max_persons=2, num_points=25, max_length=16):
        super(global_fusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_encoder = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.context_encoder = nn.Conv1d(in_channels, in_channels, max_persons, 1)
        self.norm2 = nn.BatchNorm1d(in_channels)
        self.act_fn = nn.ReLU()
        self.max_persons = max_persons
        self.hidden_dims = in_channels
        # self.init_weights()

    def init_weights(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):

                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # N*M,C,T,V
        bs, c_new, T, V = x.size()
        x = x.view(bs, c_new, T*V)
        x = self.global_pool(x)
        x = self.global_encoder(x)
        x = self.act_fn(self.norm1(x))
        x = x.reshape(-1, self.max_persons, c_new).permute(0, 2, 1, 3)
        x = self.global_pool(x)
        x = self.context_encoder(x)
        x = self.act_fn(self.norm2(x))
        return x.reshape(-1, c_new)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25    
        # A = np.random.randn(1, num_point, num_point)
        # A_eye = np.eye(num_point)[np.newaxis, :]
        # A = np.concatenate([A_eye, A_eye, A], axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        # self.max_pool = nn.MaxPool2d((3, 1), (2, 1), padding=(1, 0))
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.interaction = InterAction(64)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)

        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        #self.l11 = global_fusion(base_channel*4, max_persons=num_person, num_points=num_point)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        # x = self.l11(x)
        x = self.drop_out(x)

        return self.fc(x)


class NaiveMultiModel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(NaiveMultiModel, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        in_channels *= 2
        A = self.graph.A # 3,25,25
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        #
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        #
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        #
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)


class MultiModalGcn(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(MultiModalGcn, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        
        A = self.graph.A # 3,25,25
        # M1_A = np.random.randn(1, 1, num_point, num_point)
        # self.M1_A = nn.Parameter(torch.from_numpy(M1_A.astype(np.float32)), requires_grad=True)
        # M2_A = np.random.randn(1, 1, num_point, num_point)
        # self.M2_A = nn.Parameter(torch.from_numpy(M2_A.astype(np.float32)), requires_grad=True)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * 2 * num_point)
        self.mm_fusion = AMA(in_channels*2, 16)
        # self.mm_fusion = multimodal_conv(in_channels, in_channels)
        # self.mm_bn = nn.BatchNorm2d(in_channels * 2)
        # self.relu = nn.ReLU(inplace=True)
        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels*2, base_channel, A, residual=False, adaptive=adaptive)
        #
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        #
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        #
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        # bn_init(self.mm_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.mm_fusion(x)
        # x1, x2 = self.mm_fusion(x[:,:C//2,...],x[:,C//2:,...],self.M1_A,self.M2_A)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.relu(self.mm_bn(x))
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

class MultiModalModel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(MultiModalModel, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A # 3,25,25
        self.num_class = num_class
        self.num_point = num_point
        self.channels_per_modal = in_channels
        self.data1_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data2_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        #
        self.l2 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l3 = MM_TCN_GCN_unit(base_channel, base_channel, A, residual=False, adaptive=adaptive)
        #
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        #
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data1_bn, 1)
        bn_init(self.data2_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x1, x2 = x[:,:self.channels_per_modal, ...], x[:, self.channels_per_modal:, ...]
        x1 = x1.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * self.channels_per_modal, T)
        x2 = x2.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * self.channels_per_modal, T)
        x1, x2 = self.data1_bn(x1), self.data2_bn(x2)
        x1 = x1.view(N, M, V, self.channels_per_modal, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, self.channels_per_modal, T, V)
        x2 = x2.view(N, M, V, self.channels_per_modal, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, self.channels_per_modal, T, V)
        x1 = self.l1(x1)
        x2 = self.l2(x2)
        # C = x1.size(1)
        # x1 = x1.view(N * M, C, T * V)
        # x2 = x2.view(N * M, C, T * V)
        x = self.l3(x1, x2)
        # x = x.view(N * M, C, T, V)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)