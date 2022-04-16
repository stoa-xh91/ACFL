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


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1,1,1), downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, (1,3,3), 1, (0,1,1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, (3,1,1), 1, (1,0,0))
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv4 = nn.Conv3d(planes, planes * self.expansion, 1, 1)
        self.bn4 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FactorizedBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1,1,1), downsample=None):
        super(FactorizedBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
      
        self.conv1 = nn.Conv3d(inplanes, planes, (3,1,1), stride, (1,0,0))
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, (1,3,3), 1, (0,1,1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, 1)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SpatialBasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1,1,1), downsample=None):
        super(SpatialBasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
      
        self.conv1 = nn.Conv3d(inplanes, planes, 1, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, (1,3,3), 1, (0,1,1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, 1)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SkeC3D(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, base_channel=64,
                 drop_out=0, adaptive=True):
        super(SkeC3D, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        

        base_channel = 32
        # stem
        self.stem = nn.Conv3d(1, base_channel, (1,7,7),1,(0,3,3))
        self.bn1 = nn.BatchNorm3d(base_channel)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = base_channel
        # 
        self.stage1 = self._make_layer(base_channel*1, SpatialBasicBlock, 4, 2)
        self.stage2 = self._make_layer(base_channel*2, FactorizedBlock, 6, 2)
        self.stage3 = self._make_layer(base_channel*4, FactorizedBlock, 3, 2)
        # self.stage4 = self._make_layer(base_channel*8, 2, 2)

        self.head = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(base_channel*16, base_channel*8)
        self.fc = nn.Linear(base_channel*8, num_class)
        self.init_weights()
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
    
    def _make_layer(self, planes, block, blocks, stride=1, temporal_stride=1):
        downsample = None
        layers = []
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=(temporal_stride, stride, stride), bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
            
        layers.append(block(self.inplanes, planes, (temporal_stride, stride, stride), downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / m.weight.size(0)))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        B, C, T, H, W = x.size()
        x = self.stem(x)
        x = self.relu(self.bn1(x))
        #
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        # x = self.stage4(x)
        #
        x = self.head(x3)
        x = torch.flatten(x, 1)
        x = self.drop_out(x)
        x = self.linear(x)
        x = torch.relu(x)
        
        logits = self.fc(x)
        return {'logits':logits, 'feature':x}

