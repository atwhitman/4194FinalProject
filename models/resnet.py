# This is a resnet-18 model. It is modified from the torch implementation to use 1-D convolutions with multiple input channels.

import torch
import torch.nn as nn
import math

#torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

# torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)



# defines a basic convolutional function
def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False)





class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,norm_layer=None):
        
        super(BasicBlock, self)__init__()
        
        # define batch normalization method
        bn = nn.BatchNorm1d
        
        
        self.conv1 = conv(inplanes, planes, stride)
        self.bn1   = bn(planes)
        self.relu  = nn.ReLU(inplace=True)
        
        self.conv2 = conv(planes, planes)
        self.bn2   = bn(planes)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(out)
        
        out = self.conv2(x)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
        
        




















