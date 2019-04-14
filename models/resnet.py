# This is a resnet-18 model. It is modified from the torch implementation to use 1-D convolutions with multiple input channels.

import torch
import torch.nn as nn
import math

# defines a basic convolutional function
def conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()       
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.conv2 = conv3x1(  planes, planes)
        
        self.bn1   = norm_layer(planes)
        self.bn2   = norm_layer(planes)
        
        self.relu  = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        
        print('inplanes {}  planes {}'.format(inplanes, planes))
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out       
        
        
class resnet(nn.Module):
    
    def __init__(self, block, layers, in_chan=6, base_chan=36, num_classes=12, feat_size=5):
        super(resnet, self).__init__() 
        
        norm_layer = nn.BatchNorm1d
        
        planes = [ base_chan*2**i for i in range(feat_size) ]
        self.inplanes = planes[0]
          
        self.conv1 = nn.Conv1d(in_chan, base_chan, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1   = nn.BatchNorm1d(base_chan)
        self.relu  = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        
        self.layer1 = self._make_layer( block, planes[0], layers[0],           norm_layer=norm_layer )
        self.layer2 = self._make_layer( block, planes[1], layers[1], stride=2, norm_layer=norm_layer )
        self.layer3 = self._make_layer( block, planes[2], layers[2], stride=2, norm_layer=norm_layer )
        self.layer4 = self._make_layer( block, planes[3], layers[3], stride=2, norm_layer=norm_layer )
        self.layer5 = self._make_layer( block, planes[4], layers[3], stride=2, norm_layer=norm_layer )
        
        
        self.fc = nn.Linear( 2**feat_size*planes[3] * block.expansion , num_classes )
    
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)                       

                        # block,   32,     2,       2,
    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
            
        downsample=None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        
        self.inplanes = planes * block.expansion
        
        for _ in range( 1, blocks ):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
            
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
        

def resnet18(**kwargs):
    model = resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

















