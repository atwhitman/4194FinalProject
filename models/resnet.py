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
        
        super(BasicBlock, self).__init__()
        
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
        
        
        
class resnet(nn.Module):
    
    def __init__(self, block, layers, in_chan=6, feat_size=5, base_chan=32, num_classes=12):
        
        self.inplanes = base_chan
        
        super(resnet, self).__init__()
        
        norm_layer = nn.BatchNorm1d
        
        
        self.conv1 = nn.Conv1d(in_chan, base_chan, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1   = nn.BatchNorm1d(base_chan)
        self.relu  = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        
    
        self.layer1 = self._make_layer( block, base_chan,   layers[0] )

        self.layer2 = self._make_layer( block, base_chan*2, layers[1] )

        self.layer3 = self._make_layer( block, base_chan*4, layers[2] )

        self.layer4 = self._make_layer( block, base_chan*8, layers[3] )

        self.fc = nn.Linear( base_chan*8 * feat_size**2, num_classes )
    
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.weight, 0)                       
#                 n = m.kernel_size * m.out_channels
#                 m.weight.data.normal_( 0, math.sqrt(2.0 / n) )
                    
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes*block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        
        for i in range( 1, blocks ):
            layers.append( block(self.inplanes, planes, stride, downsample))
            
        return nn.Sequential(*layers)
            
            
    def forward(x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
        

def resnet18(**kwargs):
    model = resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

















