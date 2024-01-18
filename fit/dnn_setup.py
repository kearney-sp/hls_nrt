import torch
import torch.nn as nn

# Build resnet from scratch for regression, replacing convolutions and pooling with dense layers
# adapted from https://www.mdpi.com/1099-4300/24/7/876
# adapted from https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = None):
        super(ResidualBlock, self).__init__()
        self.dense1 = nn.Sequential(
                        nn.Linear(in_channels, out_channels, bias=False),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.dense2 = nn.Sequential(
                        nn.Linear(out_channels, out_channels, bias=False),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.dense1(x)
        out = self.dense2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetRegressor(nn.Module):
    def __init__(self, block, layers, n_inputs=1):
        super(ResNetRegressor, self).__init__()
        self.inplanes = 64
        self.dense1 = nn.Sequential(
                        nn.Linear(n_inputs, 64, bias=False),
                        nn.BatchNorm1d(64),
                        nn.ReLU())
        self.layer0 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 128, layers[1])
        self.layer2 = self._make_layer(block, 256, layers[2])
        self.layer3 = self._make_layer(block, 512, layers[3])
        self.fc = nn.Linear(512, 1)
        
    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.inplanes != planes:   
            downsample = nn.Sequential(
                nn.Linear(self.inplanes, planes, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x