# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from torch import nn
from torchvision.models import resnet18
from torchvision.models import resnet50

from timm.models.registry import register_model

RESNET_TYPES = {'resnet18':resnet18, 'resnet50':resnet50}

__all__ = ["resnet_multi_channel"]

class ResnetCompressBlock(nn.Module):
    def __init__(self, num_input_channels, middle_channel_multiplier=3, num_output_channels=3):
        super().__init__()
        
        self.num_input_channels = num_input_channels
        self.middle_channel_multiplier = middle_channel_multiplier
        self.num_output_channels = num_output_channels
        
        num_middle_channels = middle_channel_multiplier * self.num_input_channels
        
        self.conv1 = nn.Conv2d(num_input_channels, num_middle_channels,
                               kernel_size=7, stride=1, padding=3, bias=False,
                               groups=self.num_input_channels)
        self.bn1 = nn.BatchNorm2d(num_middle_channels)
        self.relu = nn.ReLU(inplace=True)
                
        self.conv2 = nn.Conv2d(num_middle_channels, self.num_output_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(self.num_output_channels)
        
        self.identity_mapping = nn.Conv2d(self.num_input_channels, self.num_output_channels,
                                          stride=1, padding=0, kernel_size=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.identity_mapping(x)
        out = self.relu(out)
        
        return out

class ResnetMultiChannel(nn.Module):

    def __init__(self, num_input_channels=18, resnet_type='resnet18', num_classes=512):
        super().__init__()

        assert resnet_type in RESNET_TYPES, f"Unknown resnet type '{resnet_type}'"

        self.compress_block = ResnetCompressBlock(num_input_channels)

        resnet_class = RESNET_TYPES[resnet_type]
        self.resnet = resnet_class(num_classes=num_classes)
        
    def forward(self, x):
        x = self.compress_block(x)
        x = self.resnet(x)
        return x

@register_model
def resnet_multi_channel(**kwargs):
    encoder = ResnetMultiChannel(**kwargs)
    return encoder
