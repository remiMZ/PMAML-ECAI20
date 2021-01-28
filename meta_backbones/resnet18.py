'''
https://github.com/Sha-Lab/FEAT/blob/master/model/networks/res18.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                        MetaBatchNorm2d, MetaLinear)


__all__ = ['resnet10', 'resnet18']

def conv3x3(in_planes, out_planes, stride=1):
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                    padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                    bias=False)

class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample    # residual branch downsample block
        self.stride = stride

    def forward(self, inputs, params=None):
        identity = inputs

        outputs = self.conv1(inputs, params=self.get_subdict(params, 'conv1'))
        outputs = self.bn1(outputs, params=self.get_subdict(params, 'bn1'))
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs, params=self.get_subdict(params, 'conv2'))
        outputs = self.bn2(outputs, params=self.get_subdict(params, 'bn2'))

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)

        return outputs


class Bottleneck(MetaModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = MetaBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs, params=None):
        identity = inputs

        outputs = self.conv1(inputs, params=self.get_subdict(params, 'conv1'))
        outputs = self.bn1(outputs, params=self.get_subdict(params, 'bn1'))
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs, params=self.get_subdict(params, 'conv2'))
        outputs = self.bn2(outputs, params=self.get_subdict(params, 'bn2'))
        outputs = self.relu(outputs)

        outputs = self.conv3(outputs, params=self.get_subdict(params, 'conv3'))
        outputs = self.bn3(outputs, params=self.get_subdict(params, 'bn3'))

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)

        return outputs


class ResNet(MetaModule):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = MetaConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_conv()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = MetaSequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                MetaBatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return MetaSequential(*layers)

    def _init_conv(self):
        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs, params=None):
        outputs = self.conv1(inputs,params=self.get_subdict(params, 'conv1'))
        outputs = self.bn1(outputs, params=self.get_subdict(params, 'bn1'))
        outputs = self.relu(outputs)

        outputs = self.layer1(outputs, params=self.get_subdict(params, 'layer1'))
        outputs = self.layer2(outputs, params=self.get_subdict(params, 'layer2'))
        outputs = self.layer3(outputs, params=self.get_subdict(params, 'layer3'))
        outputs = self.layer4(outputs, params=self.get_subdict(params, 'layer4'))

        # outputs = self.avgpool(outputs)

        return outputs


def resnet10(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


