"""参考https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py"""
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)

def conv3x3_1(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv3x3_2(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU()
    )

class Conv6(MetaModule):
    def __init__(self, in_channels=3, hid_channels=64):
        super(Conv6, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels

        self.encoder = MetaSequential(
            conv3x3_1(in_channels, hid_channels),             # 42*42
            conv3x3_1(hid_channels, hid_channels),            # 21*21
            conv3x3_1(hid_channels, hid_channels),            # 10*10
            conv3x3_1(hid_channels, hid_channels),             # 5*5
            conv3x3_2(hid_channels, hid_channels),
            conv3x3_2(hid_channels, hid_channels)
        )

    def forward(self, inputs, params=None):
        features = self.encoder(inputs, params=self.get_subdict(params, 'encoder'))
        return features


def conv6(**kwargs):
    return Conv6(**kwargs)


