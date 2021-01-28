import torch.nn as nn
from torchmeta.modules import MetaModule, MetaLinear

import sys
sys.path.append('..')

from global_utils import get_meta_backbone

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, backbone, in_features, num_ways):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_features = in_features
        self.num_ways = num_ways

        self.encoder = get_meta_backbone(backbone)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = MetaLinear(in_features, num_ways) #1600

    def forward(self, inputs, params=None):
        features = self.encoder(inputs, params=self.get_subdict(params, 'encoder'))          #50*5*5*64
        features = self.avg_pool(features)
        features = features.view((features.size(0), -1))                                  #50*1600
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits
