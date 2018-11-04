# -*- coding: utf-8 -*-
"""
https://github.com/ShichenLiu/CondenseNet/blob/master/models/condensenet.py
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from layers import Conv, LearnedGroupConv


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, group_1x1, group_3x3, bottleneck,
                condense_factor, dropout_rate):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = LearnedGroupConv(in_channels, bottleneck * growth_rate,
                                       kernel_size=1, groups=self.group_1x1,
                                       condense_factor=condense_factor,
                                       dropout_rate=dropout_rate)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, **kwargs):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, **kwargs)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet_1(nn.Module):
    def __init__(self, stages, growth, group_1x1, group_3x3, bottleneck,
                condense_factor, dropout_rate, num_classes=1000, model_type=1):

        super(CondenseNet_1, self).__init__()

        self.stages = stages
        self.growth = growth
        assert len(self.stages) == len(self.growth)
        self.progress = 0.0
        self.init_stride = 2
        self.pool_size = 3
        self.model_type = model_type

        self.features = nn.Sequential()
        ### Initial nChannels should be 1
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1
        self.features.add_module('init_conv', nn.Conv2d(1, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i, group_1x1=group_1x1, group_3x3=group_3x3, bottleneck=bottleneck,
                            condense_factor=condense_factor, dropout_rate=dropout_rate)
        ### Linear layer
        self.classifier = nn.Sequential(
            nn.Linear(4896, 1024),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i, **kwargs):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            **kwargs
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            if self.model_type == 1:
                self.features.add_module('pool_last',
                                        nn.AvgPool2d(self.pool_size))
            else:
                self.features.add_module('pool_last',
                                        nn.MaxPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


class CondenseNet_2(nn.Module):
    def __init__(self, stages, growth, group_1x1, group_3x3, bottleneck,
                condense_factor, dropout_rate, num_classes=1000, model_type=1):

        super(CondenseNet_2, self).__init__()

        self.stages = stages
        self.growth = growth
        assert len(self.stages) == len(self.growth)
        self.progress = 0.0
        self.init_stride = 2
        self.pool_size = 3
        self.model_type = model_type

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(1, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i, group_1x1=group_1x1, group_3x3=group_3x3, bottleneck=bottleneck,
                            condense_factor=condense_factor, dropout_rate=dropout_rate)
        ### Linear layer
        self.fc = nn.Linear(4896, 1024)
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1)
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1)
        )
        self.classifier4 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1)
        )
        # self.regression = nn.Conv2d(2064, 3, kernel_size=[2,1], stride=1, padding=0)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i, **kwargs):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            **kwargs
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            if self.model_type == 1:
                self.features.add_module('pool_last',
                                        nn.AvgPool2d(self.pool_size))
            else:
                self.features.add_module('pool_last',
                                        nn.MaxPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.fc(out)
        out1 = self.classifier1(out)
        out2 = self.classifier2(out)
        out3 = self.classifier3(out)
        out4 = self.classifier4(out)

        return out1, out2, out3, out4