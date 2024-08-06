import torch.nn as nn
import torch

from primitives import Conv2d, Reduction_A, Stem, Reduction_B, Inception_ResNet_A, Inception_ResNet_B, Inception_ResNet_C


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=100,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            k=256, l=256, m=384, n=384, groups=1):
        super(ResNet, self).__init__()
        groups = 1
        blocks = []
        blocks.append(Stem(in_channels, 320))
        for i in range(s0_depth):
            blocks.append(Inception_ResNet_A(320, 0.17, groups))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(s1_depth):
            blocks.append(Inception_ResNet_B(1088, 0.10, groups))
        blocks.append(Reduction_B(1088, 256, 288, 320, 256, 384))
        for i in range(s2_depth - 1):
            blocks.append(Inception_ResNet_C(2080, 0.20, groups))
        if s2_depth > 0:
            blocks.append(Inception_ResNet_C(2080, scale=0.20, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(
            2080, 1536, 1, stride=1, padding=0,
            bias=False
        )
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        _x = self.features(x)
        _x = self.conv(_x)
        _x = self.global_average_pooling(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.dropout(_x)
        _x = self.linear(_x)
        return _x
