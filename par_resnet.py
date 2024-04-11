import torch

from parnassia.search_space.primitives import *
import torch.nn as nn
import numpy as np


class MacroStage(nn.Module):
    def __init__(self, cell, partitions: int, cell_channels: int, stem_scale: int):
        super(MacroStage, self).__init__()

        self.name = "MacroStageA"
        self.partitions = partitions
        self.cells = nn.ModuleList(cell(cell_channels) for _ in range(self.partitions))
        self.relu = nn.ReLU(inplace=True)
        self.stem_scale = stem_scale

    def forward(self, x):
        cell_out = []
        for p in range(self.partitions):
            cell_out.append(self.cells[p](x))
        cell_out = torch.stack(cell_out)
        return self.relu(torch.sum(cell_out, dim=0) + x * self.stem_scale)


class ParResNet(nn.Module):
    def __init__(self, in_channels=3, classes=1000, k=256, l=256, m=384, n=384):
        super(ParResNet, self).__init__()

        self.stem = CIFARStem(in_channels, 320)

        self.stage1 = []
        for _ in range(5):
            self.stage1.append(MacroStage(Inception_ResNet_A, 2, 320, 0.17))
        self.stage1 = nn.Sequential(*self.stage1)
        self.reduction1 = Reduction_A(320, k, l, m, n)

        self.stage2 = []
        for _ in range(10):
            self.stage2.append(MacroStage(Inception_ResNet_B, 2, 1088, 0.1))
        self.stage2 = nn.Sequential(*self.stage2)
        self.reduction2 = Reduction_B(1088)

        self.stage3 = []
        for _ in range(5):
            self.stage3.append(MacroStage(Inception_ResNet_C, 2, 2080, 0.2))
        self.stage3 = nn.Sequential(*self.stage3)

        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        _x = self.stem(x)
        _x = self.stage1(_x)
        _x = self.reduction1(_x)
        _x = self.stage2(_x)
        _x = self.reduction2(_x)
        _x = self.stage3(_x)

        _x = self.conv(_x)
        _x = self.global_average_pooling(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.linear(_x)


        return _x