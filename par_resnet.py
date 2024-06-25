import torch

from parnassia.search_space.primitives import *
import torch.nn as nn
import numpy as np


class CellA(nn.Module):
    out_channels = 64 + 32 + 32

    def __init__(self, in_channels):
        super(CellA, self).__init__()
        self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv2d(48, 64, 3, stride=1, padding=1, bias=False)
        )

        self.conv = nn.Conv2d(128, 320, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)

        return x_res


class CellB(nn.Module):
    out_channels = 192 + 192

    def __init__(self, in_channels):
        super(CellB, self).__init__()
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Conv2d(128, 160, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(160, 192, (7, 1), stride=1, padding=(3, 0), bias=False)
        )

        self.conv = nn.Conv2d(384, 1088, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)

        return x_res


class CellC(nn.Module):
    out_channels = 192 + 256

    def __init__(self, in_channels, activation=True):
        super(CellC, self).__init__()
        self.activation = activation
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 224, (1, 3), stride=1, padding=(0, 1), bias=False),
            Conv2d(224, 256, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(448, 2080, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)

        return x_res


class MacroStage(nn.Module):
    def __init__(self, cell, partitions: int, cell_channels: int, cell_scale: float):
        super(MacroStage, self).__init__()

        self.name = "MacroStageA"
        self.partitions = partitions
        self.cells = nn.ModuleList(cell(cell_channels) for _ in range(self.partitions))
        self.relu = nn.ReLU()
        self.cell_scale = cell_scale

    def forward(self, x):
        cell_out = self.cells[0](x)

        # for p in range(1, self.partitions):
        #     cell_out += self.cells[p](x)
        # cell_out = torch.cat(cell_out, dim=1)
        res = self.relu(cell_out * self.cell_scale + x)
        return res


class ParResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=1000,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            k=256, l=256, m=384, n=384, groups=1):
        super(ParResNet, self).__init__()

        self.stem = Stem(in_channels)

        self.stage1 = []
        for _ in range(s0_depth):
            self.stage1.append(MacroStage(CellA, groups, 320, 0.17))
        self.stage1 = nn.Sequential(*self.stage1)
        self.reduction1 = Reduction_A(320, k, l, m, n)

        self.stage2 = []
        for _ in range(s1_depth):
            self.stage2.append(MacroStage(CellB, groups, 1088, 0.1))
        self.stage2 = nn.Sequential(*self.stage2)
        self.reduction2 = Reduction_B(1088)

        self.stage3 = []
        for _ in range(s2_depth):
            self.stage3.append(MacroStage(CellC, groups, 2080, 0.2))
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


class MoEReductionA(nn.Module):
    def __init__(self, C_in, k, l, m, n):
        super(MoEReductionA, self).__init__()
        self.reduce = Conv2d(C_in, C_in//2, 1, stride=1, padding=0, bias=True)
        self.branch_0 = Conv2d(C_in//2, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(C_in//2, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        _x = self.reduce(x)
        x0 = self.branch_0(_x)
        x1 = self.branch_1(_x)
        x2 = self.branch_2(_x)
        return torch.cat((x0, x1, x2), dim=1)


class MoEReductionB(nn.Module):
    def __init__(self, C_in):
        super(MoEReductionB, self).__init__()
        self.reduce = Conv2d(C_in, C_in//2, 1, 0, 1, True)

        self.branch_0 = nn.Sequential(
            Conv2d(C_in//2, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 384, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(C_in//2, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(C_in//2, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=1, padding=1, bias=False),
            Conv2d(288, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        _x = self.reduce(x)
        x0 = self.branch_0(_x)
        x1 = self.branch_1(_x)
        x2 = self.branch_2(_x)
        x3 = self.branch_3(_x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class MoEResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=1000,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            partitions=2,
            k=256, l=256, m=384, n=384):
        super(MoEResNet, self).__init__()

        self.stem = CIFARStem(in_channels, 320)

        self.stage00 = []
        self.stage01 = []
        for _ in range(s0_depth):
            self.stage00.append(Inception_ResNet_A(320, 0.17))
            self.stage01.append(Inception_ResNet_A(320, 0.17))

        self.stage00 = nn.Sequential(*self.stage00)
        self.stage01 = nn.Sequential(*self.stage01)
        self.reduction0 = Reduction_A(640, k, l, m, n)

        self.stage10 = []
        self.stage11 = []

        for _ in range(s1_depth):
            self.stage10.append(Inception_ResNet_B(1408, 0.1))
            self.stage11.append(Inception_ResNet_B(1408, 0.1))

        self.stage10 = nn.Sequential(*self.stage10)
        self.stage11 = nn.Sequential(*self.stage11)

        self.reduction1 = Reduction_B(2816)

        self.stage20 = []
        self.stage21 = []

        for _ in range(s2_depth):
            self.stage20.append(Inception_ResNet_C(3808, 0.2))
            self.stage21.append(Inception_ResNet_C(3808, 0.2))

        self.stage20 = nn.Sequential(*self.stage20)
        self.stage21 = nn.Sequential(*self.stage21)

        self.conv = Conv2d(7616, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        _x = self.stem(x)

        _x0 = self.stage00(_x)
        _x1 = self.stage01(_x)

        _x = self.reduction0(torch.cat((_x0, _x1), dim=1))

        _x0 = self.stage10(_x)
        _x1 = self.stage11(_x)

        _x = self.reduction1(torch.cat((_x0, _x1), dim=1))

        _x0 = self.stage20(_x)
        _x1 = self.stage21(_x)
        _x = self.conv(torch.cat((_x0, _x1), dim=1))
        _x = self.global_average_pooling(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.linear(_x)

        return _x
