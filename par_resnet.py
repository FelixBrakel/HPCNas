import torch

from parnassia.search_space.primitives import *
import torch.nn as nn
import numpy as np


class CellA(nn.Module):
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
    def __init__(self, in_channels):
        super(CellC, self).__init__()
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
    def __init__(self, cell, partitions: int, cell_channels: int, cell_scale: float, activation=True):
        super(MacroStage, self).__init__()
        self.activation = activation
        self.partitions = partitions
        self.cells = nn.ModuleList(cell(cell_channels) for _ in range(self.partitions))
        if self.activation:
            self.relu = nn.ReLU(inplace=True)
        self.cell_scale = cell_scale

    def forward(self, x):
        cell_out = self.cells[0](x)

        for p in range(1, self.partitions):
            cell_out += self.cells[p](x)
        if self.activation:
            return self.relu(x + self.cell_scale * cell_out)
        return x + self.cell_scale * cell_out


class ParMacroStage(nn.Module):
    def __init__(self, cell, partitions: int, cell_channels: int, cell_scale: float, activation=True):
        super(ParMacroStage, self).__init__()
        self.activation = activation
        self.partitions = partitions
        self.cells = nn.ModuleList()
        self.cell_scale = cell_scale
        self.devices = [
            torch.device(f'cuda:{i % torch.cuda.device_count()}') for i in range(self.partitions)
        ]
        # Distribute cells across available GPUs
        for i in range(self.partitions):
            self.cells.append(cell(cell_channels).to(self.devices[i]))
        if self.activation:
            self.relu = nn.ReLU()

    def forward(self, x):
        print(x.device)
        cell_out = self.cells[0](x)
        
        for p in range(1, self.partitions):
            x_device = x.to(self.devices[p])
            cell_out = cell_out + self.cells[p](x_device).to(self.devices[0])

        res = cell_out * self.cell_scale + x
        if self.activation:
            res = self.relu(res)
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
        for _ in range(s2_depth - 1):
            self.stage3.append(MacroStage(CellC, groups, 2080, 0.2))
        self.stage3.append(MacroStage(CellC, groups, 2080, 0.2, activation=False))
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
    

class ParParResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=1000,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            k=256, l=256, m=384, n=384, groups=1):
        super(ParParResNet, self).__init__()

        self.stem = Stem(in_channels)

        self.stage1 = []
        for _ in range(s0_depth):
            self.stage1.append(ParMacroStage(CellA, groups, 320, 0.17))
        self.stage1 = nn.Sequential(*self.stage1)
        self.reduction1 = Reduction_A(320, k, l, m, n)

        self.stage2 = []
        for _ in range(s1_depth):
            self.stage2.append(ParMacroStage(CellB, groups, 1088, 0.1))
        self.stage2 = nn.Sequential(*self.stage2)
        self.reduction2 = Reduction_B(1088)

        self.stage3 = []
        for _ in range(s2_depth):
            self.stage3.append(ParMacroStage(CellC, groups, 2080, 0.2))
        self.stage3 = nn.Sequential(*self.stage3)

        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        print(x.device)
        _x = self.stem(x)
        print(_x.device)
        _x = self.stage1(_x)
        print(_x.device)
        _x = self.reduction1(_x)
        _x = self.stage2(_x)
        _x = self.reduction2(_x)
        _x = self.stage3(_x)

        _x = self.conv(_x)
        _x = self.global_average_pooling(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.linear(_x)

        return _x
