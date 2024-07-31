import torch.nn as nn
import torch

from primitives import Conv2d, Stem, Reduction_A, Reduction_B


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
