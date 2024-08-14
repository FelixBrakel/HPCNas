import torch.nn as nn

from primitives import Conv2d, Stem, Reduction_A, Reduction_B, Inception_ResNet_A, Inception_ResNet_B, Inception_ResNet_C


class MacroStage(nn.Module):
    def __init__(self, cell, partitions: int, cell_channels: int, cell_scale: float, activation=True):
        super(MacroStage, self).__init__()
        self.activation = activation
        self.partitions = partitions
        self.cells = nn.ModuleList(cell(cell_channels, activation=False) for _ in range(self.partitions))
        if self.activation:
            self.relu = nn.ReLU()
        self.cell_scale = cell_scale

    def forward(self, x):
        cell_out = self.cells[0](x)

        for p in range(1, self.partitions):
            cell_out += self.cells[p](x)
        if self.activation:
            return self.relu(x + self.cell_scale * cell_out)
        return x + self.cell_scale * cell_out


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

        self.stem = Stem(in_channels, 320)

        self.stage1 = []
        for _ in range(s0_depth):
            self.stage1.append(MacroStage(Inception_ResNet_A, groups, 320, 0.17))
        self.stage1 = nn.Sequential(*self.stage1)
        self.reduction1 = Reduction_A(320, k, l, m, n)
        # self.reduction1 = Small_Reduction(320, 1088)

        self.stage2 = []
        for _ in range(s1_depth):
            self.stage2.append(MacroStage(Inception_ResNet_B, groups, 1088, 0.1))
        self.stage2 = nn.Sequential(*self.stage2)
        self.reduction2 = Reduction_B(1088)
        # self.reduction2 = Small_Reduction(1088, 2080)

        self.stage3 = []
        for _ in range(s2_depth - 1):
            self.stage3.append(MacroStage(Inception_ResNet_C, groups, 2080, 0.2))
        self.stage3.append(MacroStage(Inception_ResNet_C, groups, 2080, 0.2, activation=False))
        self.stage3 = nn.Sequential(*self.stage3)

        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
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
        _x = self.dropout(_x)
        _x = self.linear(_x)

        return _x
    

