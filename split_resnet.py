import torch.nn as nn
import torch

from primitives import SmallStem, Small_Reduction, Conv2d


class SplitCellA(nn.Module):
    def __init__(self, in_channels, groups=1, scale=0.17):
        super(SplitCellA, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(
            in_channels//groups, 32//groups, 1, stride=1, padding=0,
            bias=False
        )
        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels//groups, 32//groups, 1, stride=1, padding=0,
                bias=False
            ),
            Conv2d(
                32//groups, 32//groups, 3, stride=1, padding=1,
                bias=False
            )
        )
        self.branch_2 = nn.Sequential(
            Conv2d(
                in_channels//groups, 32//groups, 1, stride=1, padding=0,
                bias=False
            ),
            Conv2d(
                32//groups, 48//groups, 3, stride=1, padding=1,
                bias=False
            ),
            Conv2d(
                48//groups, 64//groups, 3, stride=1, padding=1,
                bias=False
            )
        )

        self.conv = nn.Conv2d(
            128//groups, in_channels//groups, 1, stride=1, padding=0,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)

        return self.relu(x_res * self.scale + x)


class SplitCellB(nn.Module):
    def __init__(self, in_channels, groups=1, scale=0.1):
        super(SplitCellB, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels//groups, 192//groups, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels//groups, 128//groups, 1, stride=1, padding=0, bias=False),
            Conv2d(128//groups, 160//groups, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(160//groups, 192//groups, (7, 1), stride=1, padding=(3, 0), bias=False)
        )

        self.conv = nn.Conv2d(384//groups, in_channels//groups, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)

        return self.relu(x_res * self.scale + x)


class SplitCellC(nn.Module):
    def __init__(self, in_channels, groups=1, scale=0.2, activation=True):
        super(SplitCellC, self).__init__()
        self.scale = scale
        self.activation = activation

        self.branch_0 = Conv2d(in_channels//groups, 192//groups, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels//groups, 192//groups, 1, stride=1, padding=0, bias=False),
            Conv2d(192//groups, 224//groups, (1, 3), stride=1, padding=(0, 1), bias=False),
            Conv2d(224//groups, 256//groups, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(448//groups, in_channels//groups, 1, stride=1, padding=0, bias=True)
        if self.activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x_res * self.scale + x)
        return x_res * self.scale + x


class SplitStage(nn.Module):
    def __init__(self, cell, in_channels, depth: int, partitions: int, cell_scale: float):
        super(SplitStage, self).__init__()

        self.name = f"SplitStage{cell}"
        if cell == SplitCellC:
            cells = [cell(in_channels, partitions, cell_scale) for _ in range(depth - 1)]
            cells.append(cell(in_channels, partitions, cell_scale, activation=False))
        else:
            cells = [cell(in_channels, partitions, cell_scale) for _ in range(depth)]

        self.features = nn.Sequential(*cells)

    def forward(self, x):
        return self.features(x)


class SplitResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=1000,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            groups=1,
            k=128, l=128, m=192, n=192):
        super(SplitResNet, self).__init__()
        self.groups = groups
        self.stem = SmallStem(in_channels, 320)

        self.s0_partitions = nn.ModuleList(
            [SplitStage(SplitCellA, 320, s0_depth, groups, 0.17) for _ in range(groups)]
        )

        # self.reduction0 = Reduction_A(160, k, l, m, n)
        self.reduction0 = Small_Reduction(320, 1088)
        self.s1_partitions = nn.ModuleList(
            [SplitStage(SplitCellB, 1088, s1_depth, groups, 0.1) for _ in range(groups)]
        )

        # self.reduction1 = Reduction_B(544, 128, 144, 160, 128, 192)
        self.reduction1 = Small_Reduction(1088, 2080)
        self.s2_partitions = nn.ModuleList(
            [SplitStage(SplitCellC, 2080, s2_depth, groups, 0.2) for _ in range(groups)]
        )

        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        _x = self.stem(x)

        s0_out = []
        for partition, xs in zip(self.s0_partitions, _x.chunk(self.groups, dim=1)):
            s0_out.append(partition(xs))

        _x = self.reduction0(torch.cat(s0_out, dim=1))

        s1_out = []
        for partition, xs in zip(self.s1_partitions, _x.chunk(self.groups, dim=1)):
            s1_out.append(partition(xs))

        _x = self.reduction1(torch.cat(s1_out, dim=1))

        s2_out = []
        for partition, xs in zip(self.s2_partitions, _x.chunk(self.groups, dim=1)):
            s2_out.append(partition(xs))

        _x = self.conv(torch.cat(s2_out, dim=1))
        _x = self.global_average_pooling(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.dropout(_x)
        _x = self.linear(_x)

        return _x
