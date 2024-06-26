import torch

from parnassia.search_space.primitives import *


class SplitReductionA(nn.Module):
    def __init__(self, C_in, k, l, m, n):
        super(SplitReductionA, self).__init__()
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


class SplitReductionB(nn.Module):
    def __init__(self, C_in):
        super(SplitReductionB, self).__init__()
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


class SplitCellA(nn.Module):
    def __init__(self, groups=1, scale=0.17):
        super(SplitCellA, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(320//groups, 32//groups, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(320//groups, 32//groups, 1, stride=1, padding=0, bias=False),
            Conv2d(32//groups, 32//groups, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Conv2d(320//groups, 32//groups, 1, stride=1, padding=0, bias=False),
            Conv2d(32//groups, 48//groups, 3, stride=1, padding=1, bias=False),
            Conv2d(48//groups, 64//groups, 3, stride=1, padding=1, bias=False)
        )

        self.conv = nn.Conv2d(128//groups, 320//groups, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)

        return self.relu(x_res * self.scale + x)


class SplitCellB(nn.Module):
    def __init__(self, groups=1, scale=0.1):
        super(SplitCellB, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(1088//groups, 192//groups, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(1088//groups, 128//groups, 1, stride=1, padding=0, bias=False),
            Conv2d(128//groups, 160//groups, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(160//groups, 192//groups, (7, 1), stride=1, padding=(3, 0), bias=False)
        )

        self.conv = nn.Conv2d(384//groups, 1088//groups, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)

        return self.relu(x_res * self.scale + x)


class SplitCellC(nn.Module):
    def __init__(self, groups=1, scale=0.2):
        super(SplitCellC, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(2080//groups, 192//groups, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(2080//groups, 192//groups, 1, stride=1, padding=0, bias=False),
            Conv2d(192//groups, 224//groups, (1, 3), stride=1, padding=(0, 1), bias=False),
            Conv2d(224//groups, 256//groups, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(448//groups, 2080//groups, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)

        return self.relu(x_res * self.scale + x)


class SplitStage(nn.Module):
    def __init__(self, cell, depth: int, partitions: int, cell_scale: float):
        super(SplitStage, self).__init__()

        self.name = f"SplitStage{cell}"
        cells = [cell(partitions, cell_scale) for _ in range(depth)]
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
            k=256, l=256, m=384, n=384):
        super(SplitResNet, self).__init__()
        self.groups = groups
        self.stem = Stem(in_channels)
        self.s0_partitions = nn.ModuleList(
            [SplitStage(SplitCellA, s0_depth, groups, 0.17) for _ in range(groups)]
        )

        self.reduction0 = Reduction_A(320, k, l, m, n)

        self.s1_partitions = nn.ModuleList(
            [SplitStage(SplitCellB, s1_depth, groups, 0.1) for _ in range(groups)]
        )

        self.reduction1 = Reduction_B(1088)

        self.s2_partitions = nn.ModuleList(
            [SplitStage(SplitCellC, s2_depth, groups, 0.2) for _ in range(groups)]
        )

        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
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
        _x = self.linear(_x)

        return _x
