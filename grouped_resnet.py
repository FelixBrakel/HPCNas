import torch.nn as nn
import torch

from primitives import Conv2d, Reduction_A, Stem, Reduction_B


class Grouped_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1):
        super(Grouped_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(
            in_channels, 32, 1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 32, 1, stride=1, padding=0,
                groups=groups, bias=False
            ),
            Conv2d(
                32, 32, 3, stride=1, padding=1,
                groups=groups, bias=False
            )
        )
        self.branch_2 = nn.Sequential(
            Conv2d(
                in_channels, 32, 1, stride=1, padding=0,
                groups=groups, bias=False
            ),
            Conv2d(
                32, 48, 3, stride=1, padding=1,
                groups=groups, bias=False
            ),
            Conv2d(
                48, 64, 3, stride=1, padding=1,
                groups=groups, bias=False
            )
        )
        self.conv = nn.Conv2d(
            128, 320, 1, stride=1, padding=0,
            groups=groups, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Grouped_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1):
        super(Grouped_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(
            in_channels, 192, 1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 128, 1, stride=1, padding=0,
                groups=groups, bias=False
            ),
            Conv2d(
                128, 160, (1, 7), stride=1, padding=(0, 3),
                groups=groups, bias=False
            ),
            Conv2d(
                160, 192, (7, 1), stride=1, padding=(3, 0),
                groups=groups, bias=False
            )
        )
        self.conv = nn.Conv2d(
            384, 1088, 1, stride=1, padding=0,
            groups=groups, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Grouped_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Grouped_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(
            in_channels, 192, 1, stride=1, padding=0,
            groups=groups, bias=False
        )
        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 192, 1, stride=1, padding=0,
                groups=groups, bias=False
            ),
            Conv2d(
                192, 224, (1, 3), stride=1, padding=(0, 1),
                groups=groups, bias=False
            ),
            Conv2d(
                224, 256, (3, 1), stride=1, padding=(1, 0),
                groups=groups, bias=False
            )
        )
        self.conv = nn.Conv2d(
            448, 2080, 1, stride=1, padding=0,
            groups=groups, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class GroupedResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=1000,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            k=256, l=256, m=384, n=384, groups=1):
        super(GroupedResNet, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(s0_depth):
            blocks.append(Grouped_ResNet_A(320, 0.17, groups))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(s1_depth):
            blocks.append(Grouped_ResNet_B(1088, 0.10, groups))
        blocks.append(Reduction_B(1088))
        for i in range(s2_depth):
            blocks.append(Grouped_ResNet_C(2080, 0.20, groups))
        blocks.append(Grouped_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(
            2080, 1536, 1, stride=1, padding=0,
            bias=False
        )
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
