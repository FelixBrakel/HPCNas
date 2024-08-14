import torch.nn as nn
import torch

from primitives import (Conv2d, SmallStem, Small_Reduction, Inception_ResNet_A,
                        Inception_ResNet_B, Inception_ResNet_C)


class Inception_ResNet_A_Half(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_A_Half, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(
            in_channels, 16, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 16, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                16, 16, 3,
                stride=1, padding=1, groups=groups, bias=False
            )
        )

        self.branch_2 = nn.Sequential(
            Conv2d(
                in_channels, 16, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                16, 24, 3,
                stride=1, padding=1, groups=groups, bias=False
            ),
            Conv2d(
                24, 32, 3,
                stride=1, padding=1, groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            64, in_channels, 1,
            stride=1, padding=0, groups=groups, bias=True
        )
        if activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Inception_ResNet_B_Half(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_B_Half, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(
            in_channels, 96, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 64, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                64, 80, (1, 7),
                stride=1, padding=(0, 3), groups=groups, bias=False
            ),
            Conv2d(
                80, 96, (7, 1),
                stride=1, padding=(3, 0), groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            192, in_channels, 1,
            stride=1, padding=0, groups=groups, bias=True
        )
        if self.activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Inception_ResNet_C_Half(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_C_Half, self).__init__()
        self.scale = scale
        self.activation = activation

        self.branch_0 = Conv2d(
            in_channels, 96, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 96, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                96, 112, (1, 3),
                stride=1, padding=(0, 1), groups=groups, bias=False
            ),
            Conv2d(
                112, 128, (3, 1),
                stride=1, padding=(1, 0), groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            224, in_channels, 1,
            stride=1, padding=0, groups=groups, bias=True
        )
        if self.activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res



class BaseResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=100,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            k=256, l=256, m=384, n=384, groups=1):
        super(BaseResNet, self).__init__()
        groups = 1
        blocks = []
        blocks.append(SmallStem(in_channels, 80))
        for i in range(s0_depth):
            blocks.append(Inception_ResNet_A_Half(80, 0.17, groups))
        blocks.append(Small_Reduction(80, 272))
        for i in range(s1_depth):
            blocks.append(Inception_ResNet_B_Half(272, 0.10, groups))
        blocks.append(Small_Reduction(272, 520))
        for i in range(s2_depth - 1):
            blocks.append(Inception_ResNet_C_Half(520, 0.20, groups))
        if s2_depth > 0:
            blocks.append(Inception_ResNet_C_Half(520, scale=0.20, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(
            520, 384, 1, stride=1, padding=0,
            bias=False
        )
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(384, classes)

    def forward(self, x):
        _x = self.features(x)
        _x = self.conv(_x)
        _x = self.global_average_pooling(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.dropout(_x)
        _x = self.linear(_x)
        return _x

    

