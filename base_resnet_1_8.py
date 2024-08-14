import torch.nn as nn
import torch

from primitives import (Conv2d, SmallStem, Small_Reduction, Inception_ResNet_A,
                        Inception_ResNet_B, Inception_ResNet_C)


class Inception_ResNet_A_Eighth(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_A_Eighth, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(
            in_channels, 4, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 4, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                4, 4, 3,
                stride=1, padding=1, groups=groups, bias=False
            )
        )

        self.branch_2 = nn.Sequential(
            Conv2d(
                in_channels, 4, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                4, 6, 3,
                stride=1, padding=1, groups=groups, bias=False
            ),
            Conv2d(
                6, 8, 3,
                stride=1, padding=1, groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            16, in_channels, 1,
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


class Inception_ResNet_B_Eighth(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_B_Eighth, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(
            in_channels, 24, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 16, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                16, 20, (1, 7),
                stride=1, padding=(0, 3), groups=groups, bias=False
            ),
            Conv2d(
                20, 24, (7, 1),
                stride=1, padding=(3, 0), groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            48, in_channels, 1,
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


class Inception_ResNet_C_Eighth(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_C_Eighth, self).__init__()
        self.scale = scale
        self.activation = activation

        self.branch_0 = Conv2d(
            in_channels, 24, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 24, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                24, 28, (1, 3),
                stride=1, padding=(0, 1), groups=groups, bias=False
            ),
            Conv2d(
                28, 32, (3, 1),
                stride=1, padding=(1, 0), groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            56, in_channels, 1,
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



class BaseResNetEighth(nn.Module):
    def __init__(
            self,
            in_channels=3,
            classes=100,
            s0_depth=10,
            s1_depth=20,
            s2_depth=10,
            k=256, l=256, m=384, n=384, groups=1):
        super(BaseResNetEighth, self).__init__()
        groups = 1
        blocks = []
        blocks.append(SmallStem(in_channels, 40))
        for i in range(s0_depth):
            blocks.append(Inception_ResNet_A_Eighth(40, 0.17, groups))
        blocks.append(Small_Reduction(40, 136))
        for i in range(s1_depth):
            blocks.append(Inception_ResNet_B_Eighth(136, 0.10, groups))
        blocks.append(Small_Reduction(136, 260))
        for i in range(s2_depth - 1):
            blocks.append(Inception_ResNet_C_Eighth(260, 0.20, groups))
        if s2_depth > 0:
            blocks.append(Inception_ResNet_C_Eighth(260, scale=0.20, activation=False))
        self.features = nn.Sequential(*blocks)
        # self.conv = Conv2d(
        #     520, 384, 1, stride=1, padding=0,
        #     bias=False
        # )
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(260, classes)

    def forward(self, x):
        _x = self.features(x)
        _x = self.global_average_pooling(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.dropout(_x)
        _x = self.linear(_x)
        return _x

    

