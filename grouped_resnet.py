import torch
import torch.nn as nn
from parnassia.search_space.primitives import Conv2d, Reduction_A


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
           Conv2d(
                in_channels, 32, 3,
                stride=2, padding=0, bias=False
            ), # 149 x 149 x 32
            Conv2d(
                32, 32, 3,
                stride=1, padding=0, bias=False
            ), # 147 x 147 x 32
            Conv2d(
                32, 64, 3,
                stride=1, padding=1, bias=False
            ), # 147 x 147 x 64
            nn.MaxPool2d(3, stride=2, padding=0), # 73 x 73 x 64
            Conv2d(
                64, 80, 1,
                stride=1, padding=0, bias=False
            ), # 73 x 73 x 80
            Conv2d(
                80, 192, 3,
                stride=1, padding=0, bias=False
            ), # 71 x 71 x 192
            nn.MaxPool2d(3, stride=2, padding=0), # 35 x 35 x 192
        )
        self.branch_0 = Conv2d(
            192, 96, 1,
            stride=1, padding=0, bias=False
        )
        self.branch_1 = nn.Sequential(
            Conv2d(
                192, 48, 1,
                stride=1, padding=0, bias=False
            ),
            Conv2d(
                48, 64, 5,
                stride=1, padding=2, bias=False
            ),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(
                192, 64, 1,
                stride=1, padding=0, bias=False
            ),
            Conv2d(
                64, 96, 3,
                stride=1, padding=1, bias=False
            ),
            Conv2d(
                96, 96, 3,
                stride=1, padding=1, bias=False
            ),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(
                192, 64, 1,
                stride=1, padding=0, bias=False
            )
        )

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class CIFARStem(nn.Module):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_in=3, C_out=64):
        super(CIFARStem, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False), nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.seq(x)


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
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
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Conv2d(128, 160, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(160, 192, (7, 1), stride=1, padding=(3, 0), bias=False)
        )
        self.conv = nn.Conv2d(384, 1088, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduction_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 384, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=1, padding=1, bias=False),
            Conv2d(288, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 224, (1, 3), stride=1, padding=(0, 1), bias=False),
            Conv2d(224, 256, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(448, 2080, 1, stride=1, padding=0, bias=True)
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
            blocks.append(Inception_ResNet_A(320, 0.17, groups))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(s1_depth):
            blocks.append(Inception_ResNet_B(1088, 0.10, groups))
        blocks.append(Reduction_B(1088))
        for i in range(s2_depth):
            blocks.append(Inception_ResNet_C(2080, 0.20, groups))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
