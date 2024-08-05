import torch.nn as nn
import torch

class Conv2d(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            padding, stride=1, groups=1, bias=True
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _x = self.conv(x)

        _x = self.bn(_x)
        _x = self.relu(_x)

        return _x


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


class SmallStem(nn.Module):
    def __init__(self, C_in, C_out):
        super(SmallStem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(
                C_in, 32, 3,
                stride=2, padding=0, bias=False
            ),  # 149 x 149 x 32
            Conv2d(
                32, 64, 3,
                stride=1, padding=0, bias=False
            ),  # 147 x 147 x 32
            nn.MaxPool2d(3, stride=2, padding=0),  # 73 x 73 x 64
            Conv2d(
                64, C_out, 3,
                stride=1, padding=0, bias=False
            ),  # 71 x 71 x C_out
            nn.MaxPool2d(3, stride=2, padding=0)  # 35 x 35 x C_out
        )

    def forward(self, x):
        return self.features(x)


class Stem(nn.Module):
    def __init__(self, C_in, C_out=320):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(
                C_in, 32, 3,
                stride=2, padding=0, bias=False
            ),  # 149 x 149 x 32
            Conv2d(
                32, 32, 3,
                stride=1, padding=0, bias=False
            ),  # 147 x 147 x 32
            Conv2d(
                32, 64, 3,
                stride=1, padding=1, bias=False
            ),  # 147 x 147 x 64
            nn.MaxPool2d(3, stride=2, padding=0), # 73 x 73 x 64
            Conv2d(
                64, 80, 1,
                stride=1, padding=0, bias=False
            ),  # 73 x 73 x 80
            Conv2d(
                80, 192, 3,
                stride=1, padding=0, bias=False
            ),  # 71 x 71 x 192
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
        if C_out != 320:
            self.out = nn.Conv2d(320, C_out, 1, stride=1, padding=0, bias=True)
        else:
            self.out = None

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        _x = torch.cat((x0, x1, x2, x3), dim=1)
        if self.out:
            return self.out(_x)
        return _x


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(
            in_channels, 32, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 32, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                32, 32, 3,
                stride=1, padding=1, groups=groups, bias=False
            )
        )

        self.branch_2 = nn.Sequential(
            Conv2d(
                in_channels, 32, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                32, 48, 3,
                stride=1, padding=1, groups=groups, bias=False
            ),
            Conv2d(
                48, 64, 3,
                stride=1, padding=1, groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            128, in_channels, 1,
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


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(
            in_channels, 192, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 128, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                128, 160, (1, 7),
                stride=1, padding=(0, 3), groups=groups, bias=False
            ),
            Conv2d(
                160, 192, (7, 1),
                stride=1, padding=(3, 0), groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            384, in_channels, 1,
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


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, groups=1, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation

        self.branch_0 = Conv2d(
            in_channels, 192, 1,
            stride=1, padding=0, groups=groups, bias=False
        )

        self.branch_1 = nn.Sequential(
            Conv2d(
                in_channels, 192, 1,
                stride=1, padding=0, groups=groups, bias=False
            ),
            Conv2d(
                192, 224, (1, 3),
                stride=1, padding=(0, 1), groups=groups, bias=False
            ),
            Conv2d(
                224, 256, (3, 1),
                stride=1, padding=(1, 0), groups=groups, bias=False
            )
        )

        self.conv = nn.Conv2d(
            448, in_channels, 1,
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


class Small_Reduction(nn.Module):
    def __init__(self, C_in, C_out):
        super(Small_Reduction, self).__init__()
        self.conv = Conv2d(C_in, C_out - C_in, 3, stride=2, padding=0, bias=False)
        self.pool = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.pool(x)
        return torch.cat((x0, x1), dim=1)

class Reduction_A(nn.Module):
    def __init__(self, C_in, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(C_in, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(C_in, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)


class Reduction_B(nn.Module):
    def __init__(self, in_channels, k=256, l=288, m=320, x=256, y=384):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, x, 1, stride=1, padding=0, bias=False),
            Conv2d(x, y, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)
