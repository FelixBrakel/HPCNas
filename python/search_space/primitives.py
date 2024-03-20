from naslib.search_spaces.core import Graph, primitives as ops
import torch.nn as nn
import torch


class Reduction_A(ops.AbstractPrimitive):
    def __init__(self, C_in, k, l, m, n):
        super().__init__(locals())
        self.branch_0 = Conv2d(C_in, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(C_in, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x, edge_data=None):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)

    def get_embedded_ops(self):
        return None


class Reduction_B(ops.AbstractPrimitive):
    def __init__(self, C_in):
        super().__init__(locals())
        self.branch_0 = nn.Sequential(
            Conv2d(C_in, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 384, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Conv2d(C_in, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(C_in, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=1, padding=1, bias=False),
            Conv2d(288, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x, edge_data=None):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)

    def get_embedded_ops(self):
        return None


class FactorizedReduce(ops.AbstractPrimitive):
    """
    Factorized reduce as used in ResNet to add some sort
    of Identity connection even though the resolution does not
    match.

    If the resolution matches it resolves to identity
    """

    def __init__(self, C_in, C_out, stride=1, affine=False, track_running_stats=False, **kwargs):
        super().__init__(locals())
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(
                        C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x, edge_data=None):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def get_embedded_ops(self):
        return None


class ConvMul(ops.AbstractPrimitive):
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, edge_data=None):
        return self.op(x) * 0.17

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class Conv2d(ops.AbstractPrimitive):
    def __init__(self, C_in, C_out, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__(locals())
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(C_out, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_data=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_embedded_ops(self):
        return None


class Stem(ops.AbstractPrimitive):
    """
    From https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/blob/master/model/inception_resnet_v2.py
    """
    def __init__(self, C_in):
        super(Stem, self).__init__(locals())
        self.features = nn.Sequential(
            Conv2d(C_in, 32, 3, stride=2, padding=0, bias=False), # 149 x 149 x 32
            Conv2d(32, 32, 3, stride=1, padding=0, bias=False), # 147 x 147 x 32
            Conv2d(32, 64, 3, stride=1, padding=1, bias=False), # 147 x 147 x 64
            nn.MaxPool2d(3, stride=2, padding=0), # 73 x 73 x 64
            Conv2d(64, 80, 1, stride=1, padding=0, bias=False), # 73 x 73 x 80
            Conv2d(80, 192, 3, stride=1, padding=0, bias=False), # 71 x 71 x 192
            nn.MaxPool2d(3, stride=2, padding=0), # 35 x 35 x 192
        )
        self.branch_0 = Conv2d(192, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(192, 48, 1, stride=1, padding=0, bias=False),
            Conv2d(48, 64, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(192, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
            Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(192, 64, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x, edge_data=None):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)

    def get_embedded_ops(self):
        return None
