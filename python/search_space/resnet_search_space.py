from naslib.search_spaces.core import Graph, primitives as ops
from naslib.search_spaces.core.graph import EdgeData
import torch
import torch.nn as nn

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




class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024


class Stem(nn.Module):
    """
    From https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/blob/master/model/inception_resnet_v2.py
    """
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels, 32, 3, stride=2, padding=0, bias=False), # 149 x 149 x 32
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

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)



class ResNetSearchSpace(Graph):
    """
    Search space for finding resnet variants
    """

    # Don't know what these do yet
    OPTIMIZER_SCOPE = [
        "n_stage_1",
    ]

    QUERYABLE = False

    def __init__(self, stage_1_cells=10, stage_2_cells=20):
        self.node_labels = {}
        super().__init__()

        stage_1_cell = self.construct_cell_stage_1()
        stage_1_cell.name = "c_1"

        stage_2_cell = self.construct_cell_stage_2()
        stage_2_cell.name = "c_2"
        self.node_idx = 1

        # Macro graph
        self.name = "resnetgraph"

        # input node
        self.add_node(self.node_idx)
        self.node_labels[self.node_idx] = "input"
        self.node_idx += 1

        # fist node is special
        self.add_node(self.node_idx)
        self.node_labels[self.node_idx] = "start"
        self.add_edge(self.node_idx-1, self.node_idx, op=Stem(3))
        self.edges[self.node_idx-1, self.node_idx].finalize()
        self.node_idx += 1

        # 10 stage 1 cells
        self.add_cells_with_skip(stage_1_cell, stage_1_cells)

        # Special intermediate node


        # 20 stage 2 cells
        self.add_cells_with_skip(stage_2_cell, stage_2_cells)

        # Another intermediate node

        # 10 more stage 2 cells
        self.add_cells_with_skip(stage_2_cell, 10)

        # postprocessing

    def add_cells_with_skip(self, cell: Graph, repetitions: int):
        """
        Add a number of repitition of a cell to the macro graph alongside a skip connection
        """
        for c in range(0, repetitions):
            # FIXME: This input node seems incorrect....
            self.add_node(self.node_idx, subgraph=cell.copy().set_input([self.node_idx]))
            self.add_edge(self.node_idx-1, self.node_idx, op=ops.Identity)
            self.node_labels[self.node_idx] = f"{cell.name} {c + 1}"
            self.edges[self.node_idx-1, self.node_idx].finalize()
            self.node_idx += 1

            # Merge cell output with skip connection
            self.add_node(self.node_idx)
            self.add_edge(self.node_idx-1, self.node_idx)
            self.add_edge(self.node_idx-2, self.node_idx)
            self.node_labels[self.node_idx] = f"add"
            self.node_idx += 1

            # Output node
            self.add_node(self.node_idx)
            self.add_edge(self.node_idx-1, self.node_idx)
            self.node_labels[self.node_idx] = "output"
            self.node_idx += 1

    @staticmethod
    def construct_cell_stage_1() -> Graph:
        cell = Graph()
        i = 1
        # input node
        cell.add_node(i)
        i += 1

        # 3 strands in a cell
        for _ in range(0, 2):
            cell.add_node(i)
            # cell.add_edge(1, i, op=Conv2d(320, 32, 1, 0))
            cell.add_edge(1, i, op=ops.MixedOp([ops.ConvBnReLU(320, 32, 1), ops.Identity()]))
            i += 1

        # strand 3 has 3 conv layers
        cell.add_node(i)
        cell.add_edge(3, i, op=ops.MixedOp([ops.Identity(), ops.ConvBnReLU()]))
        i += 1

        # combine operators
        cell.add_node(i, comb_op=lambda tensors: torch.cat(tensors, dim=1))
        # All strands lead to the post node
        cell.add_edges_from([(n, i, EdgeData({'op': Conv2d()})) for n in [1, 2, 4]])
        i += 1

        cell.add_node(i)
        cell.add_edge(i-1, i, op=ConvMul(128, 320, 1, 0))
        cell.edges[i-1, i].finalize()
        i += 1

        return cell

    @staticmethod
    def construct_cell_stage_2() -> Graph:
        cell = Graph()
        i = 1

        # input node
        cell.add_node(i)
        i += 1

        for _ in range(0, 2):
            cell.add_node(i)
            cell.add_edge(i-1, i)
            i += 1

        cell.add_node(i)
        cell.add_edges_from([n, i] for n in [1, i-1])
        i += 1

        cell.add_node(i)
        cell.add_edge(i-1, i)
        i += 1

        return cell


class EdgeData:
    def __init__(self, **kwargs):
        self.
