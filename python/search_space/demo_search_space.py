# from naslib.search_spaces.core import Graph
# from naslib.search_spaces.core import primitives as ops
from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
from naslib.search_spaces.core.graph import EdgeData

from search_space.primitives import *
import torch
import torch.nn as nn


class CellA(Graph):
    def __init__(self):
        super().__init__()
        self.name = "cell A"

        # input node
        input_node = self.append_node()

        # 3 strands
        concat_node = self.append_node(comb_op=lambda tensors: torch.cat(tensors, dim=1))
        s2 = self.append_node()
        s3_1 = self.append_node()

        self.add_edges_from(
            [(
                input_node, s,
                EdgeData(data={
                    'op': [
                        ops.ConvBnReLU,
                        FactorizedReduce
                    ],
                    'C_in': 320,
                    'C_out': 32,
                    'kernel_size': 1,
                    'stride': 1,
                    'affine': False,
                    'track_running_stats': False
                })
            ) for s in [concat_node, s2, s3_1]]
        )

        # strand 3 has 3 conv layers
        s3_2 = self.append_node()
        self.add_edge(
            s3_1, s3_2,
            op=[
                ops.ConvBnReLU,
                FactorizedReduce
            ],
            C_in=32, C_out=48, kernel_size=3, stride=1, affine=False, track_running_stats=False
        )

        # All strands lead to the post node
        self.add_edge(
            s2, concat_node,
            op=[
                ops.ConvBnReLU,
                FactorizedReduce
            ],
            C_in=32, C_out=32, kernel_size=3, stride=1, affine=False, track_running_stats=False
        )
        self.add_edge(
            s3_2, concat_node,
            op=[
                ops.ConvBnReLU,
                FactorizedReduce
            ],
            C_in=48, C_out=64, kernel_size=3, stride=1, affine=False, track_running_stats=False
        )

        out = self.append_node()

        self.add_edge_final(
            concat_node, out,
            op=ConvMul(128, 320, 1, 1)
        )


class CellB(Graph):
    def __init__(self):
        super().__init__()

        self.name = "cell B"
        # input node
        in_node = self.append_node()

        s2_1 = self.append_node()
        self.add_edge(
            in_node, s2_1,
            op=[
                ops.ConvBnReLU
            ],
            C_in=1088, C_out=128, kernel_size=1
        )

        s2_2 = self.append_node()
        self.add_edge(
            s2_1, s2_2,
            op=[
                Conv2d
            ],
            C_in=128, C_out=160, kernel_size=(1, 7), stride=1, padding=(0, 3)
        )

        concat = self.append_node(comb_op=lambda tensors: torch.cat(tensors, dim=1))
        self.add_edge(
            s2_2, concat,
            op=[
                Conv2d
            ],
            C_in=160, C_out=192, kernel_size=(7, 1), stride=1, padding=(3, 0)
        )

        self.add_edge(
            in_node, concat,
            op=[
                Conv2d
            ],
            C_in=1088, C_out=192, kernel_size=1, padding=0
        )

        out_node = self.append_node()
        self.add_edge_final(
            concat, out_node,
            op=ConvMul(384, 1088, 1)
        )


class CellC(Graph):
    def __init__(self):
        super().__init__()
        self.name = "cell C"
        # input node
        in_node = self.append_node()

        s2_1 = self.append_node()
        self.add_edge(
            in_node, s2_1,
            op=[
                ops.ConvBnReLU
            ],
            C_in=2080, C_out=192, kernel_size=1
        )

        s2_2 = self.append_node()
        self.add_edge(
            s2_1, s2_2,
            op=[
                Conv2d
            ],
            C_in=192, C_out=224, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )

        concat = self.append_node(comb_op=lambda tensors: torch.cat(tensors, dim=1))
        self.add_edge(
            s2_2, concat,
            op=[
                Conv2d
            ],
            C_in=224, C_out=256, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )

        self.add_edge(
            in_node, concat,
            op=[
                Conv2d
            ],
            C_in=2080, C_out=192, kernel_size=1, padding=0
        )

        out_node = self.append_node()
        self.add_edge_final(
            concat, out_node,
            op=ConvMul(448, 2080, 1)
        )


class DemoSpace(Graph):
    OPTIMIZER_SCOPE = [
        "s1",
        "s2",
        "s3"
    ]

    QUERYABLE = False

    def __init__(self):
        super().__init__()
        self.node_idx = 1
        self.node_labels = {}

        # Macro graph
        self.name = "DemoGraph"

        # input node
        input_node = self.append_node()

        # Stem
        stem = self.append_node()
        self.add_edge_final(input_node, stem, op=ops.Stem(3, 320))

        s1_output = self.add_par_cell_with_skip(CellA(), stem, "s1", 2, 2)

        reduce_a = self.append_node()

        self.add_edge_final(
            s1_output, reduce_a,
            op=Reduction_A(320, 256, 256, 384, 384)
        )

        s2_output = self.add_par_cell_with_skip(CellB(), reduce_a, "s2", 4, 2)

        reduce_b = self.append_node()

        self.add_edge_final(
            s2_output, reduce_b,
            op=Reduction_B(1088)
        )

        s3_output = self.add_par_cell_with_skip(CellC(), reduce_b, "s3", 2, 2)

        # Post
        post_node = self.append_node()

        self.add_edge_final(s3_output, post_node, op=ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2080, 10)
        ))

        self.compile()

    def add_cells_with_skip(self, cell: Graph, input_idx: int, scope: str, repetitions: int) -> int:
        """
        Add a number of repitition of a cell to the macro graph alongside a skip connection
        :param cell: Search space cell to insert
        :param input_idx: input to the first cell in the sequence
        :param scope: optimizer scope
        :param repetitions: number of cells in the sequence
        :return: index of the sink node
        """
        first = True
        for c in range(0, repetitions):
            # Add the cell
            if first:
                cell_node = self.append_node(subgraph=cell.set_input([input_idx]).set_scope(scope))
                first = False
            else:
                cell_node = self.append_node(subgraph=cell.copy().set_input([input_idx]).set_scope(scope))
            self.add_edge_final(input_idx, cell_node)

            # Merge cell output with skip connection
            merge_node = self.append_node()
            self.add_edge_final(cell_node, merge_node)
            self.add_edge_final(input_idx, merge_node)

            # Output node
            output_node = self.append_node()
            self.add_edge_final(merge_node, output_node, op=ops.Sequential(nn.ReLU()))

            input_idx = output_node

        return output_node

    def add_par_cell_with_skip(self, cell: Graph, input_idx: int, scope: str, repetitions: int, par_degree: int) -> int:
        """
        Add a number of parallel repitition of a cell to the macro graph alongside a skip connection
        :param cell: Search space cell to insert
        :param input_idx: input to the first cell in the sequence
        :param scope: optimizer scope
        :param repetitions: number of cells in the sequence
        :param par_degree: degree of cell parallelism
        :return: index of the sink node
        """
        first = True
        for r in range(0, repetitions):
            # Add the parallel cells
            cells = []
            for p in range(0, par_degree):
                if first:
                    cells.append(self.append_node(subgraph=cell.set_input([input_idx]).set_scope(scope), name=f"{cell.name}r{r}p{p}"))
                    first = False
                else:
                    cell_cpy = cell.copy()
                    cell_cpy.name = "cell_cpy"
                    cells.append(self.append_node(subgraph=cell_cpy.set_input([input_idx]).set_scope(scope), name=f"{cell.name}r{r}p{p}"))

            # Merge node
            merge_node = self.append_node(name=f"add {scope} r{r}")

            for c in cells:
                self.add_edge_final(input_idx, c)

                # Merge cell output with skip connection
                self.add_edge_final(c, merge_node)

            # Output node
            output_node = self.append_node(name=f"out {scope}r{r}")

            self.add_edge_final(input_idx, merge_node)
            self.add_edge_final(merge_node, output_node, op=ops.Sequential(nn.ReLU()))

            input_idx = output_node

        return output_node

    @staticmethod
    def construct_cell_stage_1() -> Graph:
        cell = Graph()
        cell.name = "cell"

        # input node
        input_node = cell.append_node()

        # 3 strands
        concat_node = cell.append_node(comb_op=lambda tensors: torch.cat(tensors, dim=1))
        s2 = cell.append_node()
        s3_1 = cell.append_node()

        cell.add_edges_from(
            [(
                input_node,
                s,
                EdgeData(data={
                    'op': [
                        ops.ConvBnReLU,
                        FactorizedReduce
                    ],
                    'C_in': 64,
                    'C_out': 32,
                    'kernel_size': 1,
                    'stride': 1,
                    'affine': False,
                    'track_running_stats': False
                })
            ) for s in [concat_node, s2, s3_1]]
        )

        # strand 3 has 3 conv layers
        s3_2 = cell.append_node()
        cell.add_edge(
            s3_1, s3_2,
            op=[
                ops.ConvBnReLU,
                FactorizedReduce
            ],
            C_in=32, C_out=48, kernel_size=3, stride=1, affine=False, track_running_stats=False
        )

        # All strands lead to the post node
        cell.add_edge(
            s2,
            concat_node,
            op=[
                ops.ConvBnReLU,
                FactorizedReduce
            ],
            C_in=32, C_out=32, kernel_size=3, stride=1, affine=False, track_running_stats=False
        )
        cell.add_edge(
            s3_2,
            concat_node,
            op=[
                ops.ConvBnReLU,
                FactorizedReduce
            ],
            C_in=48, C_out=64, kernel_size=3, stride=1, affine=False, track_running_stats=False
        )

        out = cell.append_node()

        cell.add_edge_final(
            concat_node,
            out,
            op=ConvMul(128, 64, 1, 1)
        )
        return cell

    @staticmethod
    def construct_cell_stage_2() -> Graph:
        pass

    @staticmethod
    def construct_cell_stage_3() -> Graph:
        pass

    def prepare_discretization(self):
        pass

    def prepare_evaluation(self):
        pass

    def get_hash(self):
        return "No hash for you"
