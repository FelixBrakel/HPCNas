from naslib.search_spaces.core import Graph, primitives as ops
import networkx as nx

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
        self.name = "resnetgraph"

        stage_2_cell = self.construct_cell_stage_2()
        stage_2_cell.name = "c_2"
        self.node_idx = 1
        # input node
        # TODO: set up network input
        self.add_node(self.node_idx)
        self.node_labels[self.node_idx] = "input"
        self.node_idx += 1

        # fist node is special
        self.add_node(self.node_idx)
        self.add_edge(self.node_idx-1, self.node_idx)
        self.node_labels[self.node_idx] = "start"
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
            self.add_node(self.node_idx, subgraph=cell.copy().set_input([self.node_idx]))
            self.add_edge(self.node_idx-1, self.node_idx)
            self.node_labels[self.node_idx] = f"{cell.name} {c + 1}"
            self.edges[self.node_idx-1, self.node_idx].set("op", ops.Identity)
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


    def construct_cell_stage_1(self) -> Graph:
        cell = Graph()
        i = 1
        # input node
        cell.add_node(i)
        i += 1

        # 3 strands in a cell
        for _ in range(0, 2):
            cell.add_node(i)
            cell.add_edge(1, i)

            i += 1

        # strand 3 has 3 conv layers
        cell.add_node(i)
        cell.add_edge(3, i)
        i += 1

        # combine operators
        cell.add_node(i)
        # All strands lead to the post node
        cell.add_edges_from([(n, i) for n in [1, 2, 4]])
        i += 1

        cell.add_node(i)
        cell.add_edge(i-1, i)
        i += 1

        return cell

    def construct_cell_stage_2(self) -> Graph:
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

def _set_cell_ops(current_edge_data, C, stride):
    pass
