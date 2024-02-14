from naslib.search_spaces.core import Graph, primitives as ops


class ResNetSearchSpace(Graph):
    """
    Search space for finding resnet variants
    """

    # Don't know what these do yet
    OPTIMIZER_SCOPE = [
        "n_stage_1",
    ]

    QUERYABLE = False

    def __init__(self, max_cells=10, min_cells=10):
        super().__init__()

        normal_cell = self.cell()

        self.name = "resnetgraph"

        # input node
        # TODO: set up network input
        self.add_node(1)

        # TODO: add ReLu between cells or at the end of every cell
        for i in range(1, min_cells + 1):
            # Cells are connected in a feed-forward manner, skip connections are incorporated in
            # the cells themselves
            self.add_node(i + 1, subgraph=normal_cell.copy().set_input([i]))
            self.add_edge(i, i + 1)
            self.edges[i, i + 1].set("op", ops.Identity)


        # TODO: allow the search to vary the amount of cells.


    def cell(self):
        cell = Graph()

        # input node
        cell.add_node(1)

        # functional operators in the cell
        for i in range(2, 7):
            cell.add_node(i)

        # postprocessing of functional operators
        cell.add_node(7)

        # All operators are connected to the post node
        cell.add_edges_from([(i, 7) for i in range(2, 7)])

        # output node
        cell.add_node(8)

        # Skip connection
        cell.add_edge(1, 8)

        # TODO:
        # Decide on op for connecting postprocessing node
        cell.add_edge(7, 8)

        return cell


def _set_cell_ops(current_edge_data, C, stride):
    pass
