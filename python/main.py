import logging

import torch
from naslib import utils
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, RegularizedEvolution, DrNASOptimizer
from naslib.search_spaces import (NasBench201SearchSpace, SimpleCellSearchSpace,
                                  NasBench301SearchSpace, NasBench101SearchSpace)
from naslib.search_spaces.core.graph import Graph
from search_space.resnet_search_space import ResNetSearchSpace
from naslib.utils import setup_logger
import networkx as nx
import matplotlib.pyplot as plt


def nas() -> Graph:
    config = utils.get_config_from_args()
    config.search.epochs = 10
    config.evaluation.epochs = 50
    # config.save_arch_weights = True
    # config.resume = True
    utils.set_seed(config.seed)
    utils.log_args(config)

    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.DEBUG)
    optimizer = DARTSOptimizer(**config.search)

    # search_space = SimpleCellSearchSpace(channels=[3, 16])
    search_space = ResNetSearchSpace()
    nx.draw_networkx(
        search_space,
        labels=search_space.node_labels,
        pos=nx.spiral_layout(search_space)
        # pos=nx.drawing.nx_agraph.graphviz_layout(search_space)
    )
    # cell = search_space.cell()
    # nx.draw_networkx(
    #     cell,
    #     pos=nx.drawing.nx_agraph.graphviz_layout(cell)
    # )
    plt.tight_layout()
    plt.show()
    # dataset_api = utils.get_dataset_api(config.search_space, config.dataset)
    # adapt the search space to the optimizer type
    # optimizer.adapt_search_space(search_space, config.dataset)

    # trainer = Trainer(optimizer, config)
    # trainer.search()
    # trainer.evaluate()

    # best = optimizer.get_final_architecture()
    # best.parse()
    # x = torch.zeros([64, 3, 3, 3])
    # logits = best(x)
    # best.eval()
    # torch.save(best.state_dict(), "./best_model.pth")
    # torch.onnx.export(best, x, "./best_model.onnx")
    # return best


def main():
    model = nas()


if __name__ == '__main__':
    main()
