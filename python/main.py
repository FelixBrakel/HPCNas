import logging

import torch
from naslib import utils
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer
from naslib.search_spaces.core.graph import Graph
from search_space.demo_search_space import DemoSpace
from naslib.utils import setup_logger
import onnx


def nas(config) -> Graph:
    utils.set_seed(config.seed)
    utils.log_args(config)

    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.DEBUG)
    optimizer = DARTSOptimizer(**config.search)
    search_space = DemoSpace()
    optimizer.adapt_search_space(search_space, config.dataset)
    optimizer.before_training()

    trainer = Trainer(optimizer, config)
    trainer.search()
    trainer.evaluate()
    best = trainer.optimizer.get_final_architecture()

    return best


def main():
    config = utils.get_config_from_args()
    config.search.epochs = 10
    config.evaluation.epochs = 10
    config.save_arch_weights = False
    if torch.cuda.is_available():
        print("Found CUDA")
    else:
        print("No CUDA found, quitting")
        return -1
    model = nas(config)
#    model.eval()
    
    torch.set_default_device('cuda')
    input = torch.randn(1, 3, 32, 32)
    out = model(input)
    torch.onnx.export(model, input, "arch2.onnx")


if __name__ == '__main__':
    main()
