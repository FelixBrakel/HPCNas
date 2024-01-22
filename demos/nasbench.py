import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor)
from naslib.search_spaces import (
    NasBench301SearchSpace,
    SimpleCellSearchSpace,
    NasBench201SearchSpace,
    HierarchicalSearchSpace)

from naslib import utils
from naslib.utils import setup_logger


def main():
    config = utils.get_config_from_args()
    # config.save_arch_weights = True
    # config.resume = True
    utils.set_seed(config.seed)
    utils.log_args(config)


    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.INFO)
    optimizers = {
        "darts": DARTSOptimizer(config),
        "gdas": GDASOptimizer(config),
        "drnas": DrNASOptimizer(config),
        "rs": RandomSearch(config),
        "re": RegularizedEvolution(config),
        "ls": LocalSearch(config),
        "bananas": Bananas(config),
        "bp": BasePredictor(config)
    }

    optimizer = optimizers["re"]

    search_space = NasBench201SearchSpace()

    dataset_api = utils.get_dataset_api(config.search_space, config.dataset)
    # adapt the search space to the optimizer type
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    trainer = Trainer(optimizer, config)
    # trainer.search()
    # trainer.evaluate(dataset_api=dataset_api)
    # arch = optimizer.get_final_architecture()
    # script = torch.jit.script(arch.parse())


    # checkpointer = utils.Checkpointer(optimizer.get_checkpointables().pop("model"))
    # extra = checkpointer.load('../model_final.pth')
    # print(checkpointer)

    # if not config.eval_only:
    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
    trainer.search(resume_from=checkpoint)
    # else:
    # checkpoint = utils.get_last_checkpoint(config, search=False) if config.resume else ""
    trainer.evaluate(resume_from=checkpoint, dataset_api=dataset_api)

    best = optimizer.get_final_architecture()
    best.parse()
    torch.save(best, "../result.pth")
    return

if __name__ == '__main__':
    main()
