import logging

import numpy
import torch
from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.torch.model import PyTorchModel
from naslib import utils
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, RegularizedEvolution, DrNASOptimizer
from naslib.search_spaces import (NasBench201SearchSpace, SimpleCellSearchSpace,
                                  NasBench301SearchSpace, NasBench101SearchSpace)
from naslib.search_spaces.core.graph import Graph
from search_space.ff_compat import FFCompatCellSearchSpace
from naslib.utils import setup_logger


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
    search_space = FFCompatCellSearchSpace()
    # dataset_api = utils.get_dataset_api(config.search_space, config.dataset)
    # adapt the search space to the optimizer type
    optimizer.adapt_search_space(search_space, config.dataset)

    trainer = Trainer(optimizer, config)
    trainer.search()
    trainer.evaluate()

    best = optimizer.get_final_architecture()
    best.parse()
    x = torch.zeros([64, 3, 3, 3])
    logits = best(x)
    best.eval()
    # torch.save(best.state_dict(), "./best_model.pth")
    # torch.onnx.export(best, x, "./best_model.onnx")
    return best


def ff_run_model(model):
    ffconfig = FFConfig()

    ffmodel = FFModel(ffconfig)

    dims_input = [ffconfig.batch_size, 3, 64, 64]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)
    torch_model = PyTorchModel(model)
    output_tensors = torch_model.torch_to_ff(ffmodel, [input_tensor, input_tensor])

    # output_tensors = file_to_ff("model_final.pth", ffmodel, [input_tensor, input_tensor])
    # t = ffmodel.softmax(output_tensors[0])

    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )
    label_tensor = ffmodel.label_tensor

    num_samples = 10000

    (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = y_train.astype('float32')

    dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
    dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)

    num_samples = dataloader_input.num_samples

    ffmodel.init_layers()

    layers = ffmodel.get_layers()
    for layer in layers:
        print(layers[layer].name)

    # layer = ffmodel.get_layer_by_name("relu_1")
    # print(layer)

    epochs = ffconfig.epochs

    ts_start = ffconfig.get_current_time()

    ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

    ts_end = ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start)
    print(
        "epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %
        (epochs, run_time, num_samples * epochs / run_time)
    )


def main():
    model = nas()
    ff_run_model(model)


if __name__ == '__main__':
    init_flexflow_runtime()
    main()
