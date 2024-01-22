import flexflow.serve as serve
import flexflow.core as ff
from flexflow.onnx.model import ONNXModel


def main():
    serve.init(
        num_gpus=1,
        memory_per_gpu=2800,
        zero_copy_memory_per_node=8000,
        tensor_parallelism_degree=1,
        pipeline_parallelism_degree=1
    )

    ffconfig = ff.FFConfig()
    # ffconfig = ff.NetConfig()
    ffmodel = ff.FFModel(ffconfig)
    # print(alexnetconfig.dataset_path)
    onnx_model = ONNXModel("alexnet_Opset18.onnx")
    dims_input = [ffconfig.batch_size, 3, 229, 229]
    input_tensor = ffmodel.create_tensor(dims_input, ff.DataType.DT_FLOAT)
    _t = onnx_model.apply(ffmodel, {"input.1": input_tensor})

    # ffmodel.compile()
    # ffmodel.forward()
    # ffmodel.generate()
    # TODO:
    #  generate sampling config
    #  compile model for inference and load weights into memory
    #  run inference

    # ff.init_flexflow_runtime() ffconfig = ff.FFConfig() alexnetconfig = ff.NetConfig() print(
    # f"Python API batchSize({ffconfig.batch_size}) workersPerNodes({ffconfig.workers_per_node})
    # numNodes({ffconfig.num_nodes})") ffmodel = ff.FFModel(ffconfig)

    # dims_input = [[ffconfig.batch_size, 3, 229, 229]]
    # input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    # t = onnx_model.apply(ffmodel, {"input.1": input})

    # ff.init()


if __name__ == '__main__':
    main()
