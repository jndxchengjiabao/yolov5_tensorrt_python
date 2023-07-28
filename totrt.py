import tensorrt as trt
import argparse


def GiB(val):
    return val * 1 << 30  # 左移运算符


def ONNX_build_engine(onnx_file_path, out_file_path, write_engine=True, batch_size=5, imgsz=512, inputname="images"):
    '''
    通过加载onnx文件，构建engine
    :param onnx_file_path: onnx文件路径
    :return: engine
    '''
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 1、动态输入第一点必须要写的
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    batch_size = batch_size  # trt推理时最大支持的batchsize
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network,
                                                                                                             G_LOGGER) as parser:
        builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(2)  # common文件可以自己去tensorrt官方例程下面找
        config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        # 重点
        profile = builder.create_optimization_profile()  # 动态输入时候需要 分别为最小输入、常规输入、最大输入
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        profile.set_shape(inputname, (1, 3, imgsz, imgsz), (1, 3, imgsz, imgsz), (batch_size, 3, imgsz, imgsz))
        config.add_optimization_profile(profile)

        network.get_input(0).shape = [1, 3, 640, 640]

        engine = builder.build_serialized_network(network, config)
        print("Completed creating Engine")
        # 保存engine文件
        if write_engine:
            engine_file_path = out_file_path
            with open(engine_file_path, "wb") as f:
                f.write(engine)
                print("Completed write Engine")
        return engine

def parse_args():
    """
    处理脚本参数
    """
    parser = argparse.ArgumentParser(description='检查TRT模型')
    parser.add_argument('--onnx_path', default='yolov5s.onnx', help='onnx模型路径', type=str)
    parser.add_argument('--trt_path', default='yolov5s.trt', help='TRT模型路径', type=str)
    parser.add_argument('--image_size', default=640, help='图像尺寸，如640', type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()
    ONNX_build_engine(opt.onnx_path, opt.trt_path, write_engine=True, batch_size=1, imgsz=opt.image_size, inputname="images")
