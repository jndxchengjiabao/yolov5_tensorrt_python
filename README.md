# yolov5_tensorrt_python

使用tensorrt和numpy进行加速推理，不依赖pytorch，不需要导入其他依赖

安装：

1.需要安装tensorrt python版

2.安装pycuda

步骤：

1.将yolov5官方代码训练好的.py模型转化为.onnx模型

$ python export.py --weights yolov5s.pt --include onnx

2.将.onnx转换为.trt模型用于加速推理

$ python totrt.py --onnx_path yolov5s.onnx --trt_path yolov5s.trt

3.修改trt.py中categories里面的类别为自己的类别,num_classes为自己的类别数量，engine_file_path改为自己trt模型的路径

4.运行trt.py或者更改demo里面的文件路径就可以使用


参考：

https://github.com/wang-xinyu/tensorrtx

https://github.com/ultralytics/yolov5



