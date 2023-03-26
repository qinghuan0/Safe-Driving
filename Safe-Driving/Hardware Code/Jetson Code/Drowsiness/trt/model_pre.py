# -*- coding: utf-8 -*-
import tensorrt as trt
import torch
from ssd_net_vgg import *

# 加载 PyTorch 模型
net = SSD()
net.cuda()
net.eval()
net.load_state_dict(torch.load('./ssd.onnx'))

# 创建 TensorRT 构建器和优化器
trt_logger = trt.Logger(trt.Logger.WARNING)
trt_builder = trt.Builder(trt_logger)
trt_builder.max_batch_size = 1
trt_builder.max_workspace_size = 1 << 30 # 1GB

# 定义模型输入和输出的名称和形状
input_name = "input"
input_shape = (3, 300, 300)
output_name = "output"
output_shape = (8732, 4 + 1 + 80)

# 创建 TensorRT 模型优化器，并指定数据类型和输入形状
trt_network = trt_builder.create_network()
trt_input = trt_network.add_input(input_name, trt.float32, input_shape)
trt_output = trt_network.add_output(output_name, trt.float32, output_shape)

# 将 PyTorch 模型转换为 TensorRT 模型
trt_converter = trt_builder.create_network()
trt_input = trt_converter.add_input(input_name, trt.float32, input_shape)
trt_output = trt_converter.add_output(output_name, trt.float32, output_shape)
trt_torch = trt.TorchFallbackFunction.apply(net, trt_converter)

# 编译 TensorRT 模型，并保存引擎和网络参数到文件
trt_engine = trt_builder.build_cuda_engine(trt_network)
trt_engine.save_to_file("ssd300.trt")

with open("ssd300.pt", "wb") as f:
    torch.save(net.state_dict(), f)