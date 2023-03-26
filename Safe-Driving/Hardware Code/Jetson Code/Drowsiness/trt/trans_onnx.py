import torch
from ssd_net_vgg import *

# Load the model
model = SSD()
model.eval()

# Create dummy inputs
input_shape = (1, 3, 300, 300)
dummy_input = torch.randn(input_shape).to('cuda:0')

# Move the model parameters to the GPU
model.to('cuda:0')

# Convert the PyTorch model to ONNX
model_input = dummy_input.to('cuda:0')
model.to('cuda:0')
torch.onnx.export(model, model_input, "ssd.onnx", verbose=True)