import onnx
import netron

# Load the ONNX model
model = onnx.load("ssd.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a summary of the model
print(onnx.helper.printable_graph(model.graph))

netron.start("ssd.onnx")

