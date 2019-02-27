import torchvision
import torch
import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

dummy_input = torch.randn(10, 3, 224, 224,device='cuda')
model = torchvision.models.inception_v3(pretrained=True).cuda()
torch.onnx.export(model, dummy_input, "inception_v3.onnx", verbose=True)

onnx_model = onnx.load("inception_v3.onnx")
# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

rep = backend.prepare(onnx_model, device="CUDA:0")
outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))

print(outputs[0])