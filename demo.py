import torch
from torch.autograd import Variable
import torch.onnx
import onnx
import caffe2.python.onnx.backend as backend
import numpy as np
import torchvision
# from onnx_tf.backend import prepare


import sys
sys.path.append("./transfer")
dummy_input = torch.randn(10, 3, 224, 224,device='cuda')
model = torchvision.models.vgg11(pretrained=True).cuda()
# model = torchvision.models.AlexNet().cuda()

# It's optional to label the input and output layers
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "vgg11.onnx", verbose=True)

#no error here
onnx_model = onnx.load("vgg11.onnx")
# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

rep = backend.prepare(onnx_model, device="CUDA:0")
outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))

print(outputs[0])
