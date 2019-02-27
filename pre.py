import vnet
import onnx

from torch.autograd import Variable
import torch.onnx
import torch as t
import torch.nn as nn
import torchvision
import caffe2.python.onnx.backend as backend
import numpy as np
# model = vnet.VNet(elu=True, nll=False)
# model = nn.parallel.DataParallel(model)
# dummy_input = torch.randn(1, 1, 64, 224, 224,device='cuda')
# state = t.load('vnet_checkpoint.pth')['state_dict']
# model.load_state_dict(state)
#
# model.train(False)
# model = model.module.cuda()
#
# torch.onnx.export(model, dummy_input, "vnet_o.onnx", verbose=True)

model = onnx.load("vnet.onnx")

onnx.checker.check_model(model)
rep = backend.prepare(model, device="CUDA:0") # or "CPU"
# For the Caffe2 backend:
#     rep.predict_net is the Caffe2 protobuf for the network
#     rep.workspace is the Caffe2 workspace for the network
#       (see the class caffe2.python.onnx.backend.Workspace)
outputs = rep.run(np.random.randn(1, 1, 64, 224, 224).astype(np.float32))
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.
print(outputs[0])