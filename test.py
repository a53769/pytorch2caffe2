import vnet
import onnx
import sys
from torch.autograd import Variable
import torch.onnx
import torch as t
import torch.nn as nn
import torchvision
model = vnet.VNet(elu=True, nll=False)
model = nn.parallel.DataParallel(model)
dummy_input = torch.randn(1, 1, 64, 224, 224,device='cuda')
state = t.load('vnet_checkpoint.pth')['state_dict']
model.load_state_dict(state)

model.train(False)
model = model.module.cuda()

torch.onnx.export(model, dummy_input, "vnet.onnx", verbose=True)

model = onnx.load("vnet.onnx")

# Check that the IR is well formed
# onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

# import caffe2.python.onnx.backend as c2
from onnx_caffe2.backend import Caffe2Backend
model_name = 'Vnet'
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(model.graph, device="CUDA")
with open(model_name + "_init.pb", "wb") as f:
    f.write(init_net.SerializeToString())
with open(model_name + "_predict.pb", "wb") as f:
    f.write(predict_net.SerializeToString())