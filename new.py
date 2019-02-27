from torch.autograd import Variable
import torch.onnx
import torchvision
import caffe2.python.onnx.backend as backend
import onnx
import numpy as np
import sys
# sys.path.append("./transfer")
# dummy_input = torch.randn(10, 3, 224, 224,device='cuda')
# model = torchvision.models.alexnet(pretrained=True).cuda()
# # model = torchvision.models.AlexNet().cuda()
#
# # It's optional to label the input and output layers
# # input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# # output_names = [ "output1" ]
#
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True)




# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))
# from onnx_caffe2.backend import Caffe2Backend
# model_name = 'Alexnet'
# init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(model.graph, device="CUDA")
# with open(model_name + "_init.pb", "wb") as f:
#     f.write(init_net.SerializeToString())
# with open(model_name + "_predict.pb", "wb") as f:
#     f.write(predict_net.SerializeToString())

rep = backend.prepare(model, device="CUDA:0")
outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
print(outputs[0])