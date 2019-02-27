import onnx
import torch.onnx

import onnx_caffe2.backend as backend
import torch as t
import vnet
import torch.nn as nn
import numpy as np
#
# model = vnet.VNet(elu=True, nll=False)
# model = nn.parallel.DataParallel(model)
# dummy_input = torch.randn(1, 1, 64, 224, 224,device='cuda').float()
# model = t.load('vnet_checkpoint.pth')['model']
# # model.load_state_dict(state)
# # model.dump_patch = True
# model.eval()
# #
# model = model.cuda()
# #
# torch.onnx.export(model, dummy_input, "vnet.onnx", verbose=True)
# # # #
model = onnx.load("vnet.onnx")
onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))
# rep = backend.prepare(model, device="CUDA")
# outputs = rep.run(np.random.randn(1, 1, 64, 224, 224).astype(np.float32))
# print(outputs[0])
out = backend.run_model(model, np.random.randn(1, 1, 64, 224, 224).astype(np.float32))
print(out)