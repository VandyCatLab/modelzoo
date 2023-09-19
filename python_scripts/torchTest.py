import tensorflow as tf
import torch
from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras
import numpy as np
from PIL import Image
from torchvision import transforms
import datasets

'''''
t_model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device':'cpu', 'num_classes': 200})

t_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

t_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
'''''

t_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)


shape = [3, 256, 256]
shape.insert(0,1)
rand_dims = tuple(shape)
#t_model.eval()
input_np = np.random.uniform(0, 1, rand_dims)
print(input_np.shape)
input_var = Variable(torch.FloatTensor(input_np))

'''
dummy_input = Variable(torch.randn(input_size))
torch.onnx.export(model, dummy_input, "onnx_test.onnx", opset_version=11, )
model = onnx.load("onnx_test.onnx")
'''
'''
tf_rep = prepare(model)
tf_rep.export_graph("onnx_test_pb")
model_pb = tf.keras.models.load_model("onnx_test_pb")
tf.keras.models.save_model(model_pb, "onnx_test_pb/onnx_test_h5.h5", save_format='h5')
'''
shape.pop(0)
k_dims = tuple(shape)
k_model = pytorch_to_keras(t_model, input_var, [k_dims], verbose=True, name_policy='renumerate')
k_model.summary()

