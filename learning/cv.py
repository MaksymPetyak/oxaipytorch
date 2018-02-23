import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

torch.__file__

from inception import inception_v3
# import imp
# import inception
# imp.reload(inception)
# net = torchvision.models.vgg19_bn(pretrained=True)
# net = torchvision.models.inception_v3(pretrained=True, transform_input=False)
# net = torchvision.models.resnet152(pretrained=True)

net = inception_v3(pretrained=True)

net.eval();

from demjson import decode
classes = decode(open("imagenet1000_clsid_to_human.txt", "r").read())

from PIL import Image

img_size=299
transform = transforms.Compose(
	[transforms.Resize(img_size),
	transforms.RandomCrop(img_size),
	transforms.ToTensor(),
	# transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
	transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

	# transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

%matplotlib inline
def imshow(img):
	img = img*0.5 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))

# image = Image.open('img/beacon.png')
image = Image.open('img/sunglasses.jpg')
image = transform(image)
imshow(image)

# image.size()
# image.unsqueeze(0).size()
# image = image[:-1,:,:]
# len(result)
result=net(Variable(image.unsqueeze(0),requires_grad=False))
result = result.data.tolist()
result = list(enumerate(result[0]))
result = sorted(result,key=lambda x: -x[1])
list(map(lambda x: (x[0],classes[x[0]], x[1]), result[:10]))

net.second_last_layer(Variable(image.unsqueeze(0),requires_grad=False))

##################

import pretrainedmodels
model_name = 'inceptionv4' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

model.eval()
net=model
import pretrainedmodels.utils as utils

load_img = utils.LoadImage()

tf_img = utils.TransformImage(model)

path_img = 'img/cat.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
    requires_grad=False)

result = model(input) # 1x1000
result = result.tolist()
result = list(enumerate(result[0]))
result = sorted(result,key=lambda x: -x[1])
list(map(lambda x: (x[0],classes[x[0]], x[1]), result[:10]))
# list(map(lambda x: (x[0],key_to_classname[class_id_to_key[x[0]]], x[1]), result[:10]))
# max, argmax = result.data.squeeze().max(0)
# class_id = argmax[0]
# classes[class_id]
# imshow(input_img)

        # Load Imagenet Synsets
# with open('imagenet_synsets.txt', 'r') as f:
#     synsets = f.readlines()
#
# # len(synsets)==1001
# # sysnets[0] == background
# synsets = [x.strip() for x in synsets]
# splits = [line.split(' ') for line in synsets]
# key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}
#
# with open('imagenet_classes.txt', 'r') as f:
#     class_id_to_key = f.readlines()
#
# class_id_to_key = [x.strip() for x in class_id_to_key]
