import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from torchvision.datasets import ImageFolder
import resnet18 as resnet

torch.cuda.set_device(0)  # 设置GPU ID
is_cuda = True
simple_transform = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),  # H, W, C -> C, W, H 归一化到(0,1)，简单直接除以255
                                       transforms.Normalize([0.485, 0.456, 0.406],  # std
                                                            [0.229, 0.224, 0.225])])

# mean  先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
# 使用 ImageFolder 必须有对应的目录结构
train = ImageFolder("../toolwearclass/train_valid_test/train", simple_transform)
valid = ImageFolder("../toolwearclass/train_valid_test/valid", simple_transform)
train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=5)
val_loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=5)

vgg=resnet.resnetcnn(3,2)
pthfile = r'H:\研究生期间\毕设相关\毕设代码\resnet_model1.pth'
vgg.load_state_dict(torch.load(pthfile))


# 提取不同层输出的 主要代码
class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


print(vgg.features)

conv_out = LayerActivations(vgg.features, 0)  # 提出第 一个卷积层的输出
img = next(iter(train_loader))[0]

# imshow(img)
o = vgg(Variable(img.cuda()))
conv_out.remove()  #
act = conv_out.features  # act 即 第0层输出的特征

# 可视化 输出
fig = plt.figure(figsize=(20, 50))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(act[0][i].detach().numpy(), cmap="gray")

plt.show()