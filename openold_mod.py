#0导入库
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import resnet18 as rnet
import others as ot
import torchvision
from torch.utils import data
from torchvision import transforms
import pandas as pd
import desnet as dnet
import resnet_c1 as rn1
import atresnet as at_res

log_dir = 'H:/研究生期间/毕设相关/毕设代码/参数保存/last_attention/resnet_model.pth'  # 模型保存路径
#log_dir = 'H:/研究生期间/毕设相关/毕设代码/resnet_model.pth'  # 模型保存路径
#查看设备
def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
image_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到256*256
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), #随机裁剪到256*256
        transforms.RandomRotation(degrees=15),#随机旋转
        transforms.RandomHorizontalFlip(p=0.5), #依概率水平旋转
        transforms.CenterCrop(size=224),#中心裁剪到224*224符合resnet的输入要求
        transforms.ToTensor(),#填充
        transforms.Normalize([0.485, 0.456, 0.406],#转化为tensor，并归一化至[0，-1]
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),#图像变换至256
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),#填充
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
print("1.数据增强完成")

batch_size=32   #一次读32张
num_class=2     #类别数
##########################加载数据############################################
data_dir='../toolwearclass'     #图片路径
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=image_transforms['train']) for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=image_transforms['valid']) for folder in ['valid', 'test']]

train_data, train_valid_data = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)   #读入顺序是否打乱（shuffle)
        for dataset in (train_ds, train_valid_ds)]

valid_data = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=True,
                                             drop_last=True)
test_data = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                            drop_last=False)
datasize=[len(train_ds),len(valid_ds),len(test_ds)]  #数据大小
print("训练集数据量为：{}，验证集数据量为：{},测试集数据量为{}".format(datasize[0],datasize[1],datasize[2]))
  #分别打印出训练集和验证集的样本数量
print("2.加载数据完毕")

################################模型构建###############################
#net=rnet.resnetcnn(3,num_class)     #定义模型
#net=rn1.resnetcnn2(3,num_class)
#定义损失函数
loss=nn.CrossEntropyLoss(reduction="none")
device=try_all_gpus()

#设置模型参数
def get_net():
    num_classes = 2
    net = rnet.resnetcnn(3, num_classes)
    return net
#带注意力机制的
def get_attennet():
    num_classes = 2
    net = at_res.attention_resnetcnn(3,num_classes)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
devices, num_epochs, lr, wd = try_all_gpus(), 50, 2e-4, 5e-4    #30
lr_period, lr_decay, net = 5, 0.96, get_attennet()   #5 0.96

#开始训练
net, preds = get_attennet(), []
net.load_state_dict(torch.load(log_dir))
net.eval()

preds=[0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0]
ans=[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1,1,1,0,1,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,1]
imgnum=len(ans)
print(imgnum)
preds=[]
for X, _ in test_data:
    y_hat = net(X)
    y_hat = y_hat.argmax(axis=1)
    c=y_hat.tolist()
    preds.append(c)
    #print(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

from tkinter import _flatten
predans=list(_flatten(preds))
print(predans)
print(predans.count(0))
print(predans.count(0)/len(predans))
