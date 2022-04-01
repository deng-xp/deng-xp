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

log_dir = 'H:/研究生期间/毕设相关/毕设代码/resnet_model.pth'  # 模型保存路径

#查看设备
def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
#图像增强
'''transforms.Resize(256),
        transforms.RandomRotation(degrees=45),  # 随机旋转
        transforms.RandomHorizontalFlip(p=0.5),  # 依概率水平旋转
        transforms.CenterCrop(size=244),  # 中心裁剪到224*224符合resnet的输入要求
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],#转化为tensor，并归一化至[0，-1]
                             [0.2023, 0.1994, 0.2010])'''
'''transforms.Resize(244),  # 图像变换至256
#transforms.CenterCrop(size=256),
transforms.ToTensor(),  # 填充
torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010])'''
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
##########################读取图像相关###################################
'''train_dir=os.path.join(data_dir,'train')        #训练集
valid_dir=os.path.join(data_dir,'valid')        #验证集
test_dir=os.path.join(data_dir,'test')          #测试集
train_valid_dir=os.path.join(data_dir,'train_valid') #总体数据
#将数据存入字典，可通过键值调用;ImageFolder返回的对象是一个包含数据集所有图像及对应标签构成的二维元组容器
data={
    'train': torchvision.datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
     #imagefolder（root, transform），前者是图片路径，后者是对图片的变换，生成的数据类型是dataset
    'valid': torchvision.datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
    'test': torchvision.datasets.ImageFolder(root=test_dir, transform=image_transforms['valid']),
    'train_iter':torchvision.datasets.ImageFolder(root=train_valid_dir, transform=image_transforms['train'])
}
datasize=[len(data['train']),len(data['valid']),len(data['test'])]  #数据大小

#DataLoader(dataset, batch_size, shuffle) dataset数据类型；分组数；是否打乱(设置以何种方式将图像输入网络)
train_data=torch.utils.data.DataLoader(data['train'],batch_size=batch_size,shuffle=True, drop_last=True)
valid_data=torch.utils.data.DataLoader(data['valid'],batch_size=batch_size,shuffle=False, drop_last=True)
test_data=torch.utils.data.DataLoader(data['test'],batch_size=batch_size,shuffle=False, drop_last=True)
#所有数据
train_valid_iter=torch.utils.data.DataLoader(data['train_iter'],batch_size=batch_size,shuffle=True, drop_last=True)

print("训练集数据量为：{}，验证集数据量为：{},测试集数据量为{}".format(datasize[0],datasize[1],datasize[2]))
  #分别打印出训练集和验证集的样本数量
print("2.加载数据完毕")'''

################################模型构建###############################
#net=rnet.resnetcnn(3,num_class)     #定义模型
#net=rn1.resnetcnn2(3,num_class)
#定义损失函数
loss=nn.CrossEntropyLoss(reduction="none")
device=try_all_gpus()

loss_x=[-0.01]    #横坐标
lossres=[1]  #存储损失值
train_x=[-0.01]    #横坐标
train_accres=[0] #存储训练准确度
valid_x=[-0.01]    #横坐标
valid_accres=[0] #存储验证准确度
#训练函数，参数定义：网络结构、训练数据、验证数据、训练轮数、学习率、权重衰减、设备、学习率调整间隔、学习率调整倍数
def train(net, train_data, valid_data, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 实现随机梯度下降算法
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)  # 随机梯度下降算法,lr学习率,momentum动量因子,weight_decay权重衰减
    # 调整学习率(一般根据epoch训练次数来调整学习率)
    # 等间隔调整学习率，参数分别为优化器、调整间隔、调整倍数
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_data), ot.Timer()
    legend = ['train loss', 'train acc']
    if(valid_data is not None):
        legend.append('valid acc')
    animator = ot.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print("epoch=",epoch)
        net.train()
        metric = ot.Accumulator(3)  #metric为n维行向量
        for i, (features, labels) in enumerate(train_data):
            timer.start()       #开始计时
            l, acc = ot.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)   #l为损失和、acc为准确的和
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:       # //为除以向下取整
                animator.add(epoch + (i + 1) / num_batches,     #横坐标
                             (metric[0] / metric[2], metric[1] / metric[2],     #第二项为训练准确率
                              None))
                loss_x.append(epoch + (i + 1) / num_batches)
                train_x.append(epoch + (i + 1) / num_batches)
                lossres.append(metric[0] / metric[2])
                train_accres.append(metric[1] / metric[2])
                print("loss=",metric[0] / metric[2],";train_acc=",metric[1] / metric[2])
        if valid_data is not None:      #验证集
            valid_acc = ot.evaluate_accuracy_gpu(net, valid_data)
            animator.add(epoch + 1, (None, None, valid_acc))
            valid_x.append(epoch + 1)
            valid_accres.append(valid_acc)
        scheduler.step()    #更新学习率
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_data is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                    f' examples/sec on {str(devices)}')
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

#不加注意力机制
'''loss = nn.CrossEntropyLoss(reduction="none")
devices, num_epochs, lr, wd = try_all_gpus(), 50, 2e-4, 5e-4    #30
lr_period, lr_decay, net = 5, 0.96, get_net()   #5 0.96'''
#加注意力机制
loss = nn.CrossEntropyLoss(reduction="none")
devices, num_epochs, lr, wd = try_all_gpus(), 5, 2e-4, 5e-4    #30
lr_period, lr_decay, net = 5, 0.96, get_attennet()   #5 0.96

#开始训练
#不加注意力机制
'''net, preds = get_net(), []
train(net, train_data, valid_data, num_epochs, lr, wd, devices, lr_period,
      lr_decay)'''
#加入注意力机制
net, preds = get_attennet(), []
train(net, train_data, valid_data, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
plt.show()
plt.cla()

torch.save(net.state_dict(),log_dir)
#测试
'''net, preds = get_net(), []
train(net, train_valid_data, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)'''

for X, _ in test_data:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())   #在preds末尾追加元素;argmax()返回最大元素的索引
    print(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))  #创建列表,从1开始，到len()+1结束
sorted_ids.sort(key=lambda x: str(x))   #str()转换为字符串形式
df = pd.DataFrame({'id': sorted_ids, 'label': preds})   #构造数据框
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission1.csv', index=False)
plt.show()


# 设置输出的图片大小
figsize = 11, 9
figure, ax = plt.subplots(figsize=figsize)

# 在同一幅图片上画两条折线
A, = plt.plot(train_x,train_accres, color='#1E90FF', label='train_acc', linewidth=5.0)
B, = plt.plot(valid_x,valid_accres,  color='#FFA500', label='valid_acc', linewidth=5.0)

# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24,
         }
legend = plt.legend(handles=[A, B], prop=font1)

# 设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=22)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]
# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24,
         }
plt.xlabel('epoch', font2)
plt.ylabel('accuracy', font2)
plt.show()

plt.clf()
figsize = 11, 9
figure, ax = plt.subplots(figsize=figsize)
C, = plt.plot(loss_x,lossres,color='#1E90FF', label='loss', linewidth=4.0)
font4 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24,
         }
plt.legend(prop=font4)
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24,
         }
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]
plt.tick_params(labelsize=22)
plt.xlabel('epoch', font3)
plt.ylabel('loss', font3)
plt.show()

a=np.array(loss_x)
np.save(r"H:\loss_x.npy",a)
b=np.array(lossres)
np.save(r"H:\lossres.npy",b)
c=np.array(train_x)
np.save(r"H:\train_x.npy",c)
d=np.array(train_accres)
np.save(r"H:\trainacc.npy",d)
e=np.array(valid_x)
np.save(r"H:\valid_x.npy",e)
f=np.array(valid_accres)
np.save(r"H:\validacc.npy",f)
