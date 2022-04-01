import torch
from torch import nn
from torch.utils import data
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils import data
from torchvision import transforms
import collections
import math
import shutil
import pandas as pd

#定义卷积层
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),         #数据归一化处理
        nn.ReLU(),                          #激活函数
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)    #二维卷积：in_channel、out_channel表示输入图像通道数
                                                                                              #后三个参数表示：卷积核大小、步长、填充方式
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):    #growth_rate是通道变化的速度（因为denes是综合考虑前面多层输出的）,num_layers表示一共多少层
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)    #net为最终网络结构
    def forward(self, x):
        for layer in self.net:
            out = layer(x)          #layer(x)表示整体网络中的每一层，layer(x)表示输入为x的情况下，其输出是多少
            x = torch.cat((out, x), dim=1)  #进行结合，构建下一层新的输入
        return x

#函数目的：通过1X1的卷积层减少通道数，并使用步幅为2的平均池化层减半宽和高
def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1),     #1X1卷积层
        nn.AvgPool2d(2, 2)                         #2X2池化层
    )
    return trans_layer

class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(densenet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.DB1 = self._make_dense_block(64, growth_rate,num=block_layers[0])
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(512)
        self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(1024)
        self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
        self.global_average = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.DB1(x)
        x = self.TL1(x)
        x = self.DB2(x)
        x = self.TL2(x)
        x = self.DB3(x)
        x = self.TL3(x)
        x = self.DB4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)

def Densenet(num_classes):
    return densenet(in_channel=3,num_classes=num_classes)