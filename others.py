import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib
import math
import time
import numpy as np
import torch
from IPython import display
from matplotlib import pyplot as plt
import matplotlib_inline

def use_svg_display():
    display.set_matplotlib_formats('svg')
#设置坐标轴
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
class Timer:  #@save
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()
class Animator:  #@save
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        #用来创建总画布figure“窗口”
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):   #hasattr用于判断对象y中是否有"__len__"属性
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]     #初始化大小为n的空数组
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):  #zip()依次将多个可迭代对象中对应的元素打包为一个个元组；enumerate()枚举可迭代对象
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()      #清除当前图形中的当前活动轴。其他轴不受影响。
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
#用于对多个变量进行累加
class Accumulator:  #@save
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n   #生成n列的list

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
#@save
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)    #返回指定维度最大值序号
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
#X为训练图像数据、y为标签、loss为损失函数、trainer为梯度下降算法、devices为设备
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):     #判断一个对象是否是一个已知的类型:X是否为list
        # 微调BERT中所需（稍后讨论）
        X = [x.to(devices[0]) for x in X]       #将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()     #把模型的参数梯度设成0
    pred = net(X)           #pred为预测值
    l = loss(pred, y)       #计算预测误差损失
    l.sum().backward()      #梯度和（返回梯度为标量）
    trainer.step()          #更新模型
    train_loss_sum = l.sum()    #计算损失函数和
    train_acc_sum = accuracy(pred, y)   #计算准确预测的数量
    return train_loss_sum, train_acc_sum
def train_batch_chvalid(net, X, y, loss, devices):
    if isinstance(X, list):     #判断一个对象是否是一个已知的类型:X是否为list
        # 微调BERT中所需（稍后讨论）
        X = [x.to(devices[0]) for x in X]       #将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    pred = net(X)           #pred为预测值
    l = loss(pred, y)       #计算预测误差损失
    l.sum().backward()      #梯度和（返回梯度为标量）
    train_loss_sum = l.sum()    #计算损失函数和
    return train_loss_sum
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def show():
    plt.show()