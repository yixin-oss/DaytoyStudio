---
title: 残差神经网络(ResNet)
---

从前面经典的卷积神经网络模型可以看到，增加网络的宽度和深度可以很好地提高网络的性能, 深的网络一般比浅的网络效果好, 因此在训练模型的时候会想要增加网络的层数, 但随之而来的是梯度消失和退化问题.

- 梯度消失

梯度在反向传播过程中会逐层递减, 导致权重更新较小, 深层的网络难以得到有效的训练. 梯度消失使得网络难以收敛和优化.

- 网络退化

随着网络层数的增加, 网络的性能反而下降, 即使深度增加, 训练误差也在增加，与预期相反, 限制了网络的有效性.

为了解决上述问题，微软研究院的何恺明提出了**残差神经网络**(ResNet).

## 残差神经网络(ResNet)

![image-20240412135350547](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240412135350547.png)

![image-20240412135417319](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240412135417319.png)

- 在残差块的末尾, 通过残差连接将输入特征直接相加到最后输出特征上, 通过简单的输入与输出特征相加, 这种跳跃连接允许网络学习残差部分, 即输入输出问题, 有助于减轻网络退化问题.


- 这种设计要求2个卷积层输入与输出形状(通道)相同, 也就是使得第二层输出与原始输入形状相同才能相加. 如果需要改变通道数, 引入额外的$1\times 1$卷积层将输入变换为需要的形状.


**注1**: 前面经典的卷积神经网络都是基于TensorFlow编程实现, 之后的代码将全部改用pytorch.

**注2**: 以下ResNet18的代码中残差基本块包含了两套卷积和一条捷径, 每套卷积中又包含两个卷积层和两个BN层. 网络的详细结构可以`print(net)`查看.

```python
# 导入必要的模块
import torch
from torch import nn, optim
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
```


```python
# 定义两个卷积路径和一条捷径的残差基本块
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # 定义一条捷径, 若输入与输出的图像尺寸有变化(stirde不为1或通道数改变), 捷径通过1x1卷积用stride修改大小
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # + short cut, element-wise add.
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```


```python
# 定义残差网络ResNet18
class ResNet18(nn.Module):
    # 定义初始化函数, 输入为残差块, 残差块数量, 默认分类数为10
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet18, self).__init__()

        # 设置第一层的通道数
        self.in_ch = 64
        # 输入图片先进行一次卷积和批标准化, 输入通道3 => 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        # 第一层, 输出通道64, 有num_blocks[0]个残差块, 残差块中第一个卷积步长自定义为1
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        # 全连接层
        self.linear = nn.Linear(512, num_classes)

    def make_layer(self, block, out_ch, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        # 定义创造层函数, 输入参数为残差块, 通道数, 残差块数量, 步长
        layers = []
        for stride in strides:
            layers.append(block(self.in_ch, out_ch, stride))
            self.in_ch = out_ch
        return nn.Sequential(*layers)
        

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4) # 经过一次 4 x 4 average pooling
        # Flatten
        # 将out重新调整为(batch_size, -1)的二维张量
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```


```python
# 打印网络的结构
net = ResNet18(BasicBlock, [2, 2, 2, 2])
print(net)
```

    ResNet18(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
      )
      (linear): Linear(in_features=512, out_features=10, bias=True)
    )



```python
def main():
    summary(net, (3, 32, 32))

main()
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 32, 32]           1,728
           BatchNorm2d-2           [-1, 64, 32, 32]             128
                Conv2d-3           [-1, 64, 32, 32]          36,864
           BatchNorm2d-4           [-1, 64, 32, 32]             128
                Conv2d-5           [-1, 64, 32, 32]          36,864
           BatchNorm2d-6           [-1, 64, 32, 32]             128
            BasicBlock-7           [-1, 64, 32, 32]               0
                Conv2d-8           [-1, 64, 32, 32]          36,864
           BatchNorm2d-9           [-1, 64, 32, 32]             128
               Conv2d-10           [-1, 64, 32, 32]          36,864
          BatchNorm2d-11           [-1, 64, 32, 32]             128
           BasicBlock-12           [-1, 64, 32, 32]               0
               Conv2d-13          [-1, 128, 16, 16]          73,728
          BatchNorm2d-14          [-1, 128, 16, 16]             256
               Conv2d-15          [-1, 128, 16, 16]         147,456
          BatchNorm2d-16          [-1, 128, 16, 16]             256
               Conv2d-17          [-1, 128, 16, 16]           8,192
          BatchNorm2d-18          [-1, 128, 16, 16]             256
           BasicBlock-19          [-1, 128, 16, 16]               0
               Conv2d-20          [-1, 128, 16, 16]         147,456
          BatchNorm2d-21          [-1, 128, 16, 16]             256
               Conv2d-22          [-1, 128, 16, 16]         147,456
          BatchNorm2d-23          [-1, 128, 16, 16]             256
           BasicBlock-24          [-1, 128, 16, 16]               0
               Conv2d-25            [-1, 256, 8, 8]         294,912
          BatchNorm2d-26            [-1, 256, 8, 8]             512
               Conv2d-27            [-1, 256, 8, 8]         589,824
          BatchNorm2d-28            [-1, 256, 8, 8]             512
               Conv2d-29            [-1, 256, 8, 8]          32,768
          BatchNorm2d-30            [-1, 256, 8, 8]             512
           BasicBlock-31            [-1, 256, 8, 8]               0
               Conv2d-32            [-1, 256, 8, 8]         589,824
          BatchNorm2d-33            [-1, 256, 8, 8]             512
               Conv2d-34            [-1, 256, 8, 8]         589,824
          BatchNorm2d-35            [-1, 256, 8, 8]             512
           BasicBlock-36            [-1, 256, 8, 8]               0
               Conv2d-37            [-1, 512, 4, 4]       1,179,648
          BatchNorm2d-38            [-1, 512, 4, 4]           1,024
               Conv2d-39            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-40            [-1, 512, 4, 4]           1,024
               Conv2d-41            [-1, 512, 4, 4]         131,072
          BatchNorm2d-42            [-1, 512, 4, 4]           1,024
           BasicBlock-43            [-1, 512, 4, 4]               0
               Conv2d-44            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-45            [-1, 512, 4, 4]           1,024
               Conv2d-46            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-47            [-1, 512, 4, 4]           1,024
           BasicBlock-48            [-1, 512, 4, 4]               0
               Linear-49                   [-1, 10]           5,130
    ================================================================
    Total params: 11,173,962
    Trainable params: 11,173,962
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 11.25
    Params size (MB): 42.63
    Estimated Total Size (MB): 53.89
    ----------------------------------------------------------------



```python
def main():
    x = torch.randn(2, 3, 32, 32)
    out = net(x)
    print(out.shape)

main()
```

    torch.Size([2, 10])



```python
# 图像预处理变换
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
```


```python
# 下载数据集, 训练与否, 数据预处理方式, 下载与否
cifar_train = datasets.CIFAR10('cifar', True, transform=transform_train, download=True)
cifar_test = datasets.CIFAR10('cifar', False, transform=transform_test, download=True)
```

    Files already downloaded and verified
    Files already downloaded and verified



```python
feature, label = cifar_train[3]
print(feature.shape, feature.dtype)
print(label)
```

    torch.Size([3, 32, 32]) torch.float32
    4

```python
batch_size=128
# 构建可迭代的数据装载器(参数为数据集, 批样本数, 是否乱序)
train_iter = DataLoader(cifar_train, batch_size=128, shuffle=True)
test_iter = DataLoader(cifar_test, batch_size=100, shuffle=False)
```

```python
def main():
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    Epochs = 100
    acc_history = []
    train_history = []
    for epoch in range(Epochs):
        train_loss_sum, train_acc_sum, num = 0, 0, 0
        net.train()
        for batchidx, (x, label) in enumerate(train_iter):
            logits = net(x)
            L = loss(logits, label)

            # backprop
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            train_loss_sum += L.item()
            train_acc_sum += (logits.argmax(dim=1) == label).sum().item()
            num += label.shape[0]
        train_loss = train_loss_sum / batchidx
        train_acc = train_acc_sum / num
        train_history.append(train_acc)

        net.eval()
        with torch.no_grad():
            total_correct = 0
            total_num =0
            for x, label in test_iter:
                logits = net(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum()
                total_num += x.size(0)

            test_acc = total_correct / total_num
            acc_history.append(test_acc)
            print(f'Epoch: {epoch}, Loss: {train_loss}, Train_Acc: {train_acc}, Test_Acc: {test_acc}')

    plt.plot(range(1, Epochs + 1), acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Test_Accuracy')
    plt.title('Epoch vs. Test_Accuracy')

    plt.plot(range(1, Epochs + 1), train_history)
    plt.xlabel('Epoch')
    plt.ylabel('Train_Accuracy')
    plt.title('Epoch vs. Train_Accuracy')
    
    plt.show()

    print('save total model.')
    torch.save(net.state_dict(),'ResNet18-pytorch.pth')

main()
```

    Epoch: 0, Loss: 1.300478709508211, Train_Acc: 0.52498, Test_Acc: 0.6049000024795532
    Epoch: 1, Loss: 0.7848536547177877, Train_Acc: 0.72258, Test_Acc: 0.723800003528595
    Epoch: 2, Loss: 0.5801862900073712, Train_Acc: 0.79894, Test_Acc: 0.7886999845504761
    Epoch: 3, Loss: 0.4524255673854779, Train_Acc: 0.84326, Test_Acc: 0.7752000093460083
    Epoch: 4, Loss: 0.3533191406574005, Train_Acc: 0.8767, Test_Acc: 0.7968000173568726
    Epoch: 5, Loss: 0.2739773054726613, Train_Acc: 0.9039, Test_Acc: 0.8180999755859375
    Epoch: 6, Loss: 0.1958709563391331, Train_Acc: 0.93186, Test_Acc: 0.8274999856948853
    Epoch: 7, Loss: 0.13914955153297157, Train_Acc: 0.95028, Test_Acc: 0.8190000057220459
    Epoch: 8, Loss: 0.10536972259481749, Train_Acc: 0.96266, Test_Acc: 0.8289999961853027
    Epoch: 9, Loss: 0.08836400117486333, Train_Acc: 0.96832, Test_Acc: 0.8375999927520752
    Epoch: 10, Loss: 0.06805563183644643, Train_Acc: 0.97582, Test_Acc: 0.8212000131607056
    Epoch: 11, Loss: 0.05809408232378654, Train_Acc: 0.97958, Test_Acc: 0.8320000171661377
    Epoch: 12, Loss: 0.05708230435131834, Train_Acc: 0.97984, Test_Acc: 0.8259999752044678
    Epoch: 13, Loss: 0.0483401582403204, Train_Acc: 0.98316, Test_Acc: 0.8263000249862671
    Epoch: 14, Loss: 0.04180757365404413, Train_Acc: 0.98572, Test_Acc: 0.8371000289916992
    Epoch: 15, Loss: 0.04230772689259492, Train_Acc: 0.98566, Test_Acc: 0.8331000208854675
    Epoch: 16, Loss: 0.04220948502098998, Train_Acc: 0.98546, Test_Acc: 0.8385999798774719
    Epoch: 17, Loss: 0.03146164209331171, Train_Acc: 0.98972, Test_Acc: 0.8391000032424927
    Epoch: 18, Loss: 0.03514277003556251, Train_Acc: 0.98792, Test_Acc: 0.8456000089645386
    Epoch: 19, Loss: 0.03599046423989467, Train_Acc: 0.98742, Test_Acc: 0.8309000134468079
    Epoch: 20, Loss: 0.03063486601608113, Train_Acc: 0.98934, Test_Acc: 0.8373000025749207
    Epoch: 21, Loss: 0.028839362433063797, Train_Acc: 0.99042, Test_Acc: 0.8458999991416931
    Epoch: 22, Loss: 0.02261277259557317, Train_Acc: 0.99228, Test_Acc: 0.829200029373169
    Epoch: 23, Loss: 0.031302750774790555, Train_Acc: 0.98932, Test_Acc: 0.8359000086784363
    Epoch: 24, Loss: 0.02696963624031737, Train_Acc: 0.9907, Test_Acc: 0.8356000185012817
    Epoch: 25, Loss: 0.021279103275484, Train_Acc: 0.9927, Test_Acc: 0.833299994468689
    Epoch: 26, Loss: 0.021790118695935234, Train_Acc: 0.99236, Test_Acc: 0.8385999798774719
    Epoch: 27, Loss: 0.03071881690964055, Train_Acc: 0.9897, Test_Acc: 0.8446999788284302
    Epoch: 28, Loss: 0.018251167449544973, Train_Acc: 0.99408, Test_Acc: 0.8371000289916992
    Epoch: 29, Loss: 0.01580918610807902, Train_Acc: 0.99476, Test_Acc: 0.8355000019073486
    Epoch: 30, Loss: 0.019879301222196468, Train_Acc: 0.9936, Test_Acc: 0.8338000178337097
    Epoch: 31, Loss: 0.019010732998140156, Train_Acc: 0.99348, Test_Acc: 0.8403000235557556
    Epoch: 32, Loss: 0.019425697318943908, Train_Acc: 0.99362, Test_Acc: 0.8424000144004822
    Epoch: 33, Loss: 0.020338820099893313, Train_Acc: 0.99326, Test_Acc: 0.8403000235557556
    Epoch: 34, Loss: 0.01879438050382305, Train_Acc: 0.99336, Test_Acc: 0.8478999733924866
    Epoch: 35, Loss: 0.01755093706927325, Train_Acc: 0.99386, Test_Acc: 0.8500999808311462
    Epoch: 36, Loss: 0.012930231594453709, Train_Acc: 0.99572, Test_Acc: 0.8453999757766724
    Epoch: 37, Loss: 0.010597185114247856, Train_Acc: 0.99636, Test_Acc: 0.838699996471405
    Epoch: 38, Loss: 0.01959872076807257, Train_Acc: 0.99346, Test_Acc: 0.8343999981880188
    Epoch: 39, Loss: 0.018881254075378634, Train_Acc: 0.99394, Test_Acc: 0.8458999991416931
    Epoch: 40, Loss: 0.012509921958777481, Train_Acc: 0.99568, Test_Acc: 0.8295999765396118
    Epoch: 41, Loss: 0.015792606963152184, Train_Acc: 0.9944, Test_Acc: 0.8406000137329102
    Epoch: 42, Loss: 0.011632570275441349, Train_Acc: 0.99622, Test_Acc: 0.8449000120162964
    Epoch: 43, Loss: 0.014979728682971118, Train_Acc: 0.99482, Test_Acc: 0.838699996471405
    Epoch: 44, Loss: 0.013055829245758314, Train_Acc: 0.99578, Test_Acc: 0.8460999727249146
    Epoch: 45, Loss: 0.009737687622747706, Train_Acc: 0.99648, Test_Acc: 0.8561000227928162
    Epoch: 46, Loss: 0.012149526452785955, Train_Acc: 0.99584, Test_Acc: 0.8442000150680542
    Epoch: 47, Loss: 0.013296451657720938, Train_Acc: 0.99582, Test_Acc: 0.8501999974250793
    Epoch: 48, Loss: 0.011425589930914635, Train_Acc: 0.99614, Test_Acc: 0.8371000289916992
    Epoch: 49, Loss: 0.010803057540248026, Train_Acc: 0.9965, Test_Acc: 0.8500999808311462
    Epoch: 50, Loss: 0.010000550045776598, Train_Acc: 0.99668, Test_Acc: 0.8496000170707703
    Epoch: 51, Loss: 0.007432474986741839, Train_Acc: 0.99762, Test_Acc: 0.8475000262260437
    Epoch: 52, Loss: 0.013419993099579678, Train_Acc: 0.99526, Test_Acc: 0.828499972820282
    Epoch: 53, Loss: 0.015232682537932236, Train_Acc: 0.99472, Test_Acc: 0.8464999794960022
    Epoch: 54, Loss: 0.006207723521621143, Train_Acc: 0.99806, Test_Acc: 0.8529000282287598
    Epoch: 55, Loss: 0.00835985738922318, Train_Acc: 0.99704, Test_Acc: 0.8449000120162964
    Epoch: 56, Loss: 0.013050546994185052, Train_Acc: 0.99586, Test_Acc: 0.8396000266075134
    Epoch: 57, Loss: 0.01013894579722173, Train_Acc: 0.99652, Test_Acc: 0.84579998254776
    Epoch: 58, Loss: 0.009989700726071487, Train_Acc: 0.99648, Test_Acc: 0.8460000157356262
    Epoch: 59, Loss: 0.008592608001410366, Train_Acc: 0.9974, Test_Acc: 0.8471999764442444
    Epoch: 60, Loss: 0.006842050987119691, Train_Acc: 0.99782, Test_Acc: 0.845300018787384
    Epoch: 61, Loss: 0.009545303711993023, Train_Acc: 0.99654, Test_Acc: 0.8438000082969666
    Epoch: 62, Loss: 0.009218481678031729, Train_Acc: 0.99696, Test_Acc: 0.8378999829292297
    Epoch: 63, Loss: 0.007699655817622186, Train_Acc: 0.99732, Test_Acc: 0.8526999950408936
    Epoch: 64, Loss: 0.009866955332500555, Train_Acc: 0.99682, Test_Acc: 0.8406999707221985
    Epoch: 65, Loss: 0.012016661322521917, Train_Acc: 0.99618, Test_Acc: 0.845300018787384
    Epoch: 66, Loss: 0.006403025254523807, Train_Acc: 0.99778, Test_Acc: 0.8514999747276306
    Epoch: 67, Loss: 0.011008849037898472, Train_Acc: 0.99612, Test_Acc: 0.8446999788284302
    Epoch: 68, Loss: 0.007007013085837034, Train_Acc: 0.99778, Test_Acc: 0.852400004863739
    Epoch: 69, Loss: 0.007240867378622613, Train_Acc: 0.99758, Test_Acc: 0.8518000245094299
    Epoch: 70, Loss: 0.004491697912983046, Train_Acc: 0.9986, Test_Acc: 0.8536999821662903
    Epoch: 71, Loss: 0.008413086927183218, Train_Acc: 0.99724, Test_Acc: 0.8449000120162964
    Epoch: 72, Loss: 0.012969305682408725, Train_Acc: 0.9955, Test_Acc: 0.8388000130653381
    Epoch: 73, Loss: 0.008549963680921698, Train_Acc: 0.9971, Test_Acc: 0.8557999730110168
    Epoch: 74, Loss: 0.003646867551287869, Train_Acc: 0.99872, Test_Acc: 0.8539999723434448
    Epoch: 75, Loss: 0.0027940699092631315, Train_Acc: 0.9991, Test_Acc: 0.8589000105857849
    Epoch: 76, Loss: 0.00561697895654316, Train_Acc: 0.99824, Test_Acc: 0.833899974822998
    Epoch: 77, Loss: 0.01636295277459841, Train_Acc: 0.99494, Test_Acc: 0.8493000268936157
    Epoch: 78, Loss: 0.0036292208312611015, Train_Acc: 0.9989, Test_Acc: 0.8515999913215637
    Epoch: 79, Loss: 0.0012174570014674548, Train_Acc: 0.9996, Test_Acc: 0.8572999835014343
    Epoch: 80, Loss: 0.0007334858212813672, Train_Acc: 0.99978, Test_Acc: 0.857699990272522
    Epoch: 81, Loss: 0.014180392358239494, Train_Acc: 0.9957, Test_Acc: 0.833899974822998
    Epoch: 82, Loss: 0.013381059732665353, Train_Acc: 0.99556, Test_Acc: 0.8468000292778015
    Epoch: 83, Loss: 0.005694386397259129, Train_Acc: 0.99814, Test_Acc: 0.8529000282287598
    Epoch: 84, Loss: 0.0031447352890772422, Train_Acc: 0.99894, Test_Acc: 0.8565000295639038
    Epoch: 85, Loss: 0.0024621304439512107, Train_Acc: 0.99934, Test_Acc: 0.8503000140190125
    Epoch: 86, Loss: 0.008869428431306657, Train_Acc: 0.99722, Test_Acc: 0.847100019454956
    Epoch: 87, Loss: 0.011784704768452241, Train_Acc: 0.99648, Test_Acc: 0.8504999876022339
    Epoch: 88, Loss: 0.002623334808492636, Train_Acc: 0.99926, Test_Acc: 0.8611000180244446
    Epoch: 89, Loss: 0.0021657184337084637, Train_Acc: 0.99924, Test_Acc: 0.8518999814987183
    Epoch: 90, Loss: 0.0030502214310917436, Train_Acc: 0.99898, Test_Acc: 0.8551999926567078
    Epoch: 91, Loss: 0.008430158374521386, Train_Acc: 0.99708, Test_Acc: 0.8450000286102295
    Epoch: 92, Loss: 0.010825034797044981, Train_Acc: 0.9963, Test_Acc: 0.8481000065803528
    Epoch: 93, Loss: 0.0046861603127464875, Train_Acc: 0.99842, Test_Acc: 0.8507999777793884
    Epoch: 94, Loss: 0.0027030261518074185, Train_Acc: 0.99922, Test_Acc: 0.8532000184059143
    Epoch: 95, Loss: 0.003941894976168652, Train_Acc: 0.9989, Test_Acc: 0.847100019454956
    Epoch: 96, Loss: 0.009912522807294759, Train_Acc: 0.99688, Test_Acc: 0.8367999792098999
    Epoch: 97, Loss: 0.008258439671632145, Train_Acc: 0.99734, Test_Acc: 0.8521000146865845
    Epoch: 98, Loss: 0.0027327944688878255, Train_Acc: 0.99902, Test_Acc: 0.8532999753952026
    Epoch: 99, Loss: 0.00256529226822423, Train_Acc: 0.9992, Test_Acc: 0.8544999957084656


![ResNet18result](https://gitee.com/yixin-oss/blogImage/raw/master/Img/ResNet18result.png)
    


    save total model.



```python
class10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
import random
ind = random.randint(0, 10000)
cifar_test = datasets.CIFAR10('cifar', False, transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
test_data = DataLoader(cifar_test)
for i, data in enumerate(test_data):
    if i == ind:
        batch_data = data
        break
x, label = batch_data
out = net(x)
pred = out.argmax(dim=1).numpy()
print(class10[pred.item()])
# x= test_data[ind][0]
x = x.squeeze(0)
x = x.numpy()
x = np.transpose(x, (1, 2, 0))
plt.figure(figsize=(1, 1))
plt.imshow(x)
plt.show()
```

    Files already downloaded and verified
    bird


![bird](https://gitee.com/yixin-oss/blogImage/raw/master/Img/bird.png)
    

| CIFAR10数据集分类模型; 训练50次 | 测试集准确率 |
| ------------------------------- | ------------ |
| LeNet5                          | 60%          |
| VGG19                           | 73%          |
| ResNet18                        | 85%          |

将测试集上的准确率进行比较, 可以明显看出ResNet18的优势, 但$85\%$的准确率还有很大的提升空间, 下一步将考虑在网络中添加更多细节化的处理.