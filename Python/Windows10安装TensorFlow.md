---
title: Windows10安装TensorFlow
tags: 
- TensorFlow
- 软件安装
categories: Tensorflow学习笔记
---

## 1. Anaconda下载及安装

[Anaconda官网](https://www.anaconda.com/products/individual#macos)

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607066052.png)

<!--more-->

下载Windows的相应版本

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607066052(1).jpg)

下载安装包后双击按引导安装

这里要**勾选!**将Anaconda加入环境变量.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607066052.jpg)

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607066358(1).jpg)

等待安装完成.

## 2. TensorFlow安装

开始-->Anaconda3(64-bit)-->Anaconda Prompt(Anaconda3)

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607066505(1).jpg" style="zoom: 67%;" />

用**conda create -n **新建一个名为tensorflow的环境，用python3.6版本(3.7也可)

```
conda create -n pytorch
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1.png" style="zoom:80%;" />

这里选择y表示同意安装相关软件包

**conda activate tensorflow**进入tensorflow虚拟环境环境

```
conda activate pytorch
```



输入以下代码安装深度学习软件包

**conda install cudatoolkit=10.1**

**conda install cudnn=7.6**

```
conda install cudatoolkit=10.1

conda install cudnn=7.6
```

如果两条安装语句报错，可能是电脑硬件不支持英伟达GPU，直接跳过这两步，安装tensorflow.

**pip install tensflow==2.1**

```
pip install tensflow==2.1
```

默认Google源，下载可能会很慢.

这里注意要指定tensorflow为2.1版本，不然默认安装的是2.3版本，后续使用一些函数可能会报错，如tf.Variable(),如果已经安装了tensorflow2.3并且出现函数报错，可卸载重新安装2.1版本

```
pip uninstall tensorflow
pip install tensorflow==2.1
```

安装完成后，进入python验证是否成功

依次输入

```
python

```

```
import tensorflow as tf

```

```
tf.__version__
```



<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607067507(1).jpg" style="zoom:80%;" />

如果显示2.1.0表示安装成功.

## 3. PyCharm集成开发环境

[PyCharm官网](https://www.jetbrains.com/)

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607068212.jpg" style="zoom: 50%;" />



<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607068212(1).jpg" style="zoom: 50%;" />

下载社区版PyCharm

打开安装包进行安装

这里右侧的添加环境变量要**！勾选**，左侧根据个人习惯(可以**都选上！**)

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607068405(1).jpg)

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607068602(1).png)

这里需要重启电脑

重启后打开PYCharm，新建工程，可以使用默认路径，文件夹命名‘’AI‘’，设置环境变量



<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607069116(1).jpg" style="zoom:67%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607069143(1).jpg" style="zoom:67%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607069206(1).jpg" style="zoom:67%;" />

新建文件

右击文件夹AI-->New-->File

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607069401.jpg)

命名为test.py

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607069432(1).jpg)

输入测试代码

```python
import tensorflow as tf

tensorflow_version=tf.__version__
gpu_available = tf.test.is_gpu_available()

print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)

a = tf.constant([2.0, 2.0], name='a')
b = tf.constant([4.0, 2.0], name='b')
result = tf.add(a, b, name='add')
print(result)
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607069567(1).jpg" style="zoom: 80%;" />

点击**Run ‘test’**，或者**Ctrl+Shift+F10**运行代码

如果出现**tf.tensor**,说明开发环境已安装成功.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607069751(1).png)

后续需要安装一些必要的包，如sklearn,numpy,matplotlib等，点击下方Terminal出现操作台，输入相应下载命令即可.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607342902(1).png)

```
pip install sklearn

pip install numpy

pip install matplotlib

```

## 补充: Juptynotebook切换到配置好的conda虚拟环境

在Anaconda Prompt中依次输入

```python
conda activate pytorch # 进入配置好的虚拟环境, 名称tensorflow

conda install ipykernel #安装ipykernel

python -m ipykernel install --name pytorch
# 自动在目录生成一个tensorflow文件夹, 里面有kernel.json文件
```

现在打开Juptynotebook, 里面就会有虚拟环境可以切换了

## 补充: Pycharm中调用pytorch

在pycharm中即使已经进入到装有torch的虚拟环境中, 也可能无法正常`import torch`, 这是因为设置的conda解释器中缺少`pytorch/torchvision`的缘故, 这时就需要手动添加解释器, 将它们安装好后即可正常调用.

![image-20240530215134325](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240530215134325.png)

![image-20240530215200339](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240530215200339.png)

显示蓝色即为安装成功.

