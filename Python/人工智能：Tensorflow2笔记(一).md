---
title:人工智能: Tensorflow2笔记(一)
tags:
- 深度学习
- Tensorflow
- 人工智能
categories: Tensorflow学习笔记
---

## 人工智能

让机器具备人的思维和意识.

### 人工智能三学派：

- 行为主义：基于控制论，构建感知-动作控制系统.（控制论，如平衡、行走、避障等自适应控制系统，实例：让机器人抬起一只脚，如何控制整体平衡）

- 符号主义：基于算数逻辑表达式，求解问题先把问题描述为表达式，再求表达式.(公式描述、实现理性思维，如专家系统)

- 连接主义：仿生学，模仿神经元连接关系.(仿脑神经元连接，实现感性思维，如神经网络)
<!--more-->

## 神经网络设计过程

√ 准备数据：采集大量“特征、标签”数据，数据量越大越好

√ 搭建网络：搭建神经网络结构

√优化参数： 训练网络获取最佳参数（反向传播过程）

√ 应用网络： 将网络保存为模型，输入新数据，输出分类或预测结果（前向传播过程）

简单来说，神经网络的设计过程就是给出模型参数随机初始值，将收集特征数据作为输入层利用模型求预测标签结果，与正确结果作比较计算偏差（损失函数），此时的偏差（损失函数）为模型中参数的函数，可以利用优化方法中的**梯度下降法**求出使偏差最小的模型参数返回给网络模型，再次利用数据重复这个过程，这样就实现了模型中的参数动态更新.(神经网络的训练实质就是利用数据不断更新模型的参数.)

- 损失函数：预测值与标准结果的差距.

  损失函数可以定量判断模型参数的优劣，当损失函数输出最小时，参数出现最优值.(求解函数最小值实质就是一个优化问题，用到所提到的**梯度下降法**.)

  常用的损失函数：均方误差
  $$
  MSE(y,y_{correct})=\frac{\sum_{k=0}^{n}(y-y_{correct})^2}{n}
  $$

- 梯度下降

  沿损失函数梯度下降的方向，寻找损失函数最小值，得到最佳参数的迭代方法.

  梯度下降法的关键是设置**步长** 和**参数**.

  **梯度：**函数对各参数求偏导后的向量.函数梯度下降的方向是函数减小的方向.

  参数迭代公式：
  $$
  w_{t+1}=w_{t}-lr*\frac{\partial loss}{\partial w_t}
  $$


  **学习率lr：**实质是梯度下降方向的步长，设置过小，梯度下降收敛缓慢；过大，在最小值附近震荡，甚至不收敛.

  

- 反向传播：实质就是参数更新

  逐层求损失函数对每层神经元参数的偏导，迭代更新所有参数.

  e.g. 损失函数 
  $$
  loss=(w+1)^2\\
  \frac{\partial loss}{\partial w}=2w+2
  $$
  参数初始化为5，学习率0.2，则

  ![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607053527(1).jpg)

  

这里0开始是第一次迭代寻找参数最优值，经过几次迭代，损失函数最小值为0，参数w最优值为-1.

TensorFlow实现上述过程：

```
import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epoch = 40

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w+1)
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
print('After %s epoch,w is %f,loss is %f'% (epoch, w.numpy(), loss))
```

## 张量（Tensor）

多维数组（列表）

阶：张量的维数

| 维数 |  阶  |    名字     |                  例子                  |
| :--: | :--: | :---------: | :------------------------------------: |
| 0-D  |  0   | 标量 scalar |              s=123,数123               |
| 1-D  |  1   | 向量 vector |        v=[1,2,3],向量（1,2,3）         |
| 2-D  |  2   | 矩阵 matrix | m=[[1,2,3],[4,5,6],[7,8,9]],3行3列矩阵 |
| 2-D  |  n   | 张量 tensor |              t=[[[...n个               |

故张量为维数可以通过数单边括号个数得到，张量可以表示0阶~n阶数组

### 数据类型

- tf.int, tf.float...整数浮点数类型 

  tf.int32, tf.float32, tf.float64

-  tf.bool布尔数类型

  tf.constant([True,False])

- tf.string 字符串类型

  tf.constant(''Hello World!")

### 创建张量

- 创建一个张量

**tf.constant(内容， dtype=数据类型（可选）)**

```python
import tensorflow as tf
x1 = tf.constant([1, 2, 3], dtype=tf.float64)
print(x1)
```



```
运行结果：
tf.Tensor([1. 2. 3.], shape=(3,), dtype=float64)
```

- 将numpy数据类型转换为Tensor数据类型

  **tf.convert_to_tensor(数据名， dtype=数据类型)**

````python
import tensorflow as tf
import numpy as np
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print(a)
print(b)
````

```
运行结果：
[0 1 2 3 4]
tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)
```

- 创建全为0的张量

  **tf.zeros(维度)**

- 创建全为1的张量

  **tf.ones(维度)**

- 创建全为指定值的张量

  **tf.fill(维度，指定值)**

```
a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print(a)
print(b)
print(c)
```

```
运行结果：
tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
tf.Tensor(
[[9 9]
 [9 9]], shape=(2, 2), dtype=int32)
```

- 生成正态分布随机数，默认均值0，标准差1

  **tf.random.normal(维度， mean=均值， stddev=标准差)**

- 生成截断式正态分布随机数：保证生成值在均值附近

  **tf.random.truncated_normal(维度， mean=均值， stddev=标准差)**

```
import tensorflow as tf

p = tf.random.normal([2, 2], mean=0, stddev=1)
print(p)

p1 = tf.random.truncated_normal([2, 2], mean=0, stddev=1)
print(p1)
```

```
运行结果：
tf.Tensor(
[[ 8.3016290e-04 -1.2167720e+00]
 [ 2.1969645e-01 -3.4597743e-01]], shape=(2, 2), dtype=float32)
tf.Tensor(
[[ 1.7608268 -1.3735857]
 [-0.7695601  0.6406028]], shape=(2, 2), dtype=float32)
```

- 生成均匀分布随机数

  **tf.random.uniform(维度，minval=最小值， maxval=最大值)**

```
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print(f)
```

```
tf.Tensor(
[[0.36967945 0.7651223 ]
 [0.53542113 0.6690651 ]], shape=(2, 2), dtype=float32)
```

### 常用函数I

- 强制tensor转换为该数据类型

  **tf.cast(张量名， dtype=数据类型)**

- 计算张量维度上元素最小值

  **tf.reduce_min()**

- 计算张量维度上元素最大值

  **tf.reduce_max()**

- 理解axis

  调整axis等于0或1控制执行维度.

  axis=0代表跨行（经度）（竖排）；axis=1代表跨列（纬度）（横排）；不指定则所有元素参与计算.

- 计算指定维度平均值

  **tf.reduce_mean(张量名， axis=操作轴)**

- 指定维度和

  **tf.reduce_sum(张量名， axis=操作轴)**

  ```
  import tensorflow as tf
  import numpy as np
  x = tf.constant([[1, 2, 3],
                  [2, 2, 3]])
  print(x)
  
  print(tf.reduce_mean(x))
  
  print(tf.reduce_sum(x, axis=1))
  ```

  ```
  tf.Tensor(
  [[1 2 3]
   [2 2 3]], shape=(2, 3), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor([6 7], shape=(2,), dtype=int32)
  ```

  

- **tf.Variable**

  **tf.Variable(初始值)**将变量**标记为“可训练”**，被标记的变量会在反向传播中记录梯度信息.神经网络训练中，常用该函数标记待训练参数.

  ```
  import tensorflow as tf
  w=tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
  print(w)
  ```

  ```
  <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
  array([[ 1.8879577 ,  0.790655  ],
         [-3.0929613 ,  0.50161296]], dtype=float32)>
  ```

  ### TensorFlow中的数学运算

  |   tf.    | 对应元素 |
  | :------: | :------: |
  |   add    |    +     |
  | subtract |    -     |
  | multiply |    *     |
  |  divide  |    /     |
  |  square  |    ^2    |
  |   pow    |    ^n    |
  |   sqrt   |   开方   |
  |  matmul  | 矩阵乘法 |

  ### 常用函数II

  **tf.data.Dataset.from_tensor_slices** 切分传入张量的第一维度，生成输入特征/标签对，构建数据集（Numpy和Tensor格式都可用该语句读入数据）

  ```
  import tensorflow as tf
  #将特征与标签配对，并逐对打印输出
  features = tf.constant([[12, 23, 10, 17], [10, 30, 21, 19], [13, 10, 22, 17], [29, 33,39,49]])
  labels=tf.constant([0, 1, 1, 3])
  dataset=tf.data.Dataset.from_tensor_slices((features, labels))
  print(dataset)
  for element in dataset :
      print(element)
  ```

  ```
  每对中特征有四个数据：
  <TensorSliceDataset shapes: ((4,), ()), types: (tf.int32, tf.int32)>
  (<tf.Tensor: shape=(4,), dtype=int32, numpy=array([12, 23, 10, 17])>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
  (<tf.Tensor: shape=(4,), dtype=int32, numpy=array([10, 30, 21, 19])>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
  (<tf.Tensor: shape=(4,), dtype=int32, numpy=array([13, 10, 22, 17])>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
  (<tf.Tensor: shape=(4,), dtype=int32, numpy=array([29, 33, 39, 49])>, <tf.Tensor: shape=(), dtype=int32, numpy=3>)
  ```

  **tf.GradientTape**

  with结构记录计算过程，gradient求出张量梯度

  ```
  with tf.GradientTape() as tape:
      若干计算过程
  grad = tape.gradient(函数,对谁求导)
  ```

  ```python
  with tf.GradientTape() as tape:
      w=tf.Variable(tf.constant(3.0))
      loss = tf.pow(w, 2)
  grad = tape.gradient(loss, w)
  print(grad)
  ```

  ```
  运行结果：
  tf.Tensor(6.0, shape=(), dtype=float32)
  ```

  **enumerate**

  python内建函数，遍历每个元素（列表、元组、字符串），组合为：**索引 元素**， 常在for循环使用.

  ```
  seq=['zero', 'one', 'two', 'three']
  for i, element in enumerate(seq):
      print(i, element)
  ```

  ```
  运行结果：
  0 zero
  1 one
  2 two
  3 three
  ```

  **tf.one_hot**

  独热编码：在分类问题中用于做标签，标记类别：1：是，0：非.

  e.g.

  |          | 0狗尾鸢尾 | 1杂色鸢尾 | 2弗吉尼亚鸢尾 |
  | :------: | :-------: | :-------: | :-----------: |
  |   标签   |     1     |           |               |
  | 独热编码 |    0.     |    1.     |      0.       |

  tf.one_hot(数据， depth=几分类)函数将待转换数据转换为one-hot形式数据输出.

  ```
  classes = 3
  labels = tf.constant([1, 0, 2])#输入元素最小是0，最大是2
  output = tf.one_hot(labels, depth=classes)
  print(output)
  ```

  ```
  运行结果：
  tf.Tensor(
  [[0. 1. 0.]
   [1. 0. 0.]
   [0. 0. 1.]], shape=(3, 3), dtype=float32)
  
  ```

  **tf.nn.softmax(x)**

  使输出符合概率分布
  $$
  Softmax(y_i)=\frac{e^{y_i}}{\sum_{j=0}^{n}e^{y_i}}
  $$
  

  n分类的n个输出通过softmax()函数，符合概率分布
  $$
  \forall x \quad P(X=x)\in [0,1],\quad \sum_{x}P(X=x)=1
  $$
  也就是每个输出值变为0~1之间概率值，概率值和为1.

  ```
  import tensorflow as tf
  y = tf.constant([1.01, 2.01, -0.66])
  y_pro = tf.nn.softmax(y)
  print("After softmax,y_pro is:", y_pro)
  ```

  ```
  输出结果：
  After softmax,y_pro is: tf.Tensor([0.25598174 0.69583046 0.0481878 ], shape=(3,), dtype=float32)
  ```

  **assign_sub**

  用于参数自更新（自减）

  调用assign_sub前，先用tf.Variable定义变量w为可训练（可自更新）.

  w.assign_sub(w要自减的内容)

  e.g.

  w.assign_sub(1):  w-=1 i.e. w=w-1

  **tf.argmax(张量名，axis=操作轴)**

  返回张量沿指定维度最大值的索引

  ```
  import tensorflow as tf
  import numpy as np
  test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
  print(test)
  print(tf.argmax(test, axis=0)) #返回每一列最大值索引
  print(tf.argmax(test, axis=1)) #返回每一行最大值索引
  ```

  ```
  运行结果：
  [[1 2 3]
   [2 3 4]
   [5 4 3]
   [8 7 2]]
  
  tf.Tensor([3 3 1], shape=(3,), dtype=int64)
  tf.Tensor([2 2 0 0], shape=(4,), dtype=int64)
  ```

### 鸢尾花数据集读入

- 数据集介绍

  共150组数据，每组4个输入特征：花萼长、花萼宽、花瓣长、花瓣宽，同时给出每组特征对应鸢尾花类别，分别用0,1,2表示.

- 从sklearn包datasets读入数据集

- 数据读入及查看

```
#鸢尾花数据集读入
from sklearn import datasets
from pandas import DataFrame
import pandas as pd

x_data = datasets.load_iris().data #.data 返回iris数据集所有输入特征
y_data = datasets.load_iris().target # .target 返回iris数据集所有标签
print('x_data from datasets: \n', x_data)
print('y_data from datasets: \n', y_data)

x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
pd.set_option('display.unicode.east_asian_width', True) # 设置列名对齐
print('x_data add index: \n', x_data)

x_data['类别'] = y_data  #新加一列，列标签为’类别‘，数据为y_data
print('x_data add a column: \n', x_data)
```

