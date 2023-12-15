---
title:人工智能: Tensorflow2笔记(三)
tags:
- 深度学习
- Tensorflow
- 人工智能
categories: Tensorflow学习笔记
---

# 神经网络优化

## 预备知识

**tf.where()**

条件语句，真返回A，假返回B

```
import tensorflow as tf
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b) #若a>b,返回a对应位置元素
#否则返回b对应位置元素
print('c:', c)
```

```
result:
c: tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
```

<!--more-->

**np.random.RandomState.rand(维度)**

返回一个[0,1)之间的随机数，维度为空，则返回标量

````
import numpy as np
rdm = np.random.RandomState(seed=1) #随机数种子seed=常数，每次生成随机数相同
a = rdm.rand() #返回一个随机标量
b = rdm.rand(2, 3) #返回维度2行3列随机数矩阵

print('a:', a)
print('b:', b)
````

````
result:
a: 0.417022004702574
b: [[7.20324493e-01 1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02 1.86260211e-01]]
````

**np.vstack(数组1，数组2)**

两个数组按垂直方向叠加

```
a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])
c1 = np.vstack((a1, a2))
print('c1:\n', c1)
```

```
result:
c1:
 [[1 2 3]
 [4 5 6]]
```

**np.mgrid(起始值：结束值：步长，起始值：结束值：步长，...)**

[起始值 结束值)

返回若干组维度相同的等差数组

**x.reval()**

将x变成一维数组，即把x拉直

**np.c_[数组1，数组2]**

返回的间隔数值点配对

**以上3个函数经常一起使用，可以生成网格坐标点**

```
import numpy as np
x, y = np.mgrid[1:3:1, 2:4:0.5]
grid = np.c_[x.ravel(), y.ravel()]
print('x:', x)
print('y:', y)
print('grid:\n', grid)
```

```
result:
x: [[1. 1. 1. 1.]
 [2. 2. 2. 2.]]
y: [[2.  2.5 3.  3.5]
 [2.  2.5 3.  3.5]]
 grid:
 [[1.  2. ]
 [1.  2.5]
 [1.  3. ]
 [1.  3.5]
 [2.  2. ]
 [2.  2.5]
 [2.  3. ]
 [2.  3.5]]
# 构成了网格坐标点
```

## 神经网络(NN)复杂度

NN复杂度：多用NN层数和NN参数个数表示

- 空间复杂度

层数 = 隐藏层层数 + 1个输出层

总参数 = 总w + 总b

- 时间复杂度

乘加运算次数

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1607497708(1).jpg"  />

图为2层NN.

第一层参数w是3行4列的，加4个偏置项b，第二层参数是4行2列的，两个偏置项，共26个参数.

每条权重线代表1次乘加运算，第1层12，第2层8次，共20次乘加运算.

## 指数衰减学习率

先用较大学习率，快速得到较优解，逐步减小学习率，得到最优解

指数衰减学习率 = 初始学习率*学习率衰减率^(当前轮数/多少轮衰减一次)

```
import tensorflow as tf
epoch = 40
w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr_base = 0.2
lr_decay = 0.99
lr_step = 1
for epoch in range(epoch):
    lr = lr_base * lr_decay ** (epoch / lr_step)
    with tf.GradientTape() as tape:
        loss = tf.square(w+1)
    grads = tape.gradient(loss, w)
    w.assign_sub(lr * grads)
    print('After %s epoch, w is %f, loss is %f, lr is %f' % (epoch, w.numpy(), loss, lr))
```

```
result:
After 0 epoch, w is 2.600000, loss is 36.000000, lr is 0.200000
After 1 epoch, w is 1.174400, loss is 12.959999, lr is 0.198000
After 2 epoch, w is 0.321948, loss is 4.728015, lr is 0.196020
After 3 epoch, w is -0.191126, loss is 1.747547, lr is 0.194060
After 4 epoch, w is -0.501926, loss is 0.654277, lr is 0.192119
After 5 epoch, w is -0.691392, loss is 0.248077, lr is 0.190198
After 6 epoch, w is -0.807611, loss is 0.095239, lr is 0.188296
After 7 epoch, w is -0.879339, loss is 0.037014, lr is 0.186413
After 8 epoch, w is -0.923874, loss is 0.014559, lr is 0.184549
After 9 epoch, w is -0.951691, loss is 0.005795, lr is 0.182703
After 10 epoch, w is -0.969167, loss is 0.002334, lr is 0.180876
... ...
```

可以看出，随着迭代轮数的增加，学习率在指数衰减.

## 激活函数

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/1607499785(1).jpg)

激活函数大大提升了模型表达能力

- 优秀的激活函数
  - 非线性：不会被单层网络替代，多层网络可逼近所有函数
  - 可微性：优化器大多用梯度下降更新参数
  - 单调性：保证单层网络损失函数是凸函数
  - 近似恒等性：f(x)≈x，网络更稳定

- 激活函数输出值范围：
  - 激活函数输出有限值，基于梯度的优化方法更稳定;
  - 无限值时，参数初始值对模型影响大，要使用更小的学习率.

$$
Sigmoid:f(x)=\frac{1}{1+e^{-x}}
$$

特点：易造成梯度消失；输出非0均值，收敛慢；幂运算复杂，训练时间长.
$$
Tanh:f(x)=\frac{1-e^{-2x}}{1+e^{-2x}}
$$
特点：输出是0均值；易造成梯度消失；幂运算复杂，训练时间长.
$$
Relu:f(x)=max(x,0)=\{_{x,x>=0}^{0,x<0}
$$
优点：在正区间内解决了梯度消失问题；只需判断输入是否大于0，计算速度快；收敛速度快于sigmoid和tanh

缺点：输出非0均值，收敛慢；Dead ReIU问题：输入特征是负数时，激活函数输出是0，反向传播梯度是0，参数无法更新，导致神经元死亡.
$$
Leaky Relu:f(x)=max(\alpha x,x)
$$
实际操作中，选择relu做激活函数的网络会更多.

对于使用激活函数的建议：

- 首选relu函数
- 学习率设置较小值
- 输入特征标准化，即满足以0为均值，1为标准差的正态分布
- 初始参数中心化，即随机生成的参数满足0为均值，sqrt(2/当前层输入特征个数)为标准差的正态分布.

## 损失函数(loss)

预测值与已知答案的差距

主流的三种损失函数：均方误差、自定义、交叉熵

**均方误差mse：**
$$
MSE(y_,y)=\frac{\sum_{i=1}^{n}(y-y_{-})^{2}}{n}
$$
**loss_mse = tf.reduce_mean(tf.square(y_-y))**

构建例子来理解：预测日销量.

x1, x2是影响日销量的因素，预先采集每日x1, x2, y_(已知答案)

构造数据集X，Y_=x1+x2  噪声：-0.05~0.05 拟合可以预测销量的函数

```
import tensorflow as tf
import numpy as np
SEED = 23455 #随机数种子，保证生成的随机数是一样的

rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x] #制造的数据集
# rdm.rand()生成0~1随机数 /10 变成0~0.1 -0.05 变成-0.05~0.05
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 15000
lr = 0.02

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss_mse = tf.reduce_mean(tf.square(y_-y))
    grads = tape.gradient(loss_mse, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print('After %d training steps,w1 is ' %(epoch) )
        print(w1.numpy(), '\n')
print('Final w1 is :', w1.numpy())
```

```
result:
Final w1 is : [[1.004296  ]
 [0.99483895]] 
```

输出参数值接近1，最后得到答案y=1.004296x1+0.99483895x2，而标准答案是y=x1+x2，拟合正确.

**交叉熵**

表示两个概率分布之间的距离.交叉熵越大，概率分布越远；交叉熵越小，概率分布越近.

交叉熵损失函数CE(Cross Entropy)
$$
H(y_,y)=-\sum y_{-}*lny
$$
e.g 二分类

已知答案y_=(1,0)，表示第一个事件发生概率为1，第二个事件发生概率为0，神经网络预测了两个结果 y1=(0.6,0.4),y2=(0.8,0.2),哪个更接近标准答案？
$$
H_{1}((1,0),(0.6,0.4))=-(1*ln0.6+0*ln0.4)=-(-0.511+0)=0.511\\
H_{2}((1,0),(0.8,0.2))=-(1*ln0.8+0*ln0.2)=-(-0.223+0)=0.2223\\
H_{1}>H_{2}
$$
y2预测更准确.

交叉熵计算公式**tf.losses.categorical_crossentropy(y_,y)**

```
import tensorflow as tf
loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
print('loss_ce1:', loss_ce1)
print('loss_ce2:', loss_ce2)
```

```
result:
loss_ce1: tf.Tensor(0.5108256, shape=(), dtype=float32)
loss_ce2: tf.Tensor(0.22314353, shape=(), dtype=float32)
```

**softmax与交叉熵结合**

输出先经过softmax符合概率分布，再计算y与y_的交叉熵损失函数，将两者结合的函数：

**tf.nn.softmax_cross_entropy_with_logits(y_,y)**

```
import tensorflow as tf
import numpy as np
y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pro = tf.nn.softmax(y)
loss_ce1 = tf.losses.categorical_crossentropy(y_, y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)

print('分步计算结果：\n', loss_ce1)
print('结合计算结果：\n', loss_ce2)
```

```
result:
分步计算结果：
 tf.Tensor(
[1.68795487e-04 1.03475622e-03 6.58839038e-02 2.58349207e+00
 5.49852354e-02], shape=(5,), dtype=float64)
结合计算结果：
 tf.Tensor(
[1.68795487e-04 1.03475622e-03 6.58839038e-02 2.58349207e+00
 5.49852354e-02], shape=(5,), dtype=float64)
```

## 欠拟合与过拟合

**欠拟合**

模型不能有效拟合数据集,是对现有数据集学习不彻底.

**过拟合**

模型对当前数据拟合得太好了，但对新数据难以判断，泛化能力差.

**欠拟合的解决**

- 增加输入特征项，给网络更多维度的输入特征
- 增加网络参数
- 减少正则化参数

**过拟合的解决**

- 数据清洗，减少噪声，使数据集更纯净.
- 增大训练集
- 采用正则化
- 增大正则化参数

**正则化缓解过拟合**

即在损失函数中引入模型复杂度指标，利用给W加权值，弱化训练数据的噪声(一般不正则化b)

损失函数变成两部分的和：

loss = loss(y与y_)+REGULARIZER * loss(w)

第一部分是模型中所有参数的损失函数，描述了预测结果与正确结果间的差距，如交叉熵、均方误差.

第二部分是参数的权重，用超参数REGULARIZER给出参数w在总loss中比重 .

loss(w)计算有两种方法

- L1正则化

$$
loss_{L1}(w)=\sum_{i} |w_{i}|
$$

大概率使很多参数变为0，该方法通过稀疏参数，即减少参数的数量，降低复杂度.

- L2正则化

$$
loss_{L2}(w)=\sum_{i}|w_{i}^{2}|
$$

使参数接近0但不为0，即通过减小参数值大小降低复杂度.





