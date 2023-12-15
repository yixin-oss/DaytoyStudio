---
title:人工智能: Tensorflow2笔记(二)
tags:
- 深度学习
- Tensorflow
- 人工智能
categories: Tensorflow学习笔记
---

## 神经网络实现鸢尾花分类

准备数据

- 数据集读入
- 数据集乱序
- 生成训练集和测试集,训练集，测试集不能有交集
- 配成（输入特征，标签）对，每次读入一部分(batch)

搭建网络

- 定义神经网络中所有可训练参数

参数优化

- 嵌套循环迭代，with结构更新参数，显示当前loss

测试效果

- 计算当前参数向后传播准确率，显示当前acc
- 准确率acc/损失函数loss可视化

<!--more-->

```
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
#从sklearn包datasets读入数据集：
from sklearn import datasets
x_data = datasets.load_iris().data  #返回iris数据集所有输入特征
y_data = datasets.load_iris().target #返回iris数据集中所有标签

#数据集乱序
np.random.seed(116) #使用相同的随机数种子，使输入特征/标签一一对应，即配对不会乱
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#数据集分出训练集，测试集,不能有交集
#打乱数据集中前120个作为训练集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x数据类型
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

#from_tensor_slices配成【输入特征、标签】对，每次喂入神经网络一部分数据(batch)
#每32对打包为一个batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#定义神经网络所有可训练参数
#输入特征是4，输入层为4个输入节点，只有一层网络，输出节点数=分类数，3分类
#参数w1是4行3列张量
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))

lr = 0.1 #学习率为0.1
train_loss_results = []  #将每轮loss记录下来，为后面画loss曲线提供数据
test_acc = [] #记录acc
Epoch = 500 #循环500次
loss_all = 0 #每轮分4个step，loss_all记录四个step生成的4个loss的和

#两层循环迭代更新参数
#第一层for循环是针对整个数据集循环，用epoch表示
#第二层for循环是针对batch的，用step表示
for epoch in range(Epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape: #with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1 #神经网络乘加运算
            y = tf.nn.softmax(y) #使输出y符合概率分布
            y_ = tf.one_hot(y_train, depth=3) #将标签转换为独热码格式，方便计算loss
            loss = tf.reduce_mean(tf.square(y_ -y)) #采用均方误差损失函数MSE
            loss_all += loss.numpy() #将每个step计算出的loss累加，为后续求loss平均值提供数据
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新
        w1.assign_sub(lr * grads[0]) #参数w1自更新
        b1.assign_sub(lr * grads[1]) #参数b1自更新
    #每个epoch 打印loss信息
    print('Epoch {}, loss: {}'.format(epoch, loss_all/4)) #120组数据，需要batch级别循环4次，除以4求得每次step迭代平均loss
    train_loss_results.append(loss_all / 4) #将4个step的loss求平均记录在此变量中
    loss_all = 0 #loss_all归零，为记录下一个epoch的loss做准备


    #测试部分
    #计算当前参数前向传播后准确率，显示当前acc
    #total_corrrect为预测对的样本个数， total_number为测试的总样本数，初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1 #y为预测结果
        y = tf.nn.softmax(y) #y符合概率分布
        pred = tf.argmax(y, axis=1) #返回y中最大值索引，即预测分类
        #将pred转换为y_test数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，correct=1，否则为0，将bool型转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        total_correct += int(correct) #将所有batch中correct数加起来
        #total_number为测试总样本数，即x_test行数，shape[0]
        total_number = x_test.shape[0]
    #总准确率为 total_correct / total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("test_acc:", acc)
    print('----------------------')


#绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.plot(train_loss_results, label='$loss$')
plt.legend()
plt.show()

#绘制Accuary曲线
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Accuary$')
plt.legend()
plt.show()
```

程序在PyCharm中运行.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/Epoch.jpg)

从图中可见，随着迭代次数增加，损失函数值逐渐减小，对测试集的预测准确率逐渐增大直至达到100%正确.下面两张图给出了损失函数及预测准确率的变化图像.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/Loss_Function_Curve.png)

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/Acc_Curve.png)





