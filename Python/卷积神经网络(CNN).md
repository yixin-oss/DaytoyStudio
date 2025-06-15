# 卷积神经网络(Convolution Neural Network)

## 引入卷积的原因

在用全连接网络处理图像时, 会存在以下两个大问题:

- #### 参数太多

  对于一个$1000\times 1000$的输入图像, 如果下一个隐藏层的神经元数目为$10^6$个, 采用全连接则有$1000\times 1000 \times 10^6=10^{12}$个权值参数, 随着隐藏层神经元数量的增多, 参数的规模也会急剧增加, 这会导致整个神经网络的训练效率非常低.

- #### 不利于空间结构的表达

  图像具有重要的空间结构信息, 相邻的像素通常具有某种相关性. 全连接网络将图像展开成一个向量处理, 破坏了其空间结构. 具体来说, 对于图像识别任务, 关键在于识别出对象的特征, 而不关心特征出现的具体位置, 但全连接的处理难以反映这种平移不变性.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240322104517616.png" alt="image-20240322104517616"  />

卷积神经网络是首个模仿人类视觉皮层中感受野的结构得到的模型.

- #### 局部连接

  隐藏层的每个神经元仅与图像中$10\times 10$的局部图像连接, 此时权值参数数量$10\times 10 \times 10^6=10^8$, 直接减少4个数量级.

![](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240322104517616.png)

- #### 参数共享

  假设在局部连接中隐藏层的每一个神经元连接的是一个$10 × 10$的局部图像，因此有$10 × 10$个权值参数，将这$10 × 10$个权值参数共享给剩下的神经元，也就是说隐藏层中$10^6$个神经元的权值参数相同，那么此时不管隐藏层神经元的数目是多少，需要训练的参数就是这$10×10$个权值参数, 大大降低了网络的训练难度.

  参数共享使得卷积网络获得了良好的平移不变性, 即对于图像识别任务, 无论识别特征在哪, 网络都能一致地识别出来.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240314191006760.png" alt="image-20240314191006760" style="zoom:67%;" />

![image-20240314191113054](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240314191113054.png)

卷积神经网络可以看作一个**特征提取器**, 就是在全连接网络的基础上, 加入卷积层和经过激活函数后加入池化层, 因此问题的关键在于卷积层和池化层的作用机制.

## 卷积

卷积其实是一种滤波的过程, 卷积核也就是滤波器. 当卷积核参数改变时, 提取的特征类型也会改变. 

### 单通道卷积运算

- 定义一个卷积核: 是一个小的矩阵, 作用是在图像中识别边缘, 线条等特征;
- 卷积核滑过图像: 卷积核初始化放置在图像左上角, 然后按照一定的步长在图像上滑动, 步长即为滑动距离；
- 计算点积: 将卷积核中的每个元素与图像中对应位置的像素值相乘, 然后将所有乘积相加;
- 生成新的特征图: 每次计算的点积结果被用来构建一个新的图像, 称为特征图;
- 重复上述过程: 多个不同的卷积核同时进行卷积操作, 这意味着我们会得到多个特征图, 每个特征图捕捉了原始图像中的不同特征.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/convolution.png" alt="convolution" style="zoom:80%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/convSobel.gif" alt="convSobel" style="zoom: 80%;" />

### 边缘检测1: Sobel卷积核及其转置

$$
S=
\begin{bmatrix}
-1 & -2 & -1\\
0 & 0 & 0\\
1 & 2 & 1
\end{bmatrix}
$$

![Android_sobel](https://gitee.com/yixin-oss/blogImage/raw/master/Img/Android_sobel.png)

### 边缘检测2: Laplace卷积核及其转置

$$
L=
\begin{bmatrix}
0 & -1 & 0\\
-1 & 4 & -1\\
0 & -1 & 0
\end{bmatrix}
$$

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/laplace.png" alt="laplace" style="zoom: 50%;" />

在卷积神经网络中, 卷积核不是手动设计出来的, 而是通过数据驱动的方式学习得到的, 也就是说卷积神经网络的训练实际是调整卷积核的参数.

### 全零填充(Zero-padding)

设输入数据的尺寸为$H\times W$, 卷积核的尺寸为$k_h\times k_w$, 当卷积核尺寸大于1时, 经过一次卷积之后得到的特征图尺寸为$(H-k_h+1)\times(W-k_w+1)$, 即输出特征图尺寸会不断缩小, 而过小的尺寸可能会导致信息的丢失, 因此引入填充技巧:

在输入数据进行卷积运算之前, 在四周补充一圈0再进行卷积, 使得输出结果与输入数据保持空间尺寸不变. 

该操作的另一个重要意义是**更充分地利用输入数据的边缘信息**.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/convZeros.png" alt="img" style="zoom: 40%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/sketch.png" alt="sketch" style="zoom: 80%;" />



### 多通道卷积运算

对于RGB三通道的彩色图片输入样本, 定义卷积核组(也是三通道), 对每一个通道分别进行卷积, 然后三通道结果相加得到一个特征图, 即**滑动窗口变为滑动块**.

![img](https://gitee.com/yixin-oss/blogImage/raw/master/Img/08cc2178a2694e65ba7addb1b423bd62.png)

![img](https://gitee.com/yixin-oss/blogImage/raw/master/Img/1845730-5ca69abe03f57d72.gif)

## 激活函数ReLU

经卷积运算输出的结果仍然是线性的, 因此要利用激活函数叠加非线性, 通常选用ReLU(修正线性单元), 负值变为0, 正值不变. 它的特点是求梯度简单, 计算速度快.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/1845730-8898029e12b1bcf6.png" alt="img" style="zoom:80%;" />

## 池化(Pooling)

卷积层的设定为: 输出特征图的空间尺寸=输入特征图的空间尺寸, 这样会带来一些缺陷.

1. 空间尺寸不变, 卷积层的运算量会很大, 增加计算代价；
2. 卷积网络结构最后要通过全连接层, 如果空间尺寸不变, 全连接层的权重数量巨大, 导致训练过慢及过拟合；
3. 前面的卷积输出存在冗余信息, 如果空间尺寸不变, 冗余会一直存在.

因此需要一种技巧减小空间尺寸, 同时保持深度不变. 

**池化**：选择一个窗口(通常$2\times 2$)在特征图上滑动, 将每个局部窗口的数据进行融合, 得到一个输出数据. 这一过程可以视为数据的提纯/增强.

- 最大池化: 局部窗口最大值.
- 平均池化: 局部窗口平均值.

通常采用最大池化, 因为ReLU激活函数把负值变为0, 正值不变, 所以激活值越大, 说明神经元对输入局部窗口数据的反应越激烈, 提取的特征更强.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/1845730-9338f08f69f43297.jpg" alt="img" style="zoom:80%;" />

经过池化计算的图像, 可以看作原特征图的"低像素版". 最大池化能够保留最强烈的特征, 降低数据量.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/v2-284a0e4cb88b9cece5b473f6cef614e1_b.webp" alt="动图" style="zoom:80%;" />

## 全连接网络(Dense)

经卷积层和池化层处理后的特征图尺寸与原输入数据相比已有大幅减小, 且包含了提取到的重要特征, 因此可以输入到全连接网络中进行训练, 即将3D特征图拉伸为1D向量, 然后与前述全连接网络的操作是类似的.

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/DvWBlldHDO6jSmpxNFdoIcp6azyIzdgBG9gicOvL0niaYMd4PjuA3fyIzOLf7icbRwRjIicRUYf3gJx0EwnoFGeSmQ/640?wx_fmt=jpeg&amp;wxfrom=5&amp;wx_lazy=1&amp;wx_co=1" alt="图片" style="zoom: 50%;" />

## Softmax

$$
p_i=\frac{e^{y_i}}{\sum_{i=1}^n e^{y_i}},
$$

将实数向量转换成概率分布, 概率和为1.  Softmax是分类任务中的常用做法, 可以直观地给出每个类别的预测概率.

### 交叉熵损失函数(Cross Entropy Loss)

对于分类问题, 常用的损失函数是交叉熵损失. 交叉熵损失可以衡量预测的概率分布与真实分布之间的差异.
$$
L=\frac{1}{N}\sum_iL_i=-\frac{1}{N}\sum_i\sum_{c=1}^My_{ic}\log(P_{ic}),
$$
其中$M$是类别数量, $y_{ic}$是符号函数, 取值0或1, $P_{ic}$是样本$i$属于类别$c$的预测概率.

## 完整链条



<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/process.png" alt="process" style="zoom: 150%;" />

## 卷积神经网络的反向传播

- 池化层

最大池化的反向传播: 只有池化窗口中最大值会传播梯度, 其它位置梯度为0, 只需要将上一层梯度值传递给最大值位置, 其它位置梯度保持为0.

- 卷积层(简单情形推导)

$$
w_i^*=w_i-\alpha\times \frac{\partial L}{\partial w_i}
$$

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/5149e97e19d99f03e7a1394ffe47af7.jpg" alt="5149e97e19d99f03e7a1394ffe47af7" style="zoom:100%;" />

在实际应用中, 反向传播可在多种框架中通过自动微分过程计算, 而不需要人为的编写. 接下来, 将进入到TensorFlow实现经典卷积神经网络的实战.

## LeNet5: MNIST手写数据集识别

网络结构

![image-20240324165120343](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240324165120343.png)

为了与现有的环节对应, 便于网络的实现, 这里我们将其简化为如下所示的结构

![1711074701432](https://gitee.com/yixin-oss/blogImage/raw/master/Img/1711074701432.png)

我们在Jupyter Notebook中基于TensorFlow搭建卷积神经网络的框架, 并对每一部分展示输出结果.


```python
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, datasets, Sequential
import matplotlib.pyplot as plt
```


```python
# 载入mnist手写数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
```


```python
# 训练集规模
print(x.shape)
print(y.shape)
```

    (60000, 28, 28)
    (60000,)

```python
# 测试集规模
print(x_test.shape)
print(y_test.shape)
```

    (10000, 28, 28)
    (10000,)

```python
# 查看数据集
# 指定窗口大小10x10英寸
# 绘制5行5列共25张训练集图片
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x[i])
plt.show()
```


![output_7_0](https://gitee.com/yixin-oss/blogImage/raw/master/Img/output_7_0.png)

```python
# 预处理函数: 转换数据类型及范围x:[-1~1], y:[0~9]
def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32)/255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y
```


```python
# 将标签数据压缩到1维
y = tf.reshape(y, [-1])
y_test = tf.reshape(y_test, [-1])
```


```python
# 构建数据集对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 批量数据Batch
train_db = train_db.shuffle(1000).map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)
```


```python
# 网络搭建
def main():
    # 网络层级
    network = Sequential([
        # 第一层6个3x3卷积核+Maxpooling+ReLU
        layers.Conv2D(6, kernel_size=[3, 3], strides=1, padding='valid'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'),
        layers.ReLU(),
        # 第二层16个3x3卷积核+Maxpooling+ReLU
        layers.Conv2D(16, kernel_size=[3, 3], strides=1, padding='valid'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'),
        layers.ReLU(),
        # flatten=>全连接层
        layers.Flatten(),
        # 全连接层
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10)
        ])
    # build网络模型
    network.build(input_shape=[None, 28, 28, 1])
    # 统计网络信息
    network.summary()
    # 优化器Adam
    optimizer = optimizers.Adam(learning_rate=1e-4)
    # 交叉熵损失函数
    criteon = losses.CategoricalCrossentropy(from_logits=True)

    acc_history = []
    Epoch_history = []
    # 训练50次
    steps = 50
    for epoch in range(steps):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # 插入通道维度 =>[b,28,28,1]
                x = tf.expand_dims(x, axis=3)
                out = network(x)
                # 将整数型的标签张量转换为独热编码张量, 其中每个类别用长度为depth的项链表示, 类别索引处值为1, 其余位置值为0
                y_onehot = tf.one_hot(y, depth=10)
                # 计算损失函数
                loss = criteon(y_onehot, out)

            # 计算梯度
            grads = tape.gradient(loss, network.trainable_variables)
            # grads: 包含损失函数对模型参数的梯度信息列表
            # zip将梯度列表和变量列表进行一一对应, 形成一个梯度-变量的元组列表
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

        # 测试集测试准确率
        total_correct, total = 0, 0
        for x, y in test_db:
            x = tf.expand_dims(x, axis=3)
            out = network(x)
            prob = tf.nn.softmax(out, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total
        acc_history.append(acc)
        print(epoch, 'acc:', acc)

    # 绘制训练次数与准确率的关系图
    plt.plot(range(1,steps + 1), acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuary')
    plt.title('Accuary vs. Epoch')
    plt.show()

    # 保存模型
    print('save total model.')
    network.save('model.keras')
    

     
if __name__ == '__main__':
    main()


```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_3"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)           │              <span style="color: #00af00; text-decoration-color: #00af00">60</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ re_lu_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">880</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)            │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ re_lu_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)            │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">400</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">120</span>)                 │          <span style="color: #00af00; text-decoration-color: #00af00">48,120</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">84</span>)                  │          <span style="color: #00af00; text-decoration-color: #00af00">10,164</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">850</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">60,074</span> (234.66 KB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">60,074</span> (234.66 KB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>
    0 acc: 0.7431
    1 acc: 0.8924
    2 acc: 0.9237
    3 acc: 0.9383
    4 acc: 0.9454
    5 acc: 0.9509
    6 acc: 0.9566
    7 acc: 0.9605
    8 acc: 0.9626
    9 acc: 0.9654
    10 acc: 0.9674
    11 acc: 0.9687
    12 acc: 0.9697
    13 acc: 0.9705
    14 acc: 0.972
    15 acc: 0.9724
    16 acc: 0.9741
    17 acc: 0.9749
    18 acc: 0.9755
    19 acc: 0.9755
    20 acc: 0.9765
    21 acc: 0.977
    22 acc: 0.9772
    23 acc: 0.9779
    24 acc: 0.9782
    25 acc: 0.9786
    26 acc: 0.979
    27 acc: 0.9795
    28 acc: 0.9801
    29 acc: 0.981
    30 acc: 0.9807
    31 acc: 0.9814
    32 acc: 0.981
    33 acc: 0.9813
    34 acc: 0.9816
    35 acc: 0.9817
    36 acc: 0.9821
    37 acc: 0.9816
    38 acc: 0.982
    39 acc: 0.9827
    40 acc: 0.9824
    41 acc: 0.9827
    42 acc: 0.9831
    43 acc: 0.9832
    44 acc: 0.9833
    45 acc: 0.9839
    46 acc: 0.984
    47 acc: 0.9842
    48 acc: 0.9845
    49 acc: 0.9848
    save total model.

![output_11_6](https://gitee.com/yixin-oss/blogImage/raw/master/Img/output_11_6.png)

可以看到训练50次, 模型在测试集上的识别准确率能达到约$98.5\%$, 可见经典的LeNet5对手写数据集的识别效果是非常好的. 最后, 我们每次随机从测试集中选取一张图片交给训练好的网络进行识别, 识别结果与人为观察结果保持一致.

```python
# 加载模型
print('load model from file')
model = tf.keras.models.load_model('model.keras')
model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 单张mnist图片识别测试
import random
indx = random.randint(0, 10000)
indy = random.randint(0, 10000)
print([indx, indy])
x, y = x_test[indx], y_test[indy]
plt.figure(figsize=(1, 1))
plt.imshow(x)
plt.show()
x_val, y_val = preprocess(x, y)
x_val = tf.expand_dims(x_val, axis=0)
out = model(x_val)
prob = tf.nn.softmax(out, axis=1)
pred = tf.argmax(prob, axis=1).numpy()[0]
# pred = tf.cast(pred, dtype=tf.int32)
print(pred)
```

    load model from file
    [7846, 100]


![output_12_1](https://gitee.com/yixin-oss/blogImage/raw/master/Img/output_12_1.png)
    


    0

尽管LeNet5对单通道的手写数据集有非常好的识别效果, 一旦升级到彩色三通道的数据集图片, LeNet5就会难以招架, 同样训练50次, 在测试集上的准确率仅能达到$40\%-60\%$左右.

## VGG: CIFAR10彩色图片分类

- VGG的一个核心理念是多个小的卷积核的叠加可以代替一个大的卷积核, 不会牺牲对信息处理的能力, 同时减少了训练参数. 例如, 2个$3\times 3$的卷积核堆叠的感受野大小相当于1个$5\times 5$的卷积核, 而3个$3\times 3$卷积核的堆叠获取到的感受野相当于1个$7\times 7$的卷积核, 这样可以使模型深度加深, 增加非线性映射, 学习和表示能力变强.

- 假设图片尺寸为$28\times 28$, 使用$5\times 5$卷积核对其进行卷积, 且stride=$1$, 得到特征图尺寸$(28-5)/1+1=24$; 

  使用2个$3\times 3$卷积核, stride=$1$, $28=>26=>24$, 所以2个$3\times 3$卷积后的感受野和1个$5\times 5$卷积核的结果是一样的. 参数量$25=>18$显著减小.

- 在VGG网络结构中, 均采用$3\times 3$小卷积核, $2\times 2$最大池化, 非线性映射均采用ReLU函数.

下面, 我们给出VGG的简化结构示意图.

![e42fb78d-b34e-4cdb-aecc-1fe3b50927c1](https://gitee.com/yixin-oss/blogImage/raw/master/Img/e42fb78d-b34e-4cdb-aecc-1fe3b50927c1.png)

从图中可以很明显的看到, 网络结构中每个子单元均是经历两次卷积后输入到最大池化层中, 需要注意的是每次卷积后都要先进行ReLU激活(图中未标注), 由于卷积采用全零填充, 因此这些卷积层均不会改变输入特征图的尺寸, 而$2\times 2$的最大池化层会导致每次特征图尺寸减半, 因此到第五套子单元后, 特征图的平面尺寸缩为$1\times 1$, 有512个channels, 将它们拉直即可输入到全连接网络层中. 由于我们采用的是CIFAR10十分类数据集, 因此这里的输出层个数应设为10(图中的100对应的是100类的问题).




```python
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import losses, layers, optimizers, datasets, Sequential
import matplotlib.pyplot as plt
```


```python
# 载入cifar10手写数据集
(x, y), (x_test, y_test) = datasets.cifar10.load_data()
```


```python
class10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```


```python
# 预处理函数: 转换数据类型及范围x:[0~1], y:[0~9]
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y
```


```python
# 将标签数据压缩到1维
y = tf.reshape(y, [-1])
y_test = tf.reshape(y_test, [-1])
print(y.shape, y_test.shape)
```

    (50000,) (10000,)

```python
# 构建数据集对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 批量数据Batch
train_db = train_db.shuffle(1000).map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)
```


```python
# 网络搭建
def main():
    # 网络层级
    network = Sequential([
        #第一组
        layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    
        #第二组
        layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    
        #第三组
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    
        #第四组
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    
        #第五组
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # flatten=>全连接层
        layers.Flatten(),
        # 全连接层
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
        ])
    # build网络模型
    network.build(input_shape=[None, 32, 32, 3])
    # 统计网络信息
    network.summary()
    # 优化器Adam
    optimizer = optimizers.Adam(learning_rate=1e-4)
    # 交叉熵损失函数
    criteon = losses.CategoricalCrossentropy(from_logits=True)

    acc_history = []
    Epoch_history = []
    # 训练50次
    steps = 50
    for epoch in range(steps):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                out = network(x)
                # 将整数型的标签张量转换为独热编码张量, 其中每个类别用长度为depth的项链表示, 类别索引处值为1, 其余位置值为0
                y_onehot = tf.one_hot(y, depth=10)
                # 计算损失函数
                loss = criteon(y_onehot, out)

            # 计算梯度
            grads = tape.gradient(loss, network.trainable_variables)
            # grads: 包含损失函数对模型参数的梯度信息列表
            # zip将梯度列表和变量列表进行一一对应, 形成一个梯度-变量的元组列表
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

        # 测试集测试准确率
        total_correct, total = 0, 0
        for x, y in test_db:
            out = network(x)
            prob = tf.nn.softmax(out, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total
        acc_history.append(acc)
        print(epoch, 'acc:', acc)

    # 绘制训练次数与准确率的关系图
    plt.plot(range(1,steps + 1), acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuary')
    plt.title('Accuary vs. Epoch')
    plt.show()

    # 保存模型
    print('save total model.')
    network.save('VGG_cifar10.keras')

     
if __name__ == '__main__':
    main()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_2"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │           <span style="color: #00af00; text-decoration-color: #00af00">1,792</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_23 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_26 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │       <span style="color: #00af00; text-decoration-color: #00af00">1,180,160</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_27 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │       <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_28 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │       <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_29 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │       <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">131,328</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │          <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">1,290</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">9,570,506</span> (36.51 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">9,570,506</span> (36.51 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>
    0 acc: 0.3288
    1 acc: 0.3952
    2 acc: 0.465
    3 acc: 0.4889
    4 acc: 0.5321
    5 acc: 0.5637
    6 acc: 0.5789
    7 acc: 0.6003
    8 acc: 0.6148
    9 acc: 0.6205
    10 acc: 0.6237
    11 acc: 0.6463
    12 acc: 0.6723
    13 acc: 0.6704
    14 acc: 0.6779
    15 acc: 0.6892
    16 acc: 0.69
    17 acc: 0.705
    18 acc: 0.7123
    19 acc: 0.7067
    20 acc: 0.7067
    21 acc: 0.711
    22 acc: 0.7164
    23 acc: 0.7246
    24 acc: 0.7268
    25 acc: 0.7271
    26 acc: 0.7159
    27 acc: 0.7165
    28 acc: 0.7219
    29 acc: 0.7331
    30 acc: 0.7356
    31 acc: 0.7365
    32 acc: 0.7373
    33 acc: 0.7234
    34 acc: 0.7233
    35 acc: 0.7263
    36 acc: 0.723
    37 acc: 0.7306
    38 acc: 0.721
    39 acc: 0.7189
    40 acc: 0.7227
    41 acc: 0.7373
    42 acc: 0.7238
    43 acc: 0.7317
    44 acc: 0.7363
    45 acc: 0.7286
    46 acc: 0.7331
    47 acc: 0.7339
    48 acc: 0.7339
    49 acc: 0.7235
    save total model.


![VGGacc](https://gitee.com/yixin-oss/blogImage/raw/master/Img/VGGacc.png)
    

可以看到训练50次, 模型在测试集上分类的准确率大约为$72.4\%$, 这个准确率虽然相比于LeNet5在CIFAR10上的表现有很大提升, 但还是未达到一个更高的标准. 由于这是一个演示, 网络结构也是简化版本的, 后期可以通过提高训练次数, 增加其他预处理步骤等进一步提高准确率.

```python
# 加载模型
print('load model from file')
model = tf.keras.models.load_model('VGG_cifar10.keras')
model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 单张mnist图片识别测试
import random
indx = random.randint(0, 10000)
indy = random.randint(0, 10000)
print([indx, indy])
x, y = x_test[indx], y_test[indy]
plt.figure(figsize=(1, 1))
plt.imshow(x)
plt.show()
x_val, y_val = preprocess(x, y)
x_val = tf.expand_dims(x_val, axis=0)
out = model(x_val)
prob = tf.nn.softmax(out, axis=1)
pred = tf.argmax(prob, axis=1).numpy()[0]
result = class10[pred]
print(result)
```

    load model from file
    [2146, 3621]


![png](https://gitee.com/yixin-oss/blogImage/raw/master/Img/output_10_1.png)

    bird

```
[1342, 9385]
```

![img](data:image/png;base64,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XL12Qj84rh5vUDLR9x0np6eICpazGpVhoUp7jjGC5RLGYcmFRYtznpOTTeGQjSN93zGOnuefPy77kOZCQLx4bs79993DvLGM2w4/OkaXON2uCUPg+OoxlZBs1hvGfiS5iUEjFSSIvuxK5HwpwuS8a4PiG5IuL2evNIe+szML+PlzCx68727e+OC9nD9Ycf7COZqqQltFJnHu3IL5alFIEYsWBFy5eszla0ccn6zZ9lu0THTdlmtHR1Tn5ujDA7QQtLF4w4e2BlMVVQjviSkx9APeFdrw0I/0w8C///uzbDYdwlikqVi0mvvvmqMlPPPMZY78CUM3cHR1w1APXH3uKjplhqFjGAZEjChRpDpyzEQXps8oPLnCU5O8YAH/JnZjihW+y9fwG0uBIcSptcbR9cWLlUrjfNypqJc9uHVRaRqcI0zc7RgSQkT6YSzcbitZb0o3qNrtaOgLBSVPclwpJYbBFV6bK1othehQPGspFEIERgf94FCyLD39WNimw+AQQrLtetabLeM4MI49IiW0L+c2jJ5x0j3fyWa/eB2+GQBfKulyM68T+dtdDF4j++pXv8p99933nT6N70p7+umnX9AJ9FJ25gBPKfHUU0/xpje9iaeffprlcvmdPqX/FNu1WH073znnzHq95t5770XKb67xcOamdCklr3vd6wBYLpe3DeA7+3a/82q1uqnnvSLJjzv23W93AL/N7EwCXlUVH/zgB1+yq/RWtf+s73zmnLY79tramRzhd+y1szuA32Z2B/DbzO4AfpvZmQT8scce44EHHqCuax566CE++9nPfqdP6VWxm5Eu/fEf//EbhATL8Su/8iuv3knkM2Yf/ehHs7U2//mf/3n+13/91/ze9743Hxwc5Oeee+47fWr/YXv3u9+dH3/88fz5z38+f+5zn8s//dM/ne+///682Wz2z3nnO9+Z3/ve9+Znnnlmf5ycnLxq53DmAH/b296WH3744f3fMcZ877335g996EPfwbN6bez555/PQP7Upz61f+yd73xn/rVf+7XX7DPP1JTunOPJJ598gXSnlJJ3vetdfPrTn/4OntlrYy+WLt3ZX/zFX3DhwgXe/OY384EPfICu6161zzxTxZMrV64QY+TSpUsvePzSpUt88Ytf/A6d1WtjLyVdCvDzP//zvOENb+Dee+/lX/7lX/it3/otnnrqKf76r//6VfncMwX47WQvJ136vve9b//7D/7gD3LPPffwkz/5k3z5y1/me77ne/7Dn3umpvQLFy6glPoGQf1vJt353Wg76dK///u//5aEhYceegiAL33pS6/KZ58pwK21vOUtb+GJJ57YP5ZS4oknnnhZ6c7vJss588gjj/Cxj32Mv/u7v/sG6dKXss997nMA3HPPPa/aSZwp++hHP5qrqsof/vCH8xe+8IX8vve9Lx8cHORnn332O31q/2H71V/91bxarfInP/nJF4RdXdflnHP+0pe+lH//938//9M//VP+t3/7t/zxj388v/GNb8zveMc7XrVzOHOA55zzH/3RH+X7778/W2vz2972tvyZz3zmO31Kr4pxw77ANx6PP/54zjnnr3zlK/kd73hHPjw8zFVV5e/93u/Nv/mbv/mqxuF3yqO3mZ2pNfyOvfZ2B/DbzO4AfpvZHcBvM7sD+G1mdwC/zewO4LeZ3QH8NrM7gN9mdgfw28zuAH6b2R3AbzP7/wF676dO0VXMOgAAAABJRU5ErkJggg==)

```
truck
```

可以看到模型会出现随机图片分类失误的情况.





