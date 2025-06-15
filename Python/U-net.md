---
title: U-net
---

# U-net

​		U-net最初是由Olaf Ronneberger等人在2015年的论文中首次开发的一种用于生物医学图像分割的卷积神经网络. 

​		其结构涉及编码(降采样)和解码(上采样)路径, 形成一种对称结果(U形状), 与传统的卷积神经网络不同, U-net不再包含全连接层而对图片做整体识别, 而是能够输出特征分割图实现对图片每个像素点的识别.

​		随着神经网络层数的加深, 我们会逐渐失去关于原始图像的信息, U-net的主要想法是在解码过程中对每一层重新注入图像(图中灰色箭头), 丰富处理过程图像的特征.

![e528761129f5514c2493d1871166da60.jpeg](https://gitee.com/yixin-oss/blogImage/raw/master/Img/e528761129f5514c2493d1871166da60.jpeg)



## 上采样(反卷积)

![上采样](https://gitee.com/yixin-oss/blogImage/raw/master/Img/902790cd6b15456999ff93809db6a5fc.gif)

## 评价函数

- DICE分数

$$
C_{DICE}=\frac{2\sum_i^N y_i\hat{y_i}}{\sum_i^N \hat{y_i}+\sum_i^N y_i}\in (0,1)
$$

分数越接近1表示模型预测的分割结果与真实分割的重合度越高.