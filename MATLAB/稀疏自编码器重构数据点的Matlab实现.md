

以下代码源自MathWorks官方文档[trainAutoencoder](https://www.mathworks.com/help/deeplearning/ref/trainautoencoder.html?s_tid=gn_loc_drop)的Examples.目标是实现1000个数据点的重构,并给出重构与原数据对比图.

<!--more-->

```matlab
%Reconstruct Observations Using Sparse Autoencoder
%用稀疏自编码器重建对象
% 生成训练集，1000个点
rng(0,'twister'); %保证可重复性
n = 1000;         %训练数据个数
r = linspace(-10,10,n)';      %训练集区间
x = 1 + r*5e-2 + sin(r)./r + 0.2*randn(n,1); %取值
% 使用训练数据训练自动编码器
hiddenSize = 25;    %隐藏单元数
autoenc = trainAutoencoder(x',hiddenSize,...
        'EncoderTransferFunction','satlin',... % 编码函数
        'DecoderTransferFunction','purelin',...% 解码函数
        'L2WeightRegularization',0.01,...      % L2权重调整器的系数
        'SparsityRegularization',4,...         % 稀疏正则项的系数   
        'SparsityProportion',0.10);            % 稀疏比例
% 生成测试集,1000个点
n = 1000;
r = sort(-10 + 20*rand(n,1));
xtest = 1 + r*5e-2 + sin(r)./r + 0.4*randn(n,1);
%利用训练后的网络对测试集进行预测
xReconstructed = predict(autoenc,xtest');
% 绘制结果图
figure;
subplot(2,2,1);plot(xtest,'r.');title('测试数据')
subplot(2,2,2);plot(xReconstructed,'go'); title('重构数据')
subplot(2,2,[3,4]);
plot(xtest,'r.');% 红色圆点代表原数据
hold on
plot(xReconstructed,'go'); % 绿色圆圈代表新建数据
title('新旧数据对比')
```

**运行结果**

trainAutoencoder函数在运行时会显示训练窗口.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/训练窗口.jpg)

重构数据与原数据前后对比.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/compare.jpg)

从图中可以看到重构数据与原数据的分布趋势基本相同，数据点位置与原数据也基本重合，稀疏自编码器对数据点的重构效果较好。