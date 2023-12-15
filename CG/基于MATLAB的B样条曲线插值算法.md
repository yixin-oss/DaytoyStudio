---
title: 基于MATLAB的B样条曲线插值算法
tags:
- B样条
- 插值
- MATLAB
- 样条工具箱
categories: 计算几何
mathjax: true
---

# 问题

​		给定$5$个数据点$\boldsymbol Q_0=(-2,1), \boldsymbol Q_1=(0,0),\boldsymbol Q_2=(1,0), \boldsymbol Q_3=(3,5), \boldsymbol Q_4=(4,2)$, 结合累加弦长参数化方法及$3$次样条插值方法给出一条$3$次B样条曲线通过这$5$个点.

# 工具

​		为了解决上述问题, 我们采用MATLAB样条工具箱中的**spapi**​函数进行B样条曲线插值. 下面对需要用到的样条工具箱中的函数进行一个简要的说明, 关于样条工具箱更多的功能介绍可翻看之前的博客[Matlab样条工具箱及曲线拟合_matlab 拟合工具箱_Daytoy Studio的博客-CSDN博客](https://blog.csdn.net/yixon_oss/article/details/130719464?spm=1001.2014.3001.5501)

| 名称  |          功能          |
| :---: | :--------------------: |
| spapi |   插值生成B样条函数    |
| fnval | 计算某点处样条函数的值 |
| fnder |    求样条函数的导数    |
| fnplt |      画样条曲线图      |

# 算法步骤

- 将输入数据点进行累加弦长参数化, 假设有$m$个点$\{Q_i\}_{i=1}^{m}$, 则有

$$
t_1=0,t_m=1,t_i=t_{i-1}+\| Q_i-Q_{i-1}\|/\sum_{j=1}^m\|Q_j-Q_{j-1}\|,i=2,...,m-1.
$$

**累加弦长法**是目前最常用的参数化方法, 它反映了数据点按弦长的分布情况, 其它参数化方法还包括均匀参数化, 向心参数化等.

- 利用工具箱函数**spapi**进行$3$次B样条插值, 并可以画出图像验证生成的曲线是否严格经过已知数据点;
- 为了便于后续的应用, 这里增加了一步求导数, 并将其以切矢的形式画出验证是否与曲线相切.

# 实验结果

![Binterpolation](https://gitee.com/yixin-oss/blogImage/raw/master/img/Binterpolation.png)

至此, 上述一个简单的B样条曲线插值问题得以解决, 这里主要是为了给后续的工程应用进行一个铺垫, 下面给出具体的代码.

# Code

```matlab
function [T] = CumuPara(P)

%累加弦长参数化
%P输入数据点, 以列向量组成的矩阵, T累加弦长参数化得到的参数

m=size(P,2);%数据点个数
T=zeros(1,m);
sum_chord=0;
for j=1:m-1
    sum_chord=sum_chord + norm(P(:,j+1)-P(:,j),2);
end
chord=0;
for i=2:m-1
    chord=chord+norm(P(:,i)-P(:,i-1),2);
    T(i)=chord/sum_chord;
end
T(m)=1;
end
```

```matlab
% 输入数据点
P=[-2,0,1,3,4;...
    1,0,0,5,2];
figure;
scatter(P(1,:),P(2,:),'bo');
hold on

% 累加弦长参数化
T=CumuPara(P);

% B样条阶数
k=4;

% 3次B样条插值
sp=spapi(k,T,P);

% 画图
fnplt(sp,'r-');
hold on

% 计算导数值
dsp=fnder(sp,1);
dp=fnval(dsp,T);

% 切矢画图
len=size(T,2);

for i=1:len
    dir=dp(:,i)/norm(dp(:,i));
    plot([P(1,i),P(1,i)+dir(1)],[P(2,i),P(2,i)+dir(2)],'k'); % 切矢
    hold on
end
```

# Reference

```latex
@book{王仁宏2008计算几何教程,
  title={计算几何教程},
  author={王仁宏 and 李崇君 and 朱春钢},
  publisher={计算几何教程},
  year={2008},
}
```

