---
title: 快速傅里叶变换(FFT)及应用实例
tags:
- FFT
- Matlab
categories: Matlab学习笔记
mathjax: true
---

## 实例一：离散点的FFT

(1)取$N=128$，生成实数序列$\{x(k)\}_{k=0}^{N-1}$;

(2)用FFT计算${x(k)}_{k=0}^{N-1}$的离散Fourier变换序列${X(j)}_{j=0}^{N-1}$;

(3)作出${x(k)}$和${X(j)}$的图并分析；

(4)设定$\delta_{0}>0$，将${|X(j)|}$中满足$|X(j)|<\delta_{0}$的数据全部置为0，再进行离散Fourier逆变换，将得到的数据与${x(k)}$比较；

(5)改变$\delta_{0}$的值，重复(4)，分析不同的$\delta_{0}$对逆变换所得数据的影响.

<!--more-->

**源代码如下：**

```matlab
function ex1601(N)
t=0:N-1;
x=randn(N,1)*20;%rndn
y=fft(x,N);
z=abs(y);
figure(1);
plot(t,x,'+',t,z,'o')
delta=input('请输入误差');
for i=0:N-1
    if z(i+1)<delta
        y(i+1)=0;
    end
end
z=real(ifft(y));
figure(2);
plot(t,x,'+',t,z,'o')
end
```

**运行结果及分析：**

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/FFT1-1.jpg)

1.本程序数据是随机产生的，“+”为原始数据，“o”为变换后的模的数据;

2.取$\delta_{0}=5$,将$\{|X(j)|\}$中满足$|X(j)|<\delta_{0}$的数据全部置为0，再进行离散Fourier逆变换."+"为原始数据，“o”为置0后变换得到的数据，与$\{x(k)\}$比较几乎重合;

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/FFT1-2.jpg)

3.取$\delta_{0}=50$,同样处理后得到的数据，与$\{x(k)\}$比较有些小误差；

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/FFT1-3.jpg)

4.取$\delta_{0}=100$，同样处理后得到的数据与$\{x(k)\}$比较误差清晰可见，但不很大.

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/FFT1-4.jpg)

由于数据源的不同，结果会有所差异.

## 实例二：卷积计算

(1)对于$N=128$,产生两个实数序列${x(k)}_{k=0}^{N-1}$和${y(k)}_{k=0}^{N-1}$；

(2)用直接方法计算${x(k)}_{k=0}^{N-1}$和${y(k)}_{k=0}^{N-1}$的卷积${z(k)}_{k=0}^{N-1}$；

(3)改用离散Fourier变换的思想，用FFT计算${z(k)}_{k=0}^{N-1}$

(4)比较两种算法所用时间.

**源代码如下：**

```matlab
function t=ex1602(N)

x=randn(N,1)*20;
y=randn(N,1)*20;
tic
for i=0:N-1
    z(i+1)=0;
    for j=0:i
        z(i+1)=z(i+1)+x(j+1)*y(i-j+1);
    end
    for j=i+1:N-1
        z(i+1)=z(i+1)+x(j+1)*y(N+i-j+1);
    end
end
t1=toc;
tic;
x1=fft(x,N);
y1=fft(y,N);
z1=ifft(x1.*y1);
t2=toc;
t=[t1,t2];
end
```

**结果：**

```matlab
t =

    0.0063    0.0045
```

**分析：**

卷积采用代码解释执行速度较慢，Fourier变换采用内部函数速度很快，用FFT计算速度要快得多.

## 实例三：级数乘积

用FFT计算多项式$\sum_{n=0}^{m}\frac{(-1)^{n}x^{2n+1}}{(2n+1)!}$和$\sum_{n=0}^{m}\frac{(-1)^{n}x^{2n}}{(2n)!}$的乘积，并与$\frac{\sin{2x}}{2}$的Taylor级数的相应项比较.

**源代码如下：**

```matlab
function [z,maxerror]=ex1603(m)
%z:乘积，maxerror：最大误差，m：阶数
len=4*m+2;
a=zeros(len,1);
a(2)=1;
for i=4:2:2*m+2
    a(i)=-1*a(i-2)/(i-2)/(i-1);
end
b=zeros(len,1);
b(1)=1;
for i=3:2:2*m+1
    b(i)=-1*b(i-2)/(i-2)/(i-1);
end
c=zeros(len,1);
c(2)=1;
for i=4:2:len
    c(i)=-4*c(i-2)/(i-1)/(i-2);
end
x=fft(a,len);
y=fft(b,len);
z1=x.*y;
z=ifft(z1);
maxerror=0;
for i=1:len
    e=abs(z(i)-c(i));
    if e>maxerror
        maxerror=e;
    end
end
end
```

**计算结果误差分析：**

|  m   |   error    |
| :--: | :--------: |
|  1   |    0.05    |
|  2   |   0.0016   |
|  3   | 2.7557e-05 |
|  4   | 3.0063e-07 |
|  5   | 2.2483e-09 |
|  6   | 1.2236e-11 |

**随着m的增加，误差迅速减少.**