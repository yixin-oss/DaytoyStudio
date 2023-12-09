---
title: Latex随手记
tags: Latex
categories: Latex写作
---

## 图表编号与章节号关联

```latex
\usepackage{amsmath}%宏包
\numberwithin{figure}{section}
\numberwithin{table}{section}
```

## 插图实现中英文双标题

```latex
\usepackage{ccaption}%宏包
```

```latex
\begin{figure}
    \centering
    \includegraphics{image}
    \bicaption{图}{中文标题}{Figure}{English title} 
\end{figure}
```

<!--more-->

### 实例

```latex
\begin{figure}[H]
  \centering
  \includegraphics[scale=0.7]{N5.eps}\\
  \bicaption{图}{N=5时, 数值解与精确解的图像}{Fig.}{Figures of numerical and exact solutions at N=5}

\end{figure}
```

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211129184426450.png)

## 表格上方设置中英文双标题

```latex
\begin{table}[H]
\centering
\bicaption{表}{不同 N 值对应数值解与精确解绝对值最大误差}{Tab.}{Different N values correspond to the maximum absolute error of numerical solution and exact solution}
\setlength{\tabcolsep}{15mm}{
\begin{tabular}{|c|c|c|}% 通过添加 | 来表示是否需要绘制竖线

\hline  % 在表格最上方绘制横线
N & h & 绝对值最大误差 \\
\hline  %在第一行和第二行之间绘制横线
4 & 1/4 & 0.002334\\
\hline
8 & 1/8 & 0.00058317\\
\hline
16 & 1/16 & 0.00014645\\
\hline
32 & 1/32 & 0.000036675\\
\hline
64 & 1/64 & 0.0000091700\\
\hline % 在表格最下方绘制横线
\end{tabular}}
\end{table}
```

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211129185001285.png)

## Latex符号

长竖线

```latex
\bigg|
```

字母上方加符号

```
\hat{}%加^号
\widehat{}
\overline{} %加横线
\widetilde{} %加波浪线
\dot{}%加一个点
\ddot{} %加两个点
```



[符号大全](https://blog.csdn.net/wangmeitingaa/article/details/88825621)

[数学符号表示方法](https://zhuanlan.zhihu.com/p/67251812)

[手写符号识别](http://detexify.kirelabs.org/classify.html)

## 矩阵编写

圆括号矩阵

```latex
\begin{pmatrix}
a & b \\
c & d 
\end{pmatrix}
```

$$
\begin{pmatrix}
a & b \\
c & d 
\end{pmatrix}
$$

方括号矩阵

```latex
\begin{bmatrix}
a & b \\
c & d 
\end{bmatrix}
```

$$
\begin{bmatrix}
a & b \\
c & d 
\end{bmatrix}
$$

大括号矩阵

```latex
\begin{Bmatrix}
a & b \\
c & d 
\end{Bmatrix}
```

$$
\begin{Bmatrix}
a & b \\
c & d 
\end{Bmatrix}
$$

行列式

```latex
\begin{vmatrix}
a & b \\
c & d 
\end{vmatrix}
```

$$
\begin{vmatrix}
a & b \\
c & d 
\end{vmatrix}
$$

范数矩阵

```latex
\begin{Vmatrix}
a & b \\
c & d 
\end{Vmatrix}
```

$$
\begin{Vmatrix}
a & b \\
c & d 
\end{Vmatrix}
$$

排出一个$n\times n$矩阵

```latex
\begin{pmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{pmatrix}
```

$$
\begin{pmatrix}
a_{11} & \cdots & a_{ln} \\
\vdots & \ddots & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{pmatrix}
$$

![图示](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211201105205227.png)

某一位置如果没有，直接空置.

**e.g.**上三角矩阵

```latex
\begin{pmatrix}
a_{11} & \cdots & a_{1n} \\
 & \ddots & \vdots \\
& \ & a_{nn}
\end{pmatrix}
```

$$
\begin{pmatrix}
a_{11} & \cdots & a_{1n} \\
 & \ddots & \vdots \\
&  & a_{nn}
\end{pmatrix}
$$

## 未完待续... ...