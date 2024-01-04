---
title: Dupin指标线
---

# 基本介绍

​		通过曲面上一点$P$可以作无数条法截线, 下面要探究法截线的法曲率之间的关系.

​		取点$P$为原点, 曲面$S$的坐标曲线在$P$点的切向量$\boldsymbol{r}_u, \boldsymbol{r}_v$为基向量, 它们构成曲面$S$在$P$点切平面上的坐标系. 给定曲面$S$在$P$点的切方向$\rm d u:\rm d v$, $k_n$是对应于该方向的法曲率, $|\frac{1}{k_n}|$是法曲率半径的绝对值. 过点$P$沿方向$\rm d u:\rm d v(d\boldsymbol{r}=\boldsymbol{r}_u\rm du+\boldsymbol{r}\rm d v)$画一线段$PN$, s.t. $|PN|=\sqrt{\frac{1}{|k_n|}}$, 对于切平面上所有的方向, $N$点的轨迹称为曲面在$P$点的**Dupin指标线**. 示意图如下.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/Dupin%E6%8C%87%E6%A0%87%E7%BA%BF.png" alt="Dupin指标线" style="zoom: 20%;" />

​		下面推导Dupin指标线在上述坐标系下的方程. 设$N$点坐标为$(x, y)$, 则有
$$
x\boldsymbol{r}_u+y\boldsymbol{r}_v=\sqrt{\frac{1}{|k_n|}}\frac{\rm d \boldsymbol{r}}{|\rm d \boldsymbol{r}|}=\frac{\boldsymbol{r}_u\rm du+\boldsymbol{r}\rm d v}{\sqrt{k_n}|\boldsymbol{r}_u \rm du+\boldsymbol{r}\rm d v|}.
$$
两端同时平方并注意到$k_n=\frac{II}{I}$, 可得
$$
Ex^2+2Fxy+Gy^2=\frac{E\rm du^2+2F\rm d u \rm dv+G\rm dv^2}{|L\rm du^2+2M\rm d u \rm dv+N\rm dv^2|}.
$$
又由$\rm du:\rm dv=x:y$,
$$
Ex^2+2Fxy+Gy^2=\frac{Ex^2+2Fxy+Gy^2}{|Lx^2+2Mxy+Ny^2|}.
$$
因此
$$
|Lx^2+2Mxy+Ny^2|=1 \Rightarrow Lx^2+2Mxy+Ny^2=\pm 1.
$$
上述方程不含$x,y$一次项, 表示以$P$为中心的二次曲线.

## Dupin指标线分类

|    条件    |  类型  |     形状     |
| :--------: | :----: | :----------: |
| $LN-M^2>0$ | 椭圆点 |     椭圆     |
| $LN-M^2<0$ | 双曲点 |  共轭双曲线  |
| $LN-M^2=0$ | 抛物点 | 一对平行直线 |
| $L=M=N=0$  |  平点  |    不存在    |

# Reference

```latex
梅向明，黄敬之编. 微分几何 第5版[M]. 北京：高等教育出版社, 2019.
```

