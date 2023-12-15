---
title: 拉普拉斯特征映射(Laplacian Eigenmaps)
---



# 介绍

拉普拉斯特征映射(Laplacian Eigenmaps) 是基于图的非线性数据降维算法, 其基本思想是从高维空间向低维空间映射过程中保持数据间原有的局部结构特性(相似性). 举例来说, 若原数据在高维空间中是邻近的, 则对应在低维空间中的表示也是邻近的. 拉普拉斯特征映射主要通过构建具有邻接矩阵(相似矩阵)的图来保证数据的局部结构特征, 下面是具体的算法流程.

# 算法流程

**输入**：$k$个点$x_1,...,x_k\in \mathcal{R}^l$.

**输出**：低维空间中代表$x_i$的$y_i$, i.e. $y_1,...,y_k\in\mathcal{R}^m(m<<l)$.

**步骤1**  构建邻近图. 如果节点$x_i$和$x_j$是邻近点, 则在它们之间形成一条边作为连接关系. 构建的原则通常有以下三种:

(1) $\epsilon$-邻近法. 预先设定一个阈值, 若节点$x_i,x_j$在$\mathcal{R}^l$空间中的欧式距离满足$\|x_i-x_j\|^2<\epsilon$, 则用边进行连接. 该方法基于几何距离信息, 建立的连接关系是自然对称的, 但阈值的选取往往是困难的, 容易出现连接关系混乱.

(2) $n$邻近法. 取每个点附近的$n$个点作为邻近点建立连接关系. 该方法便于实现, 所建立的连接关系是对称的, 但是缺乏几何直观性.

(3) 全连接法. 直接建立所有点之间的连接关系, 该方法最为直接, 但会对后续的计算过程造成不便.

**步骤2** 选择权重. 在具有连接关系的两点$x_i,x_j$之间的边上赋予权重$W_{ij}$, 得到图的邻接矩阵$W$. 通常有两种方式：

(1) 热核函数. 
$$
W_{ij}=
\begin{cases}
& e^{-\frac{\|x_i-x_j\|^2}{t}}(t\in R),connected,\\
& 0, else.
\end{cases}
$$
(2) 简单形式.
$$
W_{ij}=
\begin{cases}
& 1, connected,\\
& 0, else.
\end{cases}
$$
**步骤3** 特征映射. 求解如下广义特征值问题
$$
L\boldsymbol{f}=\lambda D\boldsymbol{f},
$$
其中$D$是对角权重矩阵, 它的对角线元素是全邻接矩阵$W$元素按行加和, i.e. $D_{ii}=\sum_j W_{ij}, L=D-W$是Laplacian矩阵, 具有对称半正定的性质. 

假设得到的特征值及对应的特征向量分别为$\lambda_0,...,\lambda_{k-1}; \boldsymbol{f}_0,...,\boldsymbol{f}_{k-1}$, 即
$$
\begin{aligned}\large
& L\boldsymbol{f}_0=\lambda_0D\boldsymbol{f}_0\\
\\
& L\boldsymbol{f}_1=\lambda_1D\boldsymbol{f}_1\\
\\
& \qquad...\\
\\
& L\boldsymbol{f}_{k-1}=\lambda_{k-1}D\boldsymbol{f}_{k-1}\\
\\
& 0=\lambda_0\le \lambda_1\le...\le \lambda_{k-1}
\\
\end{aligned}
$$
将非零特征值对应的特征向量$\boldsymbol{f}_1,...,\boldsymbol{f}_{m}$按列排列, 形成矩阵$U$, 将$U$的每一行记为$y_i,i=1,...,k$, 则$y_i$就是$x_i$从高维空间映射到低维空间的表示.
$$
x_i\rightarrow (\boldsymbol{f}_1(i),...,\boldsymbol{f}_{m}(i)).
$$

# 公式推导

Laplacian Eigenmaps 的目标是**原空间中相近的点在映射到新的低维空间中时仍然比较相近**.

1. 给出如下优化目标函数

$$
\min \sum_{i,j}\|y_i-y_j\|^2 W_{ij},
$$

其中$W_{ij}$为邻接矩阵$W$的元素, 距离较远的两点之间的边权重较小, 而距离较近的两点间边的权重较大.

2. 目标函数优化

$$
\begin{aligned}
& \sum_{i=1}^n \sum_{j=1}^n\left\|y_i-y_j\right\|^2 W_{i j} \\
& =\sum_{i=1}^n \sum_{j=1}^n\left(y_i^T y_i-2 y_i^T y_j+y_j^T y_j\right) W_{i j} \\
& =\sum_{i=1}^n\left(\sum_{j=1}^n W_{i j}\right) y_i^T y_i+\sum_{j=1}^n\left(\sum_{i=1}^n W_{i j}\right) y_j^T y_j-2 \sum_{i=1}^n \sum_{j=1}^n y_i^T y_j W_{i j} \\
& =2 \sum_{i=1}^n D_{i i} y_i^T y_i-2 \sum_{i=1}^n \sum_{j=1}^n y_i^T y_j W_{i j} \\
& =2 \operatorname{tr}\left(Y^T D Y\right)-2 \operatorname{tr}\left(Y^T W Y\right) \\
& =2 \operatorname{tr}\left[Y^T(D-W) Y\right] \\
& =2 \operatorname{tr}\left(Y^T L Y\right).
\end{aligned}
$$



变换后的优化目标如下
$$
\min trace(Y^TLY), \quad s.t.\quad Y^TDY=I
$$
其中约束条件$Y^TDY=I$是为了保证优化问题有解.

3. Lagrange乘子法进行求解

$$
\begin{aligned}
& \mathrm{f}(\mathrm{Y})=\operatorname{tr}\left(\mathrm{Y}^{\mathrm{T}} \mathrm{LY}\right)+\operatorname{tr}\left[\lambda\left(\mathrm{Y}^{\mathrm{T}} \mathrm{DY}-\mathrm{I}\right)\right], \\
& \frac{\partial \mathrm{f}(\mathrm{Y})}{\partial \mathrm{Y}}=\mathrm{LY}+\mathrm{L}^{\mathrm{T}} \mathrm{Y}+\mathrm{D}^{\mathrm{T}} \mathrm{Y} \lambda^{\mathrm{T}}+\mathrm{DY} \lambda=2 \mathrm{LY}+2 \mathrm{DY} \lambda=0, \\
& \mathrm{LY}=-\mathrm{DY} \lambda.
\end{aligned}
$$

显然最终转化为求解广义特征值问题. 最后为了目标函数最小化, 选择**最小的$m$个非零特征值对应的特征向量**作为降维后的结果输出.

# 简单举例

给定一个无向图

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1800705-20200711150853303-1200010572.png" alt="img" style="zoom:67%;" />

用简单形式表示它的邻接矩阵
$$
W=
\left(\begin{array}{llllll}
0 & 1 & 0 & 0 & 1 & 0 \\
1 & 0 & 1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 & 1 & 1 \\
1 & 1 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0
\end{array}\right),
$$
$W$显然是对称矩阵, 矩阵中$1$表示有连接, $0$表示无连接.

将$W$按行相加并置于对角线上, 得到权重矩阵$D$
$$
D=
\left(\begin{array}{llllll}
2 & 0 & 0 & 0 & 0 & 0 \\
0 & 3 & 0 & 0 & 0 & 0 \\
0 & 0 & 2 & 0 & 0 & 0 \\
0 & 0 & 0 & 3 & 0 & 0 \\
0 & 0 & 0 & 0 & 3 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{array}\right),
$$
则可得到Laplacian矩阵$L=D-W$
$$
L=
\left(\begin{array}{rrrrrr}
2 & -1 & 0 & 0 & -1 & 0 \\
-1 & 3 & -1 & 0 & -1 & 0 \\
0 & -1 & 2 & -1 & 0 & 0 \\
0 & 0 & -1 & 3 & -1 & -1 \\
-1 & -1 & 0 & -1 & 3 & 0 \\
0 & 0 & 0 & -1 & 0 & 1
\end{array}\right).
$$

# Demo

以二维数据点为例, 使用热核来计算全连接图的矩阵并生成Laplacian矩阵, 将数据点降维到一维坐标轴直线上.

```matlab
% 生成全连接图的拉普拉斯矩阵
n = 100; % 节点数量

% 生成节点之间的欧氏距离矩阵
X = rand(n, 2); % 假设节点的坐标为二维随机数据点
dist_matrix = pdist2(X, X, 'euclidean');

% 设置热核参数
sigma = 0.1;

% 计算权重矩阵
W = exp(-dist_matrix.^2 / (2 * sigma^2)); % 使用热核计算权重，这里采用高斯核函数

D = diag(sum(W)); % 度矩阵，对角线为每个节点的度
L = D - W; % 拉普拉斯矩阵

% 计算拉普拉斯特征变换
[V, lambda] = eig(L); % 计算拉普拉斯特征向量和特征值
lambda=diag(lambda);
[lambda, ind] = sort(lambda, 'ascend');%'ascend' 升序排列(默认)。ind表示最小的位置
Y=V(:,ind(2)); % Y即为二维数据点对应在一维空间中的表示
```

# Reference

[流形学习之拉普拉斯特征映射](https://blog.csdn.net/zailushag/article/details/113362439)

[降维和聚类系列: 拉普拉斯特征映射](https://www.cnblogs.com/picassooo/p/13282900.html)

[拉普拉斯特征映射](https://zhuanlan.zhihu.com/p/358529704)

[流形学习: 拉普拉斯映射](https://blog.csdn.net/weixin_45591044/article/details/122403243)

