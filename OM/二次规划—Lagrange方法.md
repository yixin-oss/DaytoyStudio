# Lagrange方法

# 基本理论

​		Lagrange方法是求解等式约束二次规划问题的一种方法.

## 等式约束的二次规划问题

​		考虑
$$
\begin{align}
& \min \quad \frac{1}{2}x^THx+c^Tx\\
& s.t. \quad Ax=b,
\end{align}
$$
其中$H$是$n$阶对称矩阵, $A$是$m\times n$阶矩阵, $rank(A)=m, x\in \mathcal{R}^n$, $b\in \mathcal{R}^m$.

## Lagrange乘子法

​		定义Lagrange函数
$$
L(x,\lambda)=\frac{1}{2}x^THx+c^Tx-\lambda^T(Ax-b),
$$
令
$$
\nabla_x L(x,\lambda)=0,\quad \nabla_{\lambda} L(x,\lambda)=0,
$$
可得
$$
\begin{align}
& Hx+c-A^T\lambda=0,\\
& -Ax+b=0.
\end{align}
$$
即
$$
\begin{bmatrix}
H & -A^T\\
-A & 0
\end{bmatrix}
\begin{bmatrix}
x \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
-c\\
-b
\end{bmatrix},
$$
此时的系数矩阵
$$
\begin{bmatrix}
H & -A^T\\
-A & 0
\end{bmatrix}
$$
称为Lagrange矩阵.

​		假设Lagrange矩阵是可逆的, 其中系数矩阵$H$可逆, 引入以下记号
$$
\begin{align}
& Q=H^{-1}-H^{-1}A^T(AH^{-1}A^T)^{-1}AH^{-1},\\
& R=(AH^{-1}A^T)^{-1}AH^{-1},\\
& S=-(AH^{-1}A^T)^{-1}.
\end{align}
$$
<img src="https://gitee.com/yixin-oss/blogImage/raw/master/image-20230627104313445.png" alt="image-20230627104313445" style="zoom: 50%;" />

根据**广义初等变换**, 原二次规划问题的解为
$$
\begin{align}
& \hat{x}=-Qc+R^Tb,\\
& \hat{\lambda}=Rc-Sb.
\end{align}
$$
​		设$x^{(k)}$是原问题的任一可行解, 即满足$Ax^{(k)}=b$. 在此点目标函数的梯度
$$
g_k=\nabla f(x^{(k)})=Hx^{(k)}+c,
$$
则可以给出$\hat{x},\hat{\lambda}$的另一种表达式
$$
\begin{align}
& \hat{x}=x^{(k)}-Qg_k,\\
& \hat{\lambda}=Rg_k.
\end{align}
$$

# Code

代码主要参考了[Dsp Tian](https://www.cnblogs.com/tiandsp/p/12088929.html)的博客.

```matlab
clear all;
close all;
clc;

% min     x1^2+2*x2^2+x3^2+x2^2-2*x1*x2+x3
% s.t.    x1+x2+x3 = 4
%         2*x1-x2+x3 = 2
%{
H=[2 -2 0;
   -2 4 0;
   0 0 2];
c = [0 0 1]';
A=[1 1 1;
   2 -1 1];
b=[4 2]';
%}

%min      2*x1^2+x2^2+x1*x2-x1-x2  
%s.t.     x1+x2 = 1
H=[4 1;
   1 2];
c=[-1 -1]';
A=[1 1];
b=1;

%min    1.5*x1^2-x1*x2+x2^2-x2*x3+0.5*x3^2+x1+x2+x3
%s.t.   x1+2*x2+x3 = 4
%{
H=[3 -1 0;
   -1 2 -1;
   0  -1 1];
c=[1 1 1]';
A=[1 2 1];
b=4;
%}

invH = inv(H);
S = -inv(A*invH*A');
R = -S*A*invH;
Q = invH-invH*A'*R;
x = -Q*c+R'*b;

[x1,x2]=meshgrid(0:0.02:0.7,0:0.02:1.5);
z1 = 2*x1.^2+x2.^2+x1.*x2-x1-x2;
mesh(x1,x2,z1);

x1 = 0:0.02:0.7;
x2 = -x1 + 1;

hold on;
plot3(x1,x2,zeros(1,length(x1)),'r');
plot3(x(1),x(2),0,'r*')
plot3(x(1),x(2),2*x(1).^2+x(2).^2+x(1).*x(2)-x(1)-x(2),'b*')
legend('待求问题函数','约束条件','最优解','最小值')
```

![](https://gitee.com/yixin-oss/blogImage/raw/master/QuadLagrange.png)

# Reference_bib

```latex
@book{陈宝林2005最优化理论与算法,
  title={最优化理论与算法},
  author={陈宝林},
  publisher={最优化理论与算法},
  year={2005},
}
```