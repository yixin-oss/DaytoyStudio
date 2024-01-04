---
title: 隐式QR迭代算法
---

# 问题描述

​		特征值在矩阵分析中有广泛的应用, 它可以用于矩阵的对角化, 相似性判断, 矩阵的谱分解等, 因此, 矩阵特征值的计算方法对于许多问题而言是至关重要的. 事实上, 求一个矩阵的特征值问题是一求一个特征多项式的根的问题, 而五阶以上的多项式的根一般不能通过有限次运算求得. 因此, 矩阵特征值的计算方法本质上都是迭代的. 

​		由于对称矩阵的特征值问题具有更良好的性质, 为了充分利用其对称性, 我们主要列出求解对称特征值问题的隐式对称QR方法, 编写相应的MATLAB代码, 并通过数值实验将计算所得特征值结果与由MATLAB内置函数`eig(A)`所得特征值进行比较, 意在说明隐式对称QR方法计算特征值是精确有效的.

# 初等正交变换

## Householder变换

设单位向量$w\in \boldsymbol{R}^n$, Householder变换定义为
$$
\begin{equation*}
			H=I-2ww^T,
		\end{equation*}
$$
其中$ww^T=1$. Householder变换可以实现向量约化: 对于非零向量$x\in \boldsymbol{R}^n$, 可以构造单位向量$w\in \boldsymbol{R}^n$, 使得Householder变换$H$满足
$$
	\begin{equation*}
			Hx=\alpha e_1, e_1=(1,0,0,...,0)^T, \alpha=\|x\|_2.
		\end{equation*}
$$
也就是说, 利用Householder变换可以在向量中引入零元素, 当然也不只局限于上述形式, 它可以将向量中任何若干相邻元素化为零. 此外, Householder变换在实际构造过程中, 往往还需要考虑到计算机系统溢出的问题而做一些处理, 这里直接用伪代码和相应的MATLAB代码来展示处理细节.

![image-20231229183554631](https://gitee.com/yixin-oss/blogImage/raw/master/image-20231229183554631.png)

```matlab
function [v,beta] =house(x)
%Householder transformation
n=length(x);
v=zeros(n,1);
a=norm(x,Inf);
x=x/a;
q=x(2:n);
s=q'*q;
v(2:n)=x(2:n);
if s==0
    beta=0;
    v(1)=0;
else
    alpha=sqrt(x(1)^2+s);
    if x(1)<=0
        v(1)=x(1)-alpha;
    else
        v(1)=-s/(x(1)+alpha);
    end
    beta=2*v(1)^2/(s+v(1)^2);
    v=v/v(1);
end
end
```

## Givens 变换

Given变换可以将向量其中一个分量化为零, 它具有如下形式:
$$
\begin{equation*}
			G(i,k,\theta)=I+s(e_ie_k^T-e_ke_i^T)+(c-1)(e_ie_i^T+e_ke_k^T)=
			\begin{bmatrix}
				1 & & \vdots & & \vdots & & \\
				 & \ddots & \vdots & & \vdots & & \\
				\cdots & \cdots & c & \cdots & s & \cdots & \cdots \\
				& & \vdots & &  \vdots & & \\
				\cdots & \cdots & -s & \cdots & c & \cdots & \cdots \\
				& & \vdots & & \vdots & \ddots & \\
				& & \vdots & & \vdots & & 1 
			\end{bmatrix}
		\end{equation*}
$$
其中$c=\cos\theta, s=\sin\theta, G(i,k,\theta)$是一个正交阵. 从几何上讲, $G(i,k,\theta)x$是在$(i,k)$坐标平面内将$x$按顺时针方向做了$\theta$度旋转, 所以Givens变换又称为平面旋转变换. 在具体计算过程中, Givens变换也要避免溢出的情况发生, 下面分别给出其伪代码和相应的MATLAB代码.

![image-20231229183727727](https://gitee.com/yixin-oss/blogImage/raw/master/image-20231229183727727.png)

```matlab
function [c,s]=givens(a,b)
    % Givens transformation
    if b==0
        c=1;s=0;
    else
        if abs(b)>=abs(a)
            r=a/b;
            s=1/sqrt(1+r^2);
            c=s*r;
        else
            r=b/a;
            c=1/sqrt(1+r^2);
            s=c*r;
        end
    end
end
```

# 隐式对称QR方法

## 三对角化

若$A$是$n$阶实对称矩阵, 其上Hessenberg分解为
$$
\begin{equation*}
				Q^TAQ=T,
			\end{equation*}
$$
其中$Q$是正交矩阵, 上Hessenberg矩阵$T$一定是对称三对角矩阵. 也就是说, 实对称矩阵的上Hessenberg化就是将其对角化.
将矩阵$A$进行分块
$$
\begin{equation*}
			A=
			\begin{bmatrix}
				a_1 & v_0^T\\
				v_0 & A_0
			\end{bmatrix},
		\end{equation*}
$$
利用Householder变换将其约化为对称三对角矩阵的第$k$步为:

(1) 计算Household变换$\tilde{H}_k$, 使得$\tilde{H}_kv_{k-1}=b_k e_1, b_k\in \boldsymbol{R}$;

(2) 计算
$$
\begin{equation*}
			\tilde{H}_kA_{k-1}\tilde{H}_k=
			\begin{bmatrix}
				a_{k+1} & v_k^T\\
				v_k & A_k
			\end{bmatrix}.
		\end{equation*}
$$
定义
$$
\begin{equation*}
			T=
			\begin{bmatrix}
				a_1 & b_1 & & \\
				b_1 & a_2 & \ddots & \\
				& \ddots & a_{n-1} & b_{n-1}\\
				& & b_{n-1} & a_n
			\end{bmatrix},\\
		\end{equation*}
$$

$$
\begin{equation*}
			Q=H_1 H_2 \cdots H_{n_2}, H_k=diag(I_k,\tilde{H}_k),
		\end{equation*}
$$

则有
$$
\begin{equation*}
			Q^TAQ=T \Rightarrow A=Q^TAQ
		\end{equation*}
$$
称为矩阵$A$的三对角分解.

在上述约化过程中, 第$k$步的约化主要任务是计算$\tilde{H}_kA_{k-1}\tilde{H}_k$. 设
$$
\begin{equation*}
			\tilde{H}_k=I-\beta vv^T, v\in \boldsymbol{R}^{n-k},
		\end{equation*}
$$
利用$A_{k-1}$的对称性可得
$$
\begin{equation*}
			\tilde{H}_kA_{k-1}\tilde{H}_k=A_{k-1}-vw^T-wv^T,
		\end{equation*}
$$
其中
$$
\begin{equation*}
			w=u-\frac{1}{2}\beta(v^Tu)v, u=\beta A_{k-1}v.
		\end{equation*}
$$
依据上述过程可以依次计算出矩阵三对角的元素, 再将三对角线之外的元素一并置为0，即可得到三对角矩阵. 算法伪代码如下:

![image-20231229184237406](https://gitee.com/yixin-oss/blogImage/raw/master/image-20231229184237406.png)

```matlab
function T = Tridecomposition(A)
% Tridiagonal decomposition
T=A;
n=size(T,1);
for k=1:n-2
    [v,beta]=house(T(k+1:n,k));
    u=beta*T(k+1:n,k+1:n)*v;
    w=u-(1/2)*(beta*v'*u)*v;
    T(k+1,k)=norm(T(k+1:n,k));
    T(k,k+1)=T(k+1,k);
    T(k+2:n, k) = 0; % Set elements below the diagonal to zero
    T(k, k+2:n) = 0; % Set elements above the diagonal to zero
    T(k+1:n,k+1:n)=T(k+1:n,k+1:n)-v*w'-w*v';
end
end
```

## 带Wilkinson 位移的隐式对称QR迭代

按上述算法将对称矩阵约化成三对角阵后, 就可以选取适当的位移进行QR迭代. 考虑带原点位移的QR迭代格式
$$
\begin{equation*}
			\begin{aligned}
				& T_k-\mu_k I =Q_kR_k,\\
				& T_{k+1}=R_kQ_k+\mu_k I, k=0,1,...,
			\end{aligned}
		\end{equation*}
$$
其中$T_0=T$是对称三对角阵. 由于QR迭代保持上Hessenberg形和对称性的特点, 因此由上述迭代格式产生的$T_k$都是对称三对角矩阵. 对于位移$\mu_k$的选取, 最著名的就是Wilkinson位移, 将$\mu_k$取为矩阵
$$
\begin{equation*}
			T_k(n-1:n,n-1:n)=
			\begin{bmatrix}
				a_{n-1} & b_{n-1}\\
				b_{n-1} & a_n
			\end{bmatrix}
		\end{equation*}
$$
的特征值中靠近$a_n$的一个, 即
$$
\begin{equation*}
			\mu_k=a_n+\delta-sgn(\delta)\sqrt{\delta^2+b_{n-1}^2}, \delta=\frac{a_{n-1}-a_{n}}{2}.
		\end{equation*}
$$
接下来考虑对称QR迭代的具体实现
$$
\begin{equation*}
			T-\mu I=QR, \tilde{T}=RQ+\mu I.
		\end{equation*}
$$
我们可以利用Given变换直接实现$T-\mu I$的QR分解, 或者采用MATLAB内置的QR分解函数完成每一步迭代. 但是, 更好的做法是以隐含的方式实现由$T$到$\tilde{T}$的变换.

事实上, 上述迭代的实质是用正交相似变换将$T$变为$\tilde{T}$, i.e.$\tilde{T}=Q^TTQ$. 给出Givens变换$G_1=G(1,2,\theta)$, 令
$$
\begin{equation*}
			B=G_1TG_1^T,
		\end{equation*}
$$
则$B$仅比对称三对角阵多出若干非零元. 因此, 只需将$B$用Givens变换约化为三对角阵, 即可得到所需的三对角阵$\tilde{T}$. 以$4\times 4$的矩阵为例, 矩阵$B$依次经过Givens变换$G_i=G(i,i+1,\theta_i),i=2,3$即可. 对于一般情形, 给出如下伪代码.

![image-20231229184531827](https://gitee.com/yixin-oss/blogImage/raw/master/image-20231229184531827.png)

```matlab
function [T1] = wilkinson(T)
% Implicit symmetric QR iteration with Wilkinson displacement
T1=T;
n=size(T1,1);
d=(T1(n-1,n-1)-T1(n,n))/2;
% Wilkinson displacement
u=T1(n,n)-T1(n,n-1)^2/(d+sign(d)*sqrt(d^2+T1(n,n-1)^2));
x=T1(1,1)-u;z=T1(2,1);
for k=1:n-1
    [c,s]=givens(x,z);
    % Givens matrix
    G=eye(n);
    G(k,k)=c;G(k+1,k+1)=c;
    G(k,k+1)=s;G(k+1,k)=-s;
    T1=G*T1*G';
    if k<n-1
        x=T1(k+1,k);z=T1(k+2,k);
    end
end
end
```

## 隐式对称QR算法

根据前面两部分内容的介绍, 我们将隐式对称QR算法的流程总结如下:

![image-20231229184829850](https://gitee.com/yixin-oss/blogImage/raw/master/image-20231229184829850.png)

在实际编程中, 我们还增加了最大迭代步数的要求. 对应的MATLAB主程序如下

```matlab
function lambda = main(A,maxit,eps)
% Implicit symmetric QR algorithm
% A:Real symmetric matrix
% maxit:Maximum iterations
% eps:convergence accuracy

% Tridiagonal decomposition 
T=Tridecomposition(A);
lambda0=diag(T);
T=wilkinson(T);
lambda=diag(T);
m=1;
while max(abs(lambda-lambda0))>eps && m<=maxit
    lambda0=lambda;
    T=wilkinson(T);
    lambda=diag(T);
    m=m+1;
end
lambda=sort(lambda);
if m>=maxit
    fprintf('The number of iterations exceeds %d, the convergence is too slow.\n',maxit);
else
    fprintf('Implicit symmetric QR algorithm converges in %d iterations.\n',m);
end
end
```

# 数值实验

1. 考虑如下实对称矩阵$A$, 最大迭代步数设为$1000$, 收敛精度阈值设为$10^{-16}$.

$$
\begin{equation*}
			A=
			\begin{bmatrix}
				1 & 1 & 0.5\\
				1 & 1 &0.25\\
				0.5& 0.25& 2
			\end{bmatrix}.
		\end{equation*}
$$

将上述算法所求得的特征值与由MATLAB内置函数`eig(A)`所得特征值进行比较, 具体结果如下：

```matlab
A=[1,1,0.5;1,1,0.25;0.5,0.25,2];
lambda=main(A,1000,1e-16)

Implicit symmetric QR algorithm converges in 51 iterations.

lambda =

  -0.016647283606310
   1.480121423189129
   2.536525860417189
   
eig(A)

ans =

  -0.016647283606310
   1.480121423189129
   2.536525860417181
```

2. 考虑如下实对称矩阵$B$, 最大迭代步数设为$1000$, 收敛精度阈值设为$10^{-16}$.

$$
\begin{equation*}
			B=
			\begin{bmatrix}
				2 & 1 & 0\\
				1 & 3 & 1\\
				0 & 1 & 4
			\end{bmatrix}.
		\end{equation*}
$$

```matlab
B=[2,1,0;1,3,1;0,1,4];
lambda=main(B,1000,1e-16)
Implicit symmetric QR algorithm converges in 27 iterations.

lambda =

   1.267949192431124
   3.000000000000004
   4.732050807568879

eig(B)

ans =

   1.267949192431123
   3.000000000000001
   4.732050807568877
```

此时由隐式对称QR算法计算得到的特征值与由MATLAB内置函数`eig(A)`所得特征值的相对误差为$6.421519\times 10^{-16}$, 可认为两者是完全一致的.

3. 考虑一个随机生成的$6\times 6$的实对称矩阵$C$, 最大迭代步数同样设为1000, 收敛精度阈值设为$10^{-16}$.

```matlab
n=6;A = randn(n); C = A + A';
lambda=main(C,1000,1e-16);

Implicit symmetric QR algorithm converges in 788 iterations.

lambda =

  -4.824833019831860
  -4.019905367956682
  -0.027151638120863
   0.334363389176644
   2.510099632698148
   5.766676337930520
   
eig(C)

ans =

  -4.824833019831782
  -4.019905367956486
  -0.027151638120862
   0.334363389176626
   2.510099632698045
   5.766676337930452
```

这里我们计算由隐式对称QR算法计算得到的特征值与由MATLAB内置函数`eig(A)`所得特征值的相对误差为$2.746606\times 10^{-14}$.

从上述数值实验可以看出, 由隐式对称QR算法计算得到的特征值与由MATLAB内置函数`eig(A)`所得特征值在符合机器精度要求下可认为是完全一致的. 隐式对称QR算法计算得到的特征值是相当精确的.