# Bezier 曲线曲面绘制

## Bernstein 基函数

在 Weierstrass 第一定理的构造性证明中，证明的关键是利用 Bernstein 基函数. 下面将介绍 Bernstein 基函数在计算几何中的应用，这类基函数具有“几何直观”的优良性质.

**n 次多项式 Bernstein 基函数**为

$$
B_i^n(t)=(_i^n)t^i(1-t)^{n-i}, i=0,1,...,n
$$

其中$(_i^n)=\frac{n!}{i!(n-i)!},i=0,1,...,n.$

**举例：**$n=3$时，三次 Bernstein 多项式为

$$
B_0^3(t)=(1-t)^3,B_1^3(t)=3t(1-t)^2,B_2^3(t)=3t^2(1-t),B_3^3(t)=t^3.
$$

对应图形如图所示.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1637800870(1).jpg" alt="3次Bernstein基函数图形" style="zoom: 80%;" />

<!--more-->

### 基本性质

- 非负性. $B_i^n(t)\geq 0, t\in[0,1].$
- 单位分解性. $\sum_{i=0}^{n}B_i^n(t)=(t+(1-t))^n=1.$
- 端点性质. 在端点$t=0,t=1$，分别只有一个 Bernstein 基函数取值为 1，其余全部为 0，即

$$
B_i^n(0)=\{_{0,i\neq 0,}^{1, i=0,} B_i^n(1)=\{^{1, i=n,}_{0, i\neq n.}
$$

- 对称性. $B_i^n(t)=B_{n-i}^n(1-t),i=0,1,...,n。$

- 递推公式. 每一个 n 次 Bernstein 基函数可以由两个 n-1 次 Bernstein 基函数递推得到，即

  $$
  B_i^n(t)=(1-t)B_i^{n-1}(t)+tB_{i-1}^{n-1}(t),i=0,...,n.
  $$

- 最大值. $n\geq 1$时，Bernstein 基函数$B_i^n(t)$在$t=\frac{i}{n}$处取得唯一最大值.

- 积分等值性. 所有 n 次 Bernstein 基函数在$[0,1]$上积分值相等.

$$
\int_0^1 B_i^n(t)dt=\frac{1}{n+1}, i=0,1,...,n.
$$

## Bezier 曲线

### 定义

称参数曲线段

$$
P(t)=\sum_{i=0}^{n}P_iB_i^n(t),t\in[0,1],
$$

为一条 n 次**Bezier 曲线**，其中$B_i^n(t)$为 n 次 Bernstein 基函数，空间向量**$P_{i}\in R^3$**称为控制顶点，依次用直线段连接相邻两个控制顶点得到的 n 边折线多边形称为**控制多边形.**

### 实例代码

对于控制顶点

$$
P_0=(0,0),P_1=(1,2),P_2=(2,-1),P_3=(3,1)
$$

平面上的三次 Bezier 曲线方程为

$$
P(t)=\sum_{i=0}^3 P_iB_i^n(t)=(3t, 10t^3-15t^2+6t)
$$

```matlab
x=[0,1,2,3];
y=[0,2,-1,1];
n=length(x)-1;
xx=0;yy=0;
syms t
for k=0:n
    B=nchoosek(n,k)*t^k*(1-t)^(n-k);
    xx=xx+x(k+1)*B;
    yy=yy+y(k+1)*B;
end
xx=collect(xx);
yy=collect(yy);
fprintf('三次Bezier曲线方程为：x(t)=%s,y(t)=%s\n',xx,yy);
t1=linspace(0,1);
xx1=subs(xx,t,t1);
yy1=subs(yy,t,t1);
figure();
plot(x,y,'g*','markersize',10);
line(x,y,'color',[0 0 1])
hold on
plot(xx1,yy1,'r-')
hold off
```

控制顶点用绿色\*标注，控制多边形设置为蓝色，Bezier 曲线为红色，结果如图所示.

![](https://s2.loli.net/2022/06/10/zAi2BE51n6TdQuI.jpg)

## 张量积型 Bernstein 基函数

对 m 次与 n 次一元 Bernstein 基函数

$$
\{B_i^m(u)\}_{i=0}^{m},\{B_j^n(v)\}_{j=0}^n,
$$

张量积型$m×n$次二元 Bernstein 基函数为

$$
B_{i,j}^{m,n}(u,v)=B_{i}^{m}(u)B_{j}^{n}(v),i=0,1,...,m,j=0,1,...,n.
$$

这$(m+1)\times(n+1)$个多项式**线性无关**，从而构成二元多项式空间的一组基.

## Bezier 曲面

### 定义

参数曲面

$$
P(u,v)=\sum_{i=0}^m\sum_{j=0}^n P_{i,j}B_{i,j}^{m,n}(u,v), (u,v)\in[0,1]\times[0,1]
$$

为$m\times n$次 Bezier 曲面，其中$B_{i,j}^{m,n}(u,v)$为张量积型 Bernstein 基函数，空间向量$P_{i,j}\in R^3$称为**控制顶点**，$i=0,1,...,m,j=0,1,...,n.$依次用直线段连接同行同列相邻两个控制顶点得到$m\times n$边折线网格称为控制网格.

### 实例代码

对于给定控制顶点

$$
P_{0,0}=(0,0,1),P_{0,1}=(0,1,2),P_{0,2}=(0,2,1),\\
P_{1,0}=(1,0,2),P_{1,1}=(1,1,2.5),P_{1,2}=(1,2,2),\\
P_{2,0}=(2,0,1),P_{2,1}=(2,1,2),P_{2,2}=(2,2,1).
$$

绘制$2\times 2$次 Bezier 曲面

$$
P(u,v)=\sum_{i=0}^2\sum_{j=0}^2 P_{i,j}B_i^2(u)B_j^2(v), (u,v)\in[0,1]\times[0,1].
$$

```matlab
%控制顶点
Px=[0,0,0;1,1,1;2,2,2];
Py=[0,1,2;0,1,2;0,1,2];
Pz=[1,2,1;2,2.5,2;1,2,1];
figure();
plot3(Px,Py,Pz,'r','linewidth',2);
hold on
plot3(Px',Py',Pz','r','linewidth',2);
hold on
plot3(Px,Py,Pz,'g.','markersize',20,'linewidth',2);
hold on
a=0;b=1;
N=10;M=10;
hx=(b-a)/N;
hy=(b-a)/M;
x=(a:hx:b)';
y=(a:hy:b)';
n=2;m=2;
[x,y]=meshgrid(x,y);
PX=zeros(N+1,M+1);
PY=zeros(N+1,M+1);
PZ=zeros(N+1,M+1);
for i=1:n+1
    for j=1:m+1
        PX=PX+Px(i,j)*nchoosek(n,i-1).*(x.^(i-1)).*((1-x).^(n-i+1)).*nchoosek(n,j-1).*(y.^(j-1)).*((1-y).^(n-j+1));
        PY=PY+Py(i,j)*nchoosek(n,i-1).*(x.^(i-1)).*((1-x).^(n-i+1)).*nchoosek(n,j-1).*(y.^(j-1)).*((1-y).^(n-j+1));
        PZ=PZ+Pz(i,j)*nchoosek(n,i-1).*(x.^(i-1)).*((1-x).^(n-i+1)).*nchoosek(n,j-1).*(y.^(j-1)).*((1-y).^(n-j+1));
    end
end
surf(PX,PY,PZ)
```

控制顶点用绿色点标注，控制网格设置为红色，结果如图所示.

![5-2](https://s2.loli.net/2022/06/10/BUioVt2vemgcAP5.jpg)
