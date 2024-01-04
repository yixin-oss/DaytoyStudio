# de Casteljau算法与de Boor算法

<div style="border: 2px solid black; background-color: white; padding: 8px; border-radius: 10px;">
	<p style="line-height: 2;">
目的: 考虑n次(有理)Bezier曲线/(有理)B样条曲线, 对参数域的某个参数, 希望给出简单、高效、几何特征明显的计算曲线上对应点坐标的方法. 方法的核心在于借助Bernstein基函数/B样条基函数的"降阶"性质.
    </p>
</div>

## (有理)Bezier曲线的de Casteljau 算法

​		对于已经构造好的一条Bezier曲线, 若需要计算曲线上对应某个参数值的点坐标, 除直接代入求值外, 我们还可以借助Bernstein基函数的性质, 采用**几何作图**的方法计算Bezier曲线上的点, 称为de Casteljau算法.

​		由Bernstein基函数的递推性质
$$
B_i^n(t)=(1-t)B_i^{n-1}(t)+tB_{i-1}^{n-1}(t), \quad B_{-1}^{n-1}(t)=B_n^{n-1}(t)=0, i=0,1,...,n.
$$

可得
$$
\begin{aligned}
P(t) & =\sum_{i=0}^nP_iB_i^n(t)\\
& = P_0(1-t)B_0^{n-1}(t)+\sum_{i=1}^{n-1}P_i[(1-t)B_i^{n-1}(t)+P_ntB_{n-1}^{n-1}(t)]\\
& = (1-t)\sum_{i=0}^{n-1}P_iB_i^{n-1}(t)+t\sum_{i=0}^{n-1}P_{i+1}B_i^{n-1}(t)\\
& = \sum_{i=0}^{n-1}((1-t)P_i+tP_{i+1})B_i^{n-1}(t).
\end{aligned}
$$
也就是说, $n$次Bezier曲线从形式上“降阶”为$n-1$次Bezier曲线. 新的控制顶点$P_i^{(1)}$落在原控制多边形的边$P_iP_{i+1}$上, 且将边按比例$t:1-t$进行分割. 若一直这样"降阶",
$$
P(t)=\sum_{i=0}^{n-1}P_i^{(1)}(t)B_i^{n-1}(t)=...=\sum_{i=0}^{n-k}P_i^{(k)}(t)B_i^{(n-k)}(t)=...=P_0^{(n)}(t),
$$
最后$n$次Bezier曲线从形式上"降阶"为0次Bezier曲线(即一点)$P_0^{(n)}(t)$, 这就是要求的曲线上的点$P(t)$. 

​		由此可得Bezier曲线上某一点求值的de Casteljau算法
$$
\begin{cases}
& P_i^{(0)}(t)=P_i^{(0)}=P_i, i=0,1,...,n,\\
& P_i^{(k)}(t)=(1-t)P_i^{(k-1)}(t)+tP_{i+1}^{(k-1)}(t), i=0,1,...,n-k; k=1,2,...,n.
\end{cases}
$$

<div style="border: 2px solid red; background-color: white; padding: 10px; border-radius: 10px;">
	<p style="line-height: 2;">
几何意义: de Casteljau算法就是在每一步的控制多边形的每一条边上, 按照比例t:1-t选择控制顶点而形成新的控制多边形, 每一次递推都将减少控制多边形的一条边, 最后在只剩一条边的控制多边形上按照比例选择的点就是所求点. 对n次Bezier曲线, 进行n级递推即得到所求结果.
    </p>
</div>

控制顶点的递归金字塔关系如下图所示.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/image-20231220185119692.png" alt="image-20231220185119692" style="zoom: 40%;" />

de Casteljau算法的示意图如下, 其中$P_0^{(3)}$即为所要求的点.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/image-20231220185210832.png" alt="image-20231220185210832" style="zoom:50%;" />

设
$$
P^{(0)}=(P_0,P_1,...,P_n)^T, P^{(k)}(t)=(P_0^{(k)},...,P_{n-k}^{(k)})^T,
$$
则de Casteljau算法可表示为矩阵形式
$$
P^{(k)}(t)=M_k(t)\cdots M_2(t)M_1(t)P^{(0)},
$$
其中$(n-k+1)\times (n-k+2)$阶矩阵$M_k(t)$为
$$
M_k(t)=
\begin{pmatrix}
1-t & t & 0 & \cdots & 0 & 0\\
0 & 1-t & t & \cdots & 0 & 0\\
\vdots & \ddots & \ddots & \ddots & \ddots & \vdots\\
0 & 0 & \cdots & 1-t & t & 0\\
0 & 0 & \cdots & 0 & 1-t & t
\end{pmatrix}.
$$

------

​		接下来, 我们考虑Bezier曲线的有理推广形式——有理Bezier曲线. 若要进行合理外推, 基函数的构造应满足一定条件:

​		(1) 尽可能保留Bernstein基函数的性质;

​		(2) 在特殊情况下可以退化为Bernstein基函数.

​		设每一个$n$次Bernstein基函数$B_i^n(t)$对应一个权因子$\omega_i \ge 0$, 令$B_i^n(t)$乘以$\omega_i$, 再用它们的和作为统一的分母进行平均, 从而得到一组多项式
$$
R_i^n(t)=\frac{\omega_iB_i^n(t)}{\sum_{i=0}^n \omega_iB_i^n(t)}, i=0,1,...,n.
$$
称$\{R_i^n(t)\}$为$n$次有理Bernstein基函数.

​		有理Bernstein基函数有如下性质:

​		(1) 非负性: $R_i^n(t)\ge 0, t\in [0,1]$.

​		(2) 单位分解性: $\sum_{i=0}^{n}R_i^n(t)=1$.

​		(3) 端点性质: 在端点$t=0,1$, 分别只有一个有理Bernstein基函数取值为1, 其余为0, 即		
$$
R_i^n(0)=
				\begin{cases}
					& 1, i=0,\\
					& 0, i\neq 0,
				\end{cases}
				R_i^n(1)=
				\begin{cases}
					& 1, i=n,\\
					& 0, i\neq n,
				\end{cases}
$$
​		(4) 退化性质: 当所有权因子都相等, 即对$i=0,1,...,n$, 有$\omega_i=\omega>0$时, 有理Bernstein基函数退化为Bernstein基函数.

​		一条$n$次有理Bezier曲线定义为
$$
R(t)=\sum_{i=0}^{n}P_iR_i^n(t)=\frac{\sum_{i=0}^{n}\omega_iP_iB_i^n(t)}{\sum_{i=0}^{n}\omega_iB_i^n(t)}, t\in [0,1],
$$
其中$R_i^n(t)$为$n$次有理Bernstein基函数, 空间向量$P_i\in \mathbb{R}^3$称为控制顶点, $\omega_i$称为权因子, $i=0,1,...,n$.

​		有理Bezier曲线的de Casteljau算法:
$$
\begin{equation*}
			R(t)=\frac{\sum_{i=0}^{n-1}\omega_i^{(1)}(t)P_i^{(1)}(t)B_i^{n-1}(t)}{\sum_{i=0}^{n-1}\omega_i^{(1)}(t)(t)B_i^{n-1}(t)}=...=\frac{\sum_{i=0}^{n-k}\omega_i^{(k)}(t)P_i^{(k)}(t)B_i^{n-k}(t)}{\sum_{i=0}^{n-k}\omega_i^{(k)}(t)(t)B_i^{n-k}(t)}=...=P_0^{(n)}(t),
		\end{equation*}
$$
点$P_0^{(n)}(t)$即为所求的$R(t)$, 其中
$$
\begin{equation*}
			\begin{cases}
				& \omega_i^{(0)}(t)=\omega_i^{(0)}=\omega_i,i=0,1,...,n,\\
				& \omega_i^{(k)}(t)=(1-t)\omega_i^{(k-1)}(t)+t\omega_{i+1}^{(k-1)}(t), k=1,2,...,n; i=0,1,...,n-k,
			\end{cases}
		\end{equation*}
$$

$$
\begin{equation*}
			\begin{cases}
				& P_i^{(0)}(t)=P_i^{(0)}(t)=P_i, i=0,1,...,n,\\
				& P_i^{(k)}(t)=\frac{(1-t)\omega_i^{(k-1)}(t)P_{i}^{(k-1)}(t)+t\omega_{i+1}^{(k-1)}(t)P_{i+1}^{(k-1)}(t)}{\omega_i^{(k)}(t)}, k=1,2,...,n; i=0,1,...,n-k.
			\end{cases}
		\end{equation*}
$$

### 数值实验

​		设已知控制顶点$P_0=(-1,1), P_1=(1,2), P_2=(3,0), P_3=(4,1)$, 对应的权因子分别为$\omega_0=1, \omega_1=2, \omega_2=2, \omega_3=1$, 利用de Casteljau算法绘制对应的有理Bezier曲线.

​		实验结果如下图所示, 其中控制顶点以绿色星号标注, 控制多边形为蓝色, 有理Bezier曲线标为红色.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/RBC.png" alt="RBC" style="zoom:77%;" />

### Codes

```matlab
function point = DeCasteljau(control_points, weights, t)

n = size(control_points, 2) - 1; 
P = control_points;
w = weights;

for k = 1: n
    for i = 0: n - k
        P(:, i + 1) = (1-t) * w(i + 1) * P(:, i+1) + t * w(i + 2) * P(:, i + 2);
        w(i + 1) = (1 - t) * w(i + 1) + t * w(i + 2);
        P(:,i + 1) = P(:, i + 1) / w(i + 1);
    end
end

point = P(:, 1);

end
```

```matlab
control_points=[-1, 1, 3, 4;
                -1, 2, 0, 1];
weights = [1 2 2 1];

num_points = 100; 
t = linspace(0, 1, num_points);
curve_points = zeros(2, num_points);

for i = 1:num_points
    point = DeCasteljau(control_points, weights, t(i));
    curve_points(:, i) = point;
end

figure;
plot(curve_points(1,:), curve_points(2,:), 'r-');
title('Rational Bezier Curve');
xlabel('x');
ylabel('y');
hold on 
plot(control_points(1,:),control_points(2,:),'g*');
hold on
plot(control_points(1,:),control_points(2,:),'b-');
```

## (有理)B样条曲线的de Boor算法

​		B样条曲线的de Boor算法, 也称为B样条曲线的**几何作图法**, 它是de Casteljau算法的推广.

​		对$p$次B样条曲线
$$
P(t)=\sum_{i=0}^nP_iN_{i,p}(t), \ t\in [t_p,t_{n+1}],
$$
若给出参数值$t\in[t_i,t_{i+1}]\subset [t_p, t_{n+1}]$, 目的是计算曲线上对应该参数的点坐标$P(t)$.

​		利用B样条基函数的局部支集性与de Boor-Cox公式,
$$
\begin{aligned}
P(t) & = \sum_{j=0}^n P_jN_{j,p}(t)\\
& = \sum_{j=i-p}^i P_jN_{j,p}(t)\\
& = \sum_{j=i-p}^i P_j[\frac{t-t_j}{t_{j+p}-t_{j}}N_{j,p-1}(t)+\frac{t_{j+p+1}-t}{t_{j+p+1}-t_j}]N_{j,p-1}(t)\\
& = \sum_{j=i-p+1}^i P_j^{(1)}(t)N_{j,p-1}(t).
\end{aligned}
$$
也就是说, 在区间$[t_i,t_{i+1}]$上, $p$次B样条曲线从形式上"降阶"为$p-1$次B样条曲线. 对任意的$i-p+1\le j\le i$, 新的控制顶点$P_j^{(1)}$落在原控制多边形的边$P_{j-1}P_j$上, 即将原控制多边形局部进行"割角", 从而形成新的控制多边形.

​		一直这样"降阶",
$$
P(t)=\sum_{j=i-p+1}^iP_j^{(1)}(t)N_{j,p-1}(t)=\cdots=\sum_{j=i-p+k}^i P_j^{(k)}(t)N_{j,p-k}(t)=\cdots=P_i^{(p)}(t),
$$
最后$p$次B样条曲线从形式上"降阶"为$0$次, 即一点$P_i^{(p)}(t)$,就是要计算的曲线上的点$P(t)$.

​		由此可得B样条曲线的de Boor算法.
$$
\begin{cases}
& P_j^{(0)}(t) = P_j^{(0)}(t)=P_j,\ j=i-p,i-p+1,...,i,\\
& P_j^{(k)}(t) = (1-\alpha_j^{(k)})P_{j-1}^{(k-1)}(t)+\alpha_j^{(k)}P_j^{(k-1)}(t),\\
& \alpha_j^{(k)}=\frac{t-t_j}{t_{j+p+1-k}-t_j}, k=1,2,...,p; j=i-p+k,i-p+k+1,...,i.
\end{cases}
$$
​		de Boor算法的几何意义与de Casteljau算法的异曲同工, 只需将比例换成与$\alpha$有关, 这里不再赘述. 事实上, de Casteljau算法可视为de Boor算法的特殊形式.

​		控制顶点的递推金字塔关系如图所示.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/image-20231220235156355.png" alt="image-20231220235156355" style="zoom:50%;" />



​		de Boor算法的示意图如下, 其中$P(\frac{1}{2})=P_3^{(3)}(\frac{1}{2})$.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/image-20231220235258389.png" alt="image-20231220235258389" style="zoom: 67%;" />

> 由B样条曲线的重节点性质, 当参数区间上的三次B样条曲线退化为三次Bezier曲线时, de Boor 算法退化为de Casteljau算法.

------

​		接下来, 我们考虑B样条曲线的有理推广形式——有理B样条曲线(NURBS曲线).

​		设$N_{i,p}(t)(i=0,1,...,n)$为定义在节点向量$U=\{t_0,t_1,...,t_{n+p+1}\}$上的$p$次B样条基函数, 称
$$
\begin{equation*}
				R_{i,p}(t)=\frac{\omega_iN_{i,p}(t)}{\sum_{i=0}^n \omega_iN_{i,p}(t)}, i=0,1,...,n
			\end{equation*}
$$
为$p$次有理B样条基函数, 其中$\omega_i\ge 0$称为权因子. 当所有权因子都相等且不为零时, $R_{i,p}(t)$就退化为B样条基函数$N_{i,p}(t)$.

​		$p$次有理B样条曲线定义为
$$
\begin{equation*}
				R(t)=\sum_{i=0}^{n}P_iR_{i,p}(t)=\frac{\sum_{i=0}^n\omega_iP_iN_{i,p}(t)}{\sum_{i=0}^n\omega_iN_{i,p}(t)}, t\in [t_p,t_{n+1}],
			\end{equation*}
$$
其中$R_{i,p}(t)$为定义在节点向量$U=\{t_0,t_1,...,t_{n+p+1}\}$上的$p$次有理B样条基函数, 空间向量$P_i\in \mathbb{R}^3$称为控制顶点, $\omega_i\ge 0, i=0,1,...,n$称为权因子. 当所有权因子都相同且不为零时, 有理B样条曲线退化为B样条曲线.

​		对$t\in[t_i,t_{i+1})\subset [t_p,t_{n+1}], $计算有理B样条曲线上点的de Boor算法为
$$
\begin{equation*}
				\begin{cases}
					\omega_j^{(0)}(t)=\omega_j^{(0)}=\omega_j, j=i-p,i-p+1,...,i,\\
					\omega_j^{(k)}(t)=(1-\alpha_j^{(k)})\omega_{j-1}^{(k-1)}(t)+\alpha_j^{(k)}\omega_j^{(k-1)}(t),\\
					\alpha_j^{(k)}=\frac{t-t_j}{t_{j+p+1-k}-t_j},k=1,2,...,p; j=i-p+k,i-p+k+1,...,i.
				\end{cases}
			\end{equation*}
$$

$$
\begin{equation*}
				\begin{cases}
					P_j^{(0)}(t)=P_j^{(0)}=P_j, j=i-p,i-p+1,...,i,\\
					P_j^{(k)}(t)=\frac{1}{\omega_j^{(k)}(t)}[(1-\alpha_j^{(k)})\omega_{j-1}^{(k-1)}(t)P_{j-1}^{(k-1)}(t)+\alpha_j^{(k)}\omega_j^{(k-1)}(t)P_j^{(k-1)}(t)],\\
					k=1,2,...,p; j=i-p+k,i-p+k+1,...,i.
				\end{cases}
			\end{equation*}
$$

$P_i^{(p)}(t)$即为所求的点$R(t)$.

### 数值实验

​		设已知控制顶点$P_0=(0,3),P_1=(1,5),P_2=(2,4),P_3=(3,7),P_4=(4,9),P_5=(5,2),P_6=(6,1),P_7=(7,4),P_8=(8,8),P_9=(9,9),P_{10}=(10,6)$, 节点向量为$U=\{0,0,0,0,0.2,0.3,0.4,0.6,0.8,0.8,1,1,1,1\}$, 权因子均为$\omega_i=1, i=0,1,...,10$. 利用de Boor算法绘制对应的有理B样条曲线.

​		实验结果如下图所示, 其中控制顶点以绿色星号标注, 控制多边形为蓝色, 有理B样条曲线标为红色. 从图中可以看出, 由于重节点插值性质, 该有理B样条曲线经过控制顶点$P_0, P_7, P_{10}$.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/RBsplineC.png" alt="RBsplineC" style="zoom:75%;" />

### Codes

```matlab
function point = deBoor(control_points, weights, knots, t)

P = control_points;
n = size(P, 2) - 1;
p = length(knots) - n - 2;

ind = find(knots > t, 1, 'first') - 2;
new_points = control_points(:, ind - p + 1: ind + 1);
w = weights(:, ind - p + 1: ind + 1);

for r = 1: p
    j = p + 1;
    for i = ind : -1 : ind - p + r
        frac1 = t - knots(i+1);
        frac2 = knots(i + p + 1 - r + 1) - knots(i + 1);
        if frac2 == 0
            alpha = 0;
        else
            alpha = frac1 / frac2;
        end
        new_points(:, j) = (1 - alpha) * w(j - 1) * new_points(:, j - 1) + alpha * w(j) *new_points(:, j);
        w(j) = (1 - alpha) * w(j - 1) + alpha * w(j);
        new_points(:,j) = new_points(:,j) / w(j);
        j = j - 1;
    end
end

point = new_points(:, end);

end
```

```matlab
control_points = [0,1,2,3,4,5,6,7,8,9,10;...
                  3,5,4,7,9,2,1,4,8,9,6];
weights = [1 1 1 1 1 1 1 1 1 1 1];  

knots = [0,0,0,0,0.2,0.3,0.4,0.6,0.8,0.8,0.8,1,1,1,1];   

num_samples = 100;

t = linspace(knots(1), knots(end), num_samples);
t(end) = t(end) - 0.001;
curve_points = zeros(2, num_samples);

for i = 1:num_samples
    point = deBoor(control_points, weights, knots, t(i));
    curve_points(:,i) = point;
end

plot(curve_points(1,:), curve_points(2,:), 'r-');
title('Rational B-spline Curve');
xlabel('x');
ylabel('y');
hold on
plot(control_points(1,:), control_points(2,:), 'g*');
hold on
plot(control_points(1,:), control_points(2,:), 'b-');
```

## Reference

[1] 王仁宏，李崇君，朱春钢编著. 计算几何教程[M]. 北京：科学出版社, 2008.
[2] 朱春钢，李彩云编. 数值逼近与计算几何[M]. 北京：高等教育出版社, 2020.

[3] [de Boor算法C++](http://www.whudj.cn/?p=535)

[4] [北极星！](https://www.cnblogs.com/zhjblogs/p/16122697.html)