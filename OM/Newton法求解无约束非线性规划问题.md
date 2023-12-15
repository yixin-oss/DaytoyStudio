## 理论基础

考虑目标函数$f$在点$x^k$处的二次逼近式
$$
f(x)\approx Q(x)=f(x^k)+\nabla f(x^k)^T(x-x^k)+\frac{1}{2}(x-x^k)^T\nabla^2f(x^k)(x-x^k)
$$
假设$Hessen$阵$\nabla^2f(x^k)$正定，函数$Q$的稳定点$x^{k+1}$是$Q(x)$的最小点，则可令
$$
\nabla Q(x^{k+1})=\nabla f(x^k)+\nabla^2f(x^k)(x^{k+1}-x^{k})=0
$$
解得
$$
x^{k+1}=x^k-[\nabla^2f(x^k)]^{-1}\nabla f(x^k)
$$
可知从点$x^k$出发，沿搜索方向$p^k=-[\nabla^2f(x^k)]^{-1}\nabla f(x^k)$并取步长$t_k=1$即可得$Q(x)$最小点.

上述方向$p^k$称为从点$x^k$出发的**Newton方向**.

<!--more-->

## 算法步骤

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211206150752688.png" alt="算法流程"  />

- 选取初始点$x^0$,给定终止误差$\varepsilon>0, k=0$;
- 计算$\nabla f(x^k)$,若$||\nabla f(x^k)||\leq \varepsilon$, 停止迭代，输出$x^k$， 否则进行第三步;
- 取$p^k=-[\nabla^2f(x^k)]^{-1}\nabla f(x^k)$;
- $x^{k+1}=x^k+p^k, k=k+1$, 转第二步.

## 求解实例

用Newton法求解
$$
min \quad f(x)=x_1^4+25x_2^4+x_1^2x_2^2
$$
选取$x^0=(2,2)^T,\varepsilon=10^{-6}$.

```matlab
function x=Optimize_Newton(x0,eps)

[~,g1,g2]=nwfun(x0);
while norm(g1)>eps
    p=-inv(g2)*g1';
    x0=x0+p;
    [~,g1,g2]=nwfun(x0);
end
x=x0;
end

function [f,df,df2]=nwfun(x)
x1=x(1);
x2=x(2);
f=x1^4+25*x2^4+x1^2*x2^2;
df(1)=4*x1^3+2*x1*x2^2;
df(2)=100*x2^3+2*x1^2*x2;
df2(1,1)=12*x1^2+x2^2;
df2(1,2)=4*x1*x2;
df2(2,1)=df2(1,2);
df2(2,2)=300*x2^2+2*x1^2;
end
```

```latex
x=Optimize_Newton([2;2],1e-6)

x =

   0.000148337033956
   0.002043996700326
```

## Reference

```latex
唐培培, 戴晓霞, 谢龙汉编著.MATLAB科学计算及分析[M].北京: 电子工业出版社.2012.
```

