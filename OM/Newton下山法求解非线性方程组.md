## 代码

```matlab
function [x,n]=NonLinearEquations_NewtonDown(x0,err)
%{
函数功能：牛顿下山法求解非线性方程组的解；
输入：
    x0：初始值；
    err：精度阈值；
输出：
    x:近似解；
    n：迭代次数；
示例：
clear;clc;
[r,n]=NonLinearEquations_NewtonDown([0 0 0],1e-6)
%}
% = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
x=x0-myfun(x0)/dmyfun(x0);
n=1;
eps=1;
while eps > err
    x0=x;
    tol=1;
    w=1;
    F1=norm(myfun(x0));
    while tol>= 0
        x=x0-w*myfun(x0)/dmyfun(x0);
        tol=norm(myfun(x))-F1;
        w=w/2;
    end
    eps = norm(x-x0);
    n=n+1;
    if(n>1000)
        disp('迭代步数太多，可能不收敛！');
        return
    end
end

function f=myfun(x)
x1=x(1);
x2=x(2);
x3=x(3);
f(1)=3*x1-cos(x2*x3)-1/2;
f(2)=x1^2-81*(x2+0.1)+sin(x3)+1.06;
f(3)=exp(-x1*x2)+20*x3+1/3*(10*pi-3);

function df=dmyfun(x)
x1=x(1);
x2=x(2);
x3=x(3);
df=[3,x3*sin(x2*x3),x2*sin(x2*x3);2*x1,-81,cos(x3);-x2*exp(-x1*x2),-x1*exp(-x1*x2),20];
```

## 实例

```matlab
[x,n]=NonLinearEquations_NewtonDown([0 0 0],1e-6)
```

运行结果：

```matlab
x =

    0.4996   -0.0900   -0.5259
n =

     6
```

方程组近似解为$[0.4996 \quad -0.0900 \quad  -0.5259]$，迭代次数为6次.

## Reference

[Matlab学习手记——非线性方程组求解：牛顿下山法](https://blog.csdn.net/u012366767/article/details/81698155?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-1.no_search_link&spm=1001.2101.3001.4242.2)

[数值计算（三十一）修正牛顿法III求解方程的根](https://zhuanlan.zhihu.com/p/124857563)