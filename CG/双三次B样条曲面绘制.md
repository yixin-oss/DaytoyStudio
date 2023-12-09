# 序

​        偶然发现自己之前竟然一直没写B样条曲面的代码，由于B样条曲面的灵活性和局部调整性，可通过改变控制顶点来很方便地生成不同形状的曲面，因此可用于数控加工中各种加工曲面实例的设计，最近可能会用到，故补充这一部分内容.

​        相关的理论及公式不再赘述，可具体查阅文献[1-5]，代码也很容易.

<!--more-->

# Code

```matlab
%构造双三次B样条曲面

U=0:0.01:1;
V=0:0.01:1;
%双三次B样条基函数
k=3;l=3;
%控制顶点
Px=[1, 1, 2, 1; 4, 4, 4, 4; 7, 6, 7, 8; 10, 9, 10, 9];
Py=[1, 3, 6, 9; 0, 3, 6, 9; 0, 3, 6, 9; 1,  4, 7, 10];
Pz=[3, 5, 5, 2; 4, 6, 7, 4; 4, 7, 6, 5; 2,  4, 5, 4];%控制顶点
m=size(Px,1)-1;n=size(Px,2)-1;
% KTP计算节点向量
knotU=KTP(m,k,U);
knotV=KTP(n,l,V);
U(end)=U(end)-0.0001;
V(end)=V(end)-0.0001;

x1=zeros(length(V),length(U));
y1=x1;z1=x1;

for i=1:length(V)
    for j=1:length(U)
        u=U(j);v=V(i);
        M=zeros(1,m+1);N=zeros(1,n+1);
        for p=0:m
            M(p+1)=BaseFunction(p,k,u,knotU);
        end
        for q=0:n
            N(q+1)=BaseFunction(q,l,v,knotV);
        end
        x1(i,j)=M*Px*N';
        y1(i,j)=M*Py*N';
        z1(i,j)=M*Pz*N';
    end
end
surf(x1,y1,z1);
```

```matlab
function [Nip_u]=BaseFunction(i,p,u,NodeVector)
%利用de Boor-Cox 公式计算基函数Ni_p(u),i是节点序号,p是次数,NodeVector为节点向量
%采用递归方式实现
if p == 0
    if (u >= NodeVector(i+1)) && (u < NodeVector(i+2)) %节点序号从0开始，但matlab从1开始，所以这里用i+1
        Nip_u = 1;
    else
        Nip_u = 0;
    end
else
    length1 = NodeVector(i+p+1) - NodeVector(i+1);
    length2 = NodeVector(i+p+2) - NodeVector(i+2); %支撑区间长度
    if length1 == 0  %规定0/0=0
        length1 = 1;
    end
    if length2 == 0
        length2 = 1;
    end
    Nip_u=(u-NodeVector(i+1))/length1*BaseFunction(i,p-1,u,NodeVector)+...
        +(NodeVector(i+p+2)-u)/length2*BaseFunction(i+1,p-1,u,NodeVector);
end    
end
```

```matlab
function U = KTP(n,p,T)

%KTP方法计算B样条曲线节点向量U
%n=控制顶点-1
%m=数据点个数-1
m=length(T)-1;
U=zeros(1,n+p+2);
if m==n
    for j=1:n-p
        U(j+p+1)=1/p*sum(T(j:j+p-1)); %%去掉了加1
    end
else  
    c=m/(n-p+1);
    for j=1:n-p
        i=fix(j*c);
        alpha=j*c-i;
        U(p+j+1)=(1-alpha)*T(i-1+1)+alpha*T(i+1);%参数标号从0开始，matlab从1开始记
    end
end
U(n+2:n+p+2)=T(end);
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/%E8%BF%90%E8%A1%8C%E7%BB%93%E6%9E%9C.png" style="zoom: 50%;" />

# Reference

```
[1] 苏步青，刘鼎元著. 计算几何[M]. 上海：上海科学技术出版社, 1981.
[2] 王国瑾，汪国昭等著. 计算机辅助几何设计[M]. 北京：高等教育出版社, 2004.
[3] 施法中. 计算机辅助几何设计与非均匀有理B样条 修订版[M]. 北京：高等教育出版社, 2013.
[4] Piegl L, Tiller W. The NURBS book. Berlin: Springer-Verlag, 1997.
[5] 王仁宏，李崇君，朱春钢. 计算几何教程[M]. 北京：科学出版社, 2008.
```

