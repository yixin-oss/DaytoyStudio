---
title: 双三次B样条曲面绘制及微分量计算
categories: 计算几何
mathjax: true
---

# 序

​        偶然发现自己之前竟然一直忽略了B样条曲面的代码，由于B样条曲面的灵活性和局部调整性，可通过改变控制顶点来很方便地生成不同形状的曲面，因此可用于数控加工中各种加工曲面实例的设计，最近可能会用到，故补充这一部分内容. 关于B样条基函数等基础知识不再赘述，可具体查阅任意文献[1-5]，代码也很容易.

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

# 微分量计算

​        由于双三次B样条曲面$r(u,v)=\sum_{i=0}^m\sum_{j=0}^n P_{i,j}N_{i,k}(u)N_{j,l}(v),P_{i,j}\in R^3$是参数曲面，故可以与微分几何知识建立联系，计算其微分量.

​        记B样条基函数$N_{i,k}(u),N_{j,l}(v)$对应的节点向量分别为
$$
knotU=\{u_0,u_1,...,u_{m+k+1}\}, knotV=\{v_0,v_1,...,v_{n+l+1}\}.
$$
对基函数求导：
$$
\begin{aligned}
& N_{i,k}^{'}(u)=\frac{k}{u_{i+k}-u_{i}}N_{i,k-1}(u)-\frac{k}{u_{i+k+1}-u_{i+1}}N_{i+1,k-1}(u),\\
& N^{''}_{i,k}(u)=\frac{k}{u_{i+k}-u_i}[\frac{k-1}{u_{i+k-1}-u_i}N_{i,k-2}(u)-\frac{k-1}{u_{i+k}-u_{i+1}}N_{i+1,k-2}(u)]\\
&-\frac{k}{u_{i+k+1}-u_{i+1}}[\frac{k-1}{u_{i+k}-u_{i+1}}N_{i+1,k-2}(u)-\frac{k-1}{u_{i+k+1}-u_{i+2}}N_{i+2,k-2}(u)].

\end{aligned}
$$
对于$N_{j,k}(v)$的求导同理，只需进行相应符号替换即可. 基函数的导数得到后，可进一步计算参数曲面在每一点处的偏导数$r_u,r_v,r_{uu},r_{u,v},r_{vv}$，相应的单位法向量为$n=\frac{r_u\times r_V}{||r_u\times r_v||}$.

​        由微分几何知识，

- 曲面的第一基本形式

$$
I=Edu^2+2Fdudv+Gdv^2\\
E=r_u\cdot r_u, F=r_u\cdot r_v, G=r_v\cdot r_v.
$$

- 曲面的第二基本形式

$$
II=Ldu^2+2Mdudv+Ndv^2\\
L=r_{uu}\cdot n, M=r_{uv}\cdot n, N=r_{vv}\cdot n.
$$

- 曲面在一点处法曲率

$$
k_{n}=\frac{II}{I}.
$$

- 曲面在一点处主方向$(du:dv)$满足如下方程

$$
(EM-FL)du^2+(EN-GL)dudv+(FN-GM)dv^2=0
$$

曲面在一点处主曲率$k_1,k_2$满足下面方程
$$
(EG-F^2)k_N-(LG-2MF+NE)k_N+(LN-M^2)=0
$$
至此，与B样条曲面相关的微分量均可以计算，下面只列出计算偏导的代码供参考.

```matlab
function [U,V,x,y,z,ru,rv,ruu,ruv,rvv] = Testfun(u,v)
 
% 用B样条曲面构建加工模型实例

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

M=zeros(1,m+1);N=zeros(1,n+1);
for p=0:m
    M(p+1)=BaseFunction(p,k,u,knotU);
end
for q=0:n
    N(q+1)=BaseFunction(q,l,v,knotV);
end
x=M*Px*N';
y=M*Py*N';
z=M*Pz*N';

Mu=zeros(1,m+1);Muu=Mu;
for i=0:m
    length1=knotU(i+k+1)-knotU(i+1);
    length11=1/length1;
    if length1==0
        length11=0;
    end
    length2=knotU(i+k+2)-knotU(i+2);
    length22=1/length2;
    if length2==0
        length22=0;
    end
    Mu(i+1)=k*length11*BaseFunction(i,k-1,u,knotU)-...
        k*length22*BaseFunction(i+1,k-1,u,knotU);
    
    length3=knotU(i+k)-knotU(i+1);
    length33=1/length3;
    if length3==0
        length33=0;
    end
    length4=knotU(i+k+1)-knotU(i+2);
    length44=1/length4;
    if length4==0
        length44=0;
    end
    length5=knotU(i+k+2)-knotU(i+3);
    length55=1/length5;
    if length5==0
        length55=0;
    end
    Muu(i+1)=k*(length11*(k-1)*(length33*BaseFunction(i,k-2,u,knotU)-length44*BaseFunction(i+1,k-2,u,knotU))-...
        length22*(k-1)*(length44*BaseFunction(i+1,k-2,u,knotU)-length55*BaseFunction(i+2,k-2,u,knotU)));
    
end

Nv=zeros(1,n+1);Nvv=Nv;
for j=0:n
    length1=knotV(j+l+1)-knotV(j+1);
    length11=1/length1;
    if length1==0
        length11=0;
    end
    length2=knotV(j+l+2)-knotV(j+2);
    length22=1/length2;
    if length2==0
        length22=0;
    end
    Nv(j+1)=l*length11*BaseFunction(j,l-1,v,knotV)-...
        l*length22*BaseFunction(j+1,l-1,v,knotV);
    length3=knotV(j+l)-knotV(j+1);
    length33=1/length3;
    if length3==0
        length33=0;
    end
    length4=knotV(j+l+1)-knotV(j+2);
    length44=1/length4;
    if length4==0
        length44=0;
    end
    length5=knotV(j+l+2)-knotV(j+3);
    length55=1/length5;
    if length5==0
        length55=0;
    end
    Nvv(j+1)=l*(length11*(l-1)*(length33*BaseFunction(j,l-2,v,knotV)-length44*BaseFunction(j+1,l-2,v,knotV))-...
        length22*(l-1)*(length44*BaseFunction(j+1,l-2,v,knotV)-length55*BaseFunction(j+2,l-2,v,knotV)));
end

ru=[Mu*Px*N',Mu*Py*N',Mu*Pz*N'];
rv=[M*Px*Nv',M*Py*Nv',M*Pz*Nv'];
ruu=[Muu*Px*N',Muu*Py*N',Muu*Pz*N'];
ruv=[Mu*Px*Nv',Mu*Py*Nv',Mu*Pz*Nv'];
rvv=[M*Px*Nvv',M*Py*Nvv',M*Pz*Nvv'];

end
```

# Reference

```
[1] 苏步青，刘鼎元著. 计算几何[M]. 上海：上海科学技术出版社, 1981.
[2] 王国瑾，汪国昭等著. 计算机辅助几何设计[M]. 北京：高等教育出版社, 2004.
[3] 施法中. 计算机辅助几何设计与非均匀有理B样条 修订版[M]. 北京：高等教育出版社, 2013.
[4] Piegl L, Tiller W. The NURBS book. Berlin: Springer-Verlag, 1997.
[5] 王仁宏，李崇君，朱春钢. 计算几何教程[M]. 北京：科学出版社, 2008.
```

