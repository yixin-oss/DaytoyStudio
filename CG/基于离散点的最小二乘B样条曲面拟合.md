---
title: 基于离散点的最小二乘B样条曲面拟合
---

## 序

​		基于离散点的最小二乘B样条曲面拟合是计算几何与计算机图形学中重要的研究基础之一, 利用MATLAB进行代码实现是十分必要的. 这里主要针对STL^*^文件中包含的三角网格顶点, 通过最小二乘拟合得到其对应的B样条曲面参数表达式, 并计算偏导数进一步得到第一第二基本形式等微分几何基本量.

[^*]: 为了简化流程得出合理结果, 这里用到的STL文件是利用UG的拟合曲面功能对源STL文件预处理后的结果, 通过调整次数得到规则的形状后, 再保存为新的STL文件导入到MATLAB中计算其表达式. (若UG能够导出拟合曲面的控制顶点及节点向量等信息, 这部分工作事实上可替代).

### 算法流程

- 读取STL文件  `TR=stlread('Tooth.stl')` (MATLAB 2018a及以上版本可用);
- 数据参数化, 选取$(x,y)$ 坐标线性映射到$[u,v]\in[0,1]^2$作为对应参数域$U\times V$;
- 确定B样条曲面的次数, 控制顶点个数, 根据参数$U,V$ 计算节点矢量$\rm knotU, knot V$;
- 构建B样条基函数对应的最小二乘系数矩阵, 右端项为离散点坐标信息, 构建线性方程组反求控制顶点信息 (可对坐标分量列三个方程组分别求解);
- 利用控制顶点坐标计算B样条曲面表达式并进行可视化验证;
- 计算B样条曲面的偏导数进一步确定微分几何基本量.

### Code--MATLAB R2023a

```matlab
% 读入STL文件
TR=stlread('Tooth_surfnew.stl');
% 读取离散数据点
V=TR.Points;
num=size(V,1);
F=TR.ConnectivityList;

% 离散数据点可视化
figure;
scatter3(V(:,1),V(:,2),V(:,3),'cyan','filled','o');

x=V(:,1);
y=V(:,2);
z=V(:,3);
% 参数化
% (x,y)->(u,v)
x_max=max(x);
x_min=min(x);
U=(x-x_min)/(x_max-x_min);

y_max=max(y);
y_min=min(y);
V=(y-y_min)/(y_max-y_min);

% 4*4次张量积型B样条
k=4;l=4;
% 控制顶点个数(m+1)*(n+1)
m=14;n=14;
% KTP计算节点向量
knotU=KTP(m,k,sort(U));
knotU(end)=knotU(end)-0.0001;
knotV=KTP(n,l,sort(V));
knotV(end)=knotV(end)-0.0001;

M=[];
for i=1:num
    u=U(i);v=V(i);
    A=coefficient_LS(k,l,m,n,knotU,knotV,u,v);
    M=[M;A];
end

Dx=M\x;
Dx=reshape(Dx,n+1,m+1);
Dx=Dx';

Dy=M\y;
Dy=reshape(Dy,n+1,m+1);
Dy=Dy';

Dz=M\z;
Dz=reshape(Dz,n+1,m+1);
Dz=Dz';

U=linspace(min(sort(U)),max(sort(U)),50);
V=linspace(min(sort(V)),max(sort(V)),50);
for i=1:length(U)
    for j=1:length(V)
        u=U(i);v=V(j);
        [x,y,z,~,~,~,~,~]=Bsurf(k,l,m,n,knotU,knotV,u,v,Dx,Dy,Dz);
        x1(i,j)=x;
        y1(i,j)=y;
        z1(i,j)=z;
    end
end
figure;
surf(x1,y1,z1);
colormap([0.8 0.8 0.8])
shading interp
axis off
camlight('headlight')
material('dull') % 设置材质为光滑，以便更好地反射光线

%% 将离散点保存为dat文件
a=x1(:);
b=y1(:);
c=z1(:);
points=[a,b,c];
% 指定要保存的dat文件名
filename = 'Tooth_points_data.dat';

% 打开dat文件以进行写入
fileID = fopen(filename, 'w');

% 将点坐标数据写入dat文件
for i = 1:size(points, 1)
    fprintf(fileID, '%.6f %.6f %.6f\n', points(i, 1), points(i, 2), points(i, 3));
end

% 关闭dat文件
fclose(fileID);
disp('点坐标数据已成功保存为dat文件。'); 

%% 计算微分几何基本量
for i=1:length(V)
    for j=1:length(U)
        u=U(j);v=V(i);
        [~,~,~,ru,rv,ruu,ruv,rvv]=Bsurf(k,l,m,n,knotU,knotV,u,v,Dx,Dy,Dz);
        normal=unit_vector(cross(ru,rv));%设计曲面在各点处法向量
        E=dot(ru,ru);F=dot(ru,rv);G=dot(rv,rv);
        L=dot(normal,ruu);M=dot(normal,ruv);N=dot(normal,rvv);
        % 系数矩阵
        I=[E,F;F,G];
        II=[L,M;M,N];
    end
end
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20241116164659783.png" alt="image-20241116164659783" style="zoom: 50%;" />

```matlab
function A=coefficient_LS(k,l,m,n,knotU,knotV,u0,v0)
% 子程序, 构建最小二乘系数矩阵

M=zeros(1,m+1);N=zeros(1,n+1);

for i=0:m
    M(i+1)=BaseFunction(i,k,u0,knotU);
end
for j=0:n
    N(j+1)=BaseFunction(j,l,v0,knotV);
end

A=M'*N;
A=A(:);
A=A';
```

```matlab
function [x,y,z,ru,rv,ruu,ruv,rvv]=Bsurf(k,l,m,n,knotU,knotV,u,v,Px,Py,Pz)

% 子程序, 计算B样条曲面坐标点及偏导数

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

```matlab
function unit=unit_vector(A)

% 子程序, 计算单位向量 

Model=sqrt(A(1)^2+A(2)^2+A(3)^2);
unit(1)=A(1)/Model;
unit(2)=A(2)/Model;
unit(3)=A(3)/Model;
end
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
U=T(1)*ones(1,n+p+2);

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
U(n+2:n+p+2)=T(end)+0.0001;
```









































