## 测地距离计算的热方法

Crane et al.(2017) 提出了利用热传导方程来计算网格测地距离的方法. 假想一根烫的针尖接触到曲面上的一点，热量会随着时间的推移逐渐扩散. 在短时间内, 热量会优先沿着测地距离最短的路径传播, 因此测地距离的计算与热传导过程密切相关.

```latex
Crane, Keenan, Weischedel, Clarisse, & Wardetzky, Max. (2017). The heat method for distance computation. Communications of the ACM, 60(11), 90-99.
```

### 理论基础

测地距离的计算可归结为求解**程函方程(Eikonal equation)** 
$$
\begin{equation*}
|\nabla \phi|=1, \phi|_\gamma=0,
\end{equation*}
$$
符号$\phi$就是测地距离, 即满足测地距离梯度的模长恒等于1, 源点测地距离值为0.

现有的求解思路多聚焦于非线性方法, 计算量大且求解困难. [Crane et al. 2017] 方法的主要贡献是通过热方程将问题**转换成线性形式**求解. 具体来说, 就是要求解**一对稀疏线性方程组**, 第一个是关于**热方程的解**, 第二个是**泊松方程的解**.

### 算法流程

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20241106212651315.png" alt="image-20241106212651315" style="zoom90%;" />

**输入**: 三角网格曲面各顶点坐标$V$, 三角面片连接关系$F$.

**I. 对固定时间$t$, 求解热方程**.

**(a) 时间离散化**, 将热传导方程$\frac{\partial u}{\partial t}=\Delta u$通过向后差分得到
$$
\begin{equation}
(I-t\Delta)u_t=u_0,
\end{equation}
$$
其中$I$为单位矩阵, $t$为时间间隔, $\Delta$ 为Laplacian算子, $u_t$为t时刻的热状态, $u_0$为初始时刻热状态.

> 对于时间步长的选取, 文献中给出的结果是取三角网格所有边平均长度的平方, i.e. $t=h^2$, 可人为调整尝试.

**(b) 空间离散化**, 在三角网格表示中, 离散Laplacian坐标为
$$
\begin{equation*}
(Lu)_i=\frac{1}{2M_i}\sum_j(\cot\alpha_{ij}+\cot\beta_{ij})(u_j-u_i),
\end{equation*}
$$
其中$M_i$是顶点$i$邻接的所有三角形面积的$1/3$, $j$ 是所有邻接顶点索引. 对于有$V$个顶点的网格模型, 基于上式可以列出 $V$ 个方程, 将它们写成矩阵形式, i.e.
$$
\begin{equation*}
L=M^{-1}L_c,
\end{equation*}
$$
其中$M=diag(M_1,...,M_{|V|})$ 是包含每个顶点相关面积的对角矩阵, $L_c$是关于对角余切($Cot$)的Laplacian矩阵. 将其代入到差分方程中, 得到**第一个**线性方程组
$$
\begin{equation*}
(M-tL_c)u=M\delta_{\gamma}, \delta_\gamma=
\begin{cases}
1, \gamma,\\
0,  \Omega/\gamma.
\end{cases}
\end{equation*}
$$

求解得到三角网格每个顶点处的离散函数值$u_i,i=1,...,|V|$.

![image-20241106213013770](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20241106213013770.png)

**II. 计算函数负单位梯度构建向量场**.

**(a) 梯度的计算** (求出来的是每个三角片中的梯度):
$$
\begin{equation*}
\nabla u = \frac{1}{2A_f}\sum_iu_i(N\times e_i),
\end{equation*}
$$
其中$A_f$是三角片的面积, $N$是单位外法向量, $e_i$是逆时针方向的边向量, $u_i$是对边的值. 上述公式实质描述了三角片内每个顶点及其对边向量对整体梯度的贡献. 

**(b) 构建负单位梯度向量场(指向测地距离增加的方向)**:
$$
\begin{equation*}
X=-\frac{\nabla u}{\|\nabla u\|}.
\end{equation*}
$$


**III. 计算离散散度, 求解Poisson方程**.

**(a) 散度的计算** (求出来的是每个顶点处的散度):
$$
\begin{equation*}
\nabla \cdot X=\frac{1}{2}\sum_j\cot \theta_1(e_1\cdot X_j)+\cot \theta_2(e_2\cdot X_j),
\end{equation*}
$$
求和$j$是对包含顶点$i$的所有三角片对应的$X_j$进行的, $e_1, e_2$是顶点$v_i$出发的两个边向量, $\theta_1, \theta_2$ 是边向量的对角.

**(b) 构建Poisson方程**

根据泛函能量极小化, 寻找测地函数拟合向量场$X$:
$$
\begin{equation*}
\min_{\phi}\int_M|\nabla\phi-X|^2.
\end{equation*}
$$
根据变分法, 上式等价于求解Euler-Lagrange方程
$$
\begin{equation*}
\Delta \phi=\nabla \cdot X,
\end{equation*}
$$
即得到**第二个**线性方程组
$$
\begin{equation*}
L_c\phi=b, b\in \mathcal{R}^{|V|},
\end{equation*}
$$
这里$b$表示单位向量场的散度, $\phi$即为网格上顶点到热源点的测地距离. 下图为原文(Crane et al. 2017)中关于热函数$u$, 梯度向量场$\nabla u$, 单位负梯度向量场$X$以及测地函数$\phi$的可视化.

![image-20241107125559852](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20241107125559852.png)

这里用MATLAB针对已知的STL曲面实现上述算法, 得到如下结果:

- 边界提取→提取边界点作为热源;

从每个三角片中提取三条边对应顶点的标号信息并统计出现次数, 若只出现一次则该边两顶点就是边界点.

- 三角片外法向量可视化验证;
- 函数$u$, 向量场$X$, 测地距离函数$\phi$ 可视化验证;

|                     三角网格曲面边界提取                     |                          求解热方程                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20241107144732139.png" style="zoom:50%;" /> | <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/Heat flow.png" style="zoom:50%;" /> |
|                          向量场$X$                           |                      测地距离函数$\phi$                      |
| <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/X.png" style="zoom:50%;" /> | <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/SBG-Heat.png" style="zoom:50%;" /> |

## Code


```matlab
%% 说明
% Geodesics in Heat:A New Approach to Computing Distance Based on Heat Flow
% [Crane K 2017]
% 基于热理论的测地场计算

ft=19;
% 初始化u,v取值范围
[aa,bb,cc,dd,~,~,~,~,~,~,~,~]=Testfun(0,0,ft);
% 设置采样点规模
u_grid=49;v_grid=49;
U=aa:(bb-aa)/u_grid:bb;
V=cc:(dd-cc)/v_grid:dd;
for i=1:length(V)
    for j=1:length(U)
        u=U(j);v=V(i);
        [~,~,~,~,x,y,z,~,~,~,~,~]=Testfun(u,v,ft);
        x1(i,j)=x;
        y1(i,j)=y;
        z1(i,j)=z;
    end
end
tri=delaunay(x1,y1);
V = [x1(:), y1(:), z1(:)];
F = tri;

fv.vertices=V; % 顶点坐标集合 |V|*3
fv.faces=F; % 三角面片连接关系集合 |F|*3

% 三角网格曲面可视化
figure;
patch(fv,'FaceColor',       [0.8 0.8 0.8], ...
         'EdgeColor',       'none',        ...
         'FaceLighting',    'gouraud',     ...
         'AmbientStrength', 0.15);
camlight('headlight');
material('dull');
hold on


nV=size(V,1); % |V|
nF=size(F,1); % |F|

%% 边界提取算法--提取边界点作为热源
% 想法：从每个三角片中提取三条边的信息并统计出现次数, 若只出现一次则该边两顶点就是边界点
% 从三角片中提取边
edges = [];
for i = 1:size(F, 1)
    e1 = sort(F(i, [1 2])); % v1 v2
    e2 = sort(F(i, [2 3])); % v2 v3
    e3 = sort(F(i, [3 1])); % v3 v1
    edges = [edges; e1; e2; e3];
end

% 统计各边出现的次数
[unique_edges, ~, idx] = unique(edges, 'rows');
edge_counts = histcounts(idx, 'BinMethod', 'integers');
% 识别边界边
boundary_edges = unique_edges(edge_counts == 1, :);
% 提取边界点标号
boundary_indices = unique(boundary_edges(:));
% 提取边界点坐标信息并可视化
% boundary_points=V(boundary_indices,:);
% plot3(boundary_points(:,1),boundary_points(:,2),boundary_points(:,3),'r*');
% 将边界点作为热源点
source=boundary_indices;

source_points=V(source,:);
plot3(source_points(:,1),source_points(:,2),source_points(:,3),'r*');
%% Step 1：求解热方程--固定的时间步长t，构建拉普拉斯余切矩阵L，三角面积对角阵M，符号函数右端项u0
% 差分法(I-t△)u_t=u_0
% 由离散拉普拉斯△表示L_c=inv(M)*L => (M-t*L)U=M*u0 
tic;
% 选取时间步长
t=10;

% 计算面积对角矩阵M=massmatrix(V,F,'barycentric')
M=zeros(nV,nV);
for i=1:nF
    v1=F(i,1);
    v2=F(i,2);
    v3=F(i,3);
    e1=V(v2,:)-V(v1,:);
    e2=V(v3,:)-V(v1,:);
    % 计算三角片的面积
    area=1/2*norm(cross(e1,e2));
    % 每个顶点获得邻接三角片面积的1/3
    M(v1,v1)=M(v1,v1)+area/3;
    M(v2,v2)=M(v2,v2)+area/3;
    M(v3,v3)=M(v3,v3)+area/3;
end
M=2*M;

% 计算拉普拉斯余切矩阵L=cotmatrix(V,F)
% L是稀疏的对称三对角矩阵
% isequal(L,L')==1
L=zeros(nV,nV);
for i=1:nF
    v1=F(i,1);
    v2=F(i,2);
    v3=F(i,3);
    % 计算边向量
    e1=V(v3,:)-V(v2,:); % v2v3
    e2=V(v3,:)-V(v1,:); % v1v3
    e3=V(v2,:)-V(v1,:); % v1v2
    % 计算边对应角的正切值cot(alpha)
    cot_alpha1=dot(e1,e2)/norm(cross(e1,e2)); % v3顶角，对边v1v2
    cot_alpha2=dot(-e2,-e3)/norm(cross(-e2,-e3)); % v1顶角，对边v2v3
    cot_alpha3=dot(e3,-e1)/norm(cross(e3,-e1)); % v2顶角，对边v1v3
    % 更新矩阵L的值
    L(v1,v2)=L(v1,v2)+cot_alpha1;
    L(v2,v1)=L(v2,v1)+cot_alpha1;
    
    L(v2,v3)=L(v2,v3)+cot_alpha2;
    L(v3,v2)=L(v3,v2)+cot_alpha2;
    
    L(v1,v3)=L(v1,v3)+cot_alpha3;
    L(v3,v1)=L(v3,v1)+cot_alpha3;
end

for i = 1:nV
    L(i, i) = -sum(L(i, :));
end

u0=zeros(nV,1);
% 热源点右端项值为1，其余点右端项为0
u0(source)=1;
% 求解第一个线性方程组Au=B，解得u
A=M-t*L;
B=M*u0;
% 处理边界条件: Dirichlet
for i=1:length(source)
    idx=source(i);
    A(idx,:)=0;
    A(idx,idx)=1;
    B(idx)=1;
end
% 求解线性方程组 Au=B
u=pinv(A)*B;

% 方程解u的可视化
figure;
colormap jet; % 使用 jet 颜色映射
trisurf(F, V(:, 1), V(:, 2), V(:, 3), u, 'FaceColor', 'interp', 'EdgeColor', 'none');
colorbar; % 显示颜色条
xlabel('X');
ylabel('Y');
zlabel('Z');
% scatter3(V(:,1), V(:,2), V(:,3), 50, u, 'filled');


%% Step 2：构建单位负梯度场——三角片离散梯度计算，单位化负梯度
figure;
% 绘制三角面片
% trisurf(F, V(:,1), V(:,2), V(:,3),'FaceColor',[0.8 0.8 0.8], ...
%          'EdgeColor','none');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% title('Triangle Faces with Normal Vectors and Gradient Vectors');
% grid on;
hold on

X=[];
for i=1:nF
    v1=F(i,1);
    v2=F(i,2);
    v3=F(i,3);
    % 计算边向量
    e1=V(v3,:)-V(v2,:);
    e2=V(v1,:)-V(v3,:);
    e3=V(v2,:)-V(v1,:);
    % 计算外法向量, 三角片的面积
    N=cross(e1,-e3);
    N=N/norm(N);
    % 可视化验证
    v_centric=(V(v1,:)+V(v2,:)+V(v3,:))/3; % 重心坐标
    % quiver3(v_centric(:,1), v_centric(:,2), v_centric(:,3), ...
    %     N(:,1), N(:,2), N(:,3), 'Color', 'r');
    area=1/2*norm(cross(e1,-e3));
    
    % 计算每个顶点对梯度的贡献
    grad_v1=cross(N,e1)/(2*area);
    grad_v2=cross(N,e2)/(2*area);
    grad_v3=cross(N,e3)/(2*area);
    
    grad=u(v1,:)*grad_v1+u(v2,:)*grad_v2+u(v3,:)*grad_v3;
    grad_norm=norm(grad);
    grad_=-grad/grad_norm;
    
    quiver3(v_centric(:,1), v_centric(:,2), v_centric(:,3), ...
        grad_(1), grad_(2), grad_(3),'Color', 'b');
    
    X=[X;grad_];
end
axis([min(V(:,1)) max(V(:,1)) min(V(:,2)) max(V(:,2))])

%% Step 3：求Poisson方程——求离散散度，求解线性方程组
Div=zeros(nV,1);
for i=1:nF
    v1=F(i,1);
    v2=F(i,2);
    v3=F(i,3);
    X_=X(i,:);
    
    % v1顶点
    e1=V(v2,:)-V(v1,:);
    e2=V(v3,:)-V(v1,:);
    e32=V(v2,:)-V(v3,:);
    cot_theta1=dot(-e2,e32)/norm(cross(-e2,e32));
    cot_theta2=dot(-e32,-e1)/norm(cross(-e32,-e1));
    Div(v1)=Div(v1)+cot_theta1*dot(e1,X_)+cot_theta2*dot(e2,X_);
    
    % v2顶点
    e1=V(v3,:)-V(v2,:);
    e2=V(v1,:)-V(v2,:);
    e13=V(v3,:)-V(v1,:);
    cot_theta1=dot(-e2,e13)/norm(cross(-e2,e13));
    cot_theta2=dot(-e13,-e1)/norm(cross(-e13,-e1));
    Div(v2)=Div(v2)+cot_theta1*dot(e1,X_)+cot_theta2*dot(e2,X_);
    
    % v3顶点
    e1=V(v1,:)-V(v3,:);
    e2=V(v2,:)-V(v3,:);
    e21=V(v1,:)-V(v2,:);
    cot_theta1=dot(-e2,e21)/norm(cross(-e2,e21));
    cot_theta2=dot(-e21,-e1)/norm(cross(-e21,-e1));
    Div(v3)=Div(v3)+cot_theta1*dot(e1,X_)+cot_theta2*dot(e2,X_);
end
Div=1/2*Div;

for i=1:length(source)
    idx=source(i);
    L(idx,:)=0;
    L(idx,idx)=1;
    Div(idx)=0;
end
% 求解Poisson方程对应的线性方程组
phi=pinv(L)*Div;
phi=phi-min(phi);

Time=toc;
fprintf('Computational time: %.2f seconds\n',Time);

% 测地距离色彩可视化
figure;
% scatter3(V(:,1), V(:,2), V(:,3), 50, phi, 'filled');
colormap jet; % 使用 jet 颜色映射
trisurf(F, V(:, 1), V(:, 2), V(:, 3), phi, 'FaceColor', 'interp', 'EdgeColor', 'none');
colorbar; % 显示颜色条
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Triangulated Mesh with Geodesic function');
```

