---
title: 内嵌物理知识神经网络(Physics Informed Neural Network, PINN)
---

# 内嵌物理知识神经网络(Physics Informed Neural Network, PINN)

## 论文引入

Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations 

[Physics-informed neural networks...](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

2019年,  布朗大学应用数学系研究团队提出了PINN, 并发表在《Journal of Computational Physics》. 自此, PINN成为AI物理领域中最主流的框架和关键词. 

## 基本介绍

​	所谓的物理信息神经网络, 其实就是将**物理方程作为约束加入到神经网络中**使其拟合得到的结果满足物理规律. 这个约束则体现在将物理方程迭代前后的差值加入到神经网络的损失函数中, 让物理方程参与指导整个训练过程. 这样, 神经网络在训练过程中优化的不仅仅是网络自己的损失函数, 还有物理方程每次迭代的差, 使得最后训练出来的结果满足物理规律. 

​	与传统的数据驱动的神经网络^*^相比, PINN在训练过程中施加了物理信息约束, 因而能用更少的样本学习到更具泛化能力的模型. 若将传统数值方法认为是纯物理驱动的方法, 那么PINN可以视为连接数据和物理知识的桥梁, 是数据与物理驱动的融合. 

*: 数据驱动的模型本身很适合处理大规模观测数据, 但由于外推或观测偏差导致模型泛化能力差, 其推测可能不符合实际物理规律, 也就是说不可信. 另一个缺点来源于物理现象的混沌本质, 比如分岔现象. 举个具体例子来说, 如果用神经网络来学习抛物线$y^2=x$上的采样点数据, 最终可能会得到一条过原点的直线, 显然不符合抛物线的结果, 那么就是不可信的.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240520203308883.png" alt="image-20240520203308883" style="zoom: 67%;" />

[内嵌物理知识神经网络（PINN）是个坑吗？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/468748367)

## 基本框架

![image-20240516132115525](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240516132115525.png)

考虑求解如下微分方程
$$
u({x})+N(x)=0,
$$
$N(x)$为关于$u(x)$的偏导数, 搭建神经网络如图所示. PINN根据偏微分方程, 在损失函数中考虑偏导数信息并使其最小化, 使得**神经网络逼近待求解的函数**$u(x)$. 损失函数中包含两项, 其中$MSE_u$是根据偏微分方程初始或边界条件构造的均方损失误差; $MSE_f$是根据微分方程构造的考虑偏导数信息的均方误差损失.

## 基于Pytorch利用PINN求解PDE

由于无网格化及自动微分的成熟, PINN可以很轻易的将微分的, 积分的, 确定的, 随机的等各种复杂的物理方程嵌入网络, 并且迭代求解过程也并不复杂.

### 实例1

考虑如下方程:
$$
\begin{equation}\begin{aligned}& \frac{\partial^2 u}{\partial x^2}-\frac{\partial^4u}{\partial y^4}=(2-x^2)e^{-y},\\& u_{yy}(x,0)=x^2, u_{yy}(x,1)=\frac{x^2}{e},\\& u(x,0)=x^2,u(x,1)=\frac{x^2}{e},\\& u(0,y)=0, u(1,y)=e^{-y}\end{aligned}\end{equation}
$$
真实解$u(x,y)=x^2e^{-y}$.

​		定义一个神经网络$\tilde{u}(x,y;\theta)$, 利用神经网络自动微分机制可以得到$\tilde{u}_{xx},\tilde{u}_{yy}, \tilde{u}_{yyyy}$, 然后在区域$[0,1]\times [0,1]$进行**随机采样**和**构造损失函数**, 由于有一项控制方程和七个边界条件, 因此需要构造7部分Loss:
$$
\begin{aligned}& L_1=\frac{1}{N_1} \sum_{\left(x_i, y_i\right) \in \Omega}\left(\tilde{u}_{x x}\left(x_i, y_i ; \theta\right)-\tilde{u}_{y y y y}\left(x_i, y_i ; \theta\right)-\left(2-x_i^2\right) e^{-y_i}\right)^2 \\& L_2=\frac{1}{N_2} \sum_{\left(x_i, y_i\right) \in[0,1] \times\{0\}}\left(\tilde{u}_{y y}\left(x_i, y_i ; \theta\right)-x_i^2\right)^2 \\& L_3=\frac{1}{N_3} \sum_{\left(x_i, y_i\right) \in[0,1] \times\{1\}}\left(\tilde{u}_{y y}\left(x_i, y_i ; \theta\right)-\frac{x_i^2}{e}\right)^2 \\& L_4=\frac{1}{N_4} \sum_{\left(x_i, y_i\right) \in[0,1] \times\{0\}}\left(\tilde{u}\left(x_i, y_i ; \theta\right)-x_i^2\right)^2 \\& L_5=\frac{1}{N_5} \sum_{\left(x_i, y_i\right) \in[0,1] \times\{1\}}\left(\tilde{u}\left(x_i, y_i ; \theta\right)-\frac{x_i^2}{e}\right)^2 \\& L_6=\frac{1}{N_6} \sum_{\left(x_i, y_i\right) \in\{0\} \times[0,1]}\left(\tilde{u}\left(x_i, y_i ; \theta\right)-0\right)^2 \\& L_7=\frac{1}{N_7} \sum_{\left(x_i, y_i\right) \in\{1\} \times[0,1]}\left(\tilde{u}\left(x_i, y_i ; \theta\right)-\exp \left(-y_i\right)\right)^2\end{aligned}
$$
训练次数$=10000$, 下图展示了网络计算近似解与真实结果的对比, 体现了PINN在求解PDE的有效性.

<center class="half">
    <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/PINNpred.png" width="350"/>
    <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/PINNexact.png" width="350"/>
</center>

### 实例2

考虑Burger's方程
$$
\begin{aligned}
& u_t+uu_x-(0.01/\pi)u_{xx}=0, x \in [-1, 1], t\in[0, 1]\\
& u(0,x)=-\sin(\pi x), \\
& u(t, -1)=u(t,1)=0.
\end{aligned}
$$
训练次数$=10000$, 下图展示了网络计算近似解, 若想对结果进行比较, 可以"构造"一个精确解$u(t,x)$代入方程中计算右端项.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240521163212216.png" alt="image-20240521163212216" style="zoom:80%;" />

下图展示了不同时刻$t=0.25, 0.75$对应的近似解.

<center class="half">
    <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240521164828192.png" width="350"/>
    <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240521164912583.png" width="350"/>
</center>

### Code

```python
"""
An example for PINN solving the following PDE
u_xx-u_yyyy=(2-x^2)*exp(-y)
"""

# Import necessary modules
import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
import matplotlib.pyplot as plt

# Domain and sampling
def interior(n=1000):
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    cond = (2 - x ** 2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def down_yy(n=100):
    x = torch.rand(n,1)
    # 生成与x形状相同的张量y, 元素值都是0
    y = torch.zeros_like(x)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond

def up_yy(n=100):
    x = torch.rand(n,1)
    # 生成与x形状相同的张量y, 元素值都是0
    y = torch.ones_like(x)
    cond = x ** 2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

def down(n=100):
    x = torch.rand(n,1)
    # 生成与x形状相同的张量y, 元素值都是0
    y = torch.zeros_like(x)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond

def up(n=100):
    x = torch.rand(n,1)
    # 生成与x形状相同的张量y, 元素值都是0
    y = torch.ones_like(x)
    cond = x ** 2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

def left(n=100):
    y = torch.rand(n,1)
    # 生成与x形状相同的张量y, 元素值都是0
    x = torch.zeros_like(y)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def right(n=100):
    y = torch.rand(n,1)
    # 生成与x形状相同的张量y, 元素值都是0
    x = torch.ones_like(y)
    cond = torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

# Neural Network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Loss
loss = nn.MSELoss()

# 递归调用自身计算更高阶导数
def gradients(u, x, order=1):
    if order == 1:
        # 设置梯度输出为与u形状相同的全1张量, 表示将一阶导数视为对u自身的梯度
        # 指示创建计算图以计算更高阶导数
        # 只计算输入张量x的导数
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                             create_graph=True,
                             only_inputs=True,)[0]
    else:
        return gradients(gradients(u, x), x, order = order - 1)

def l_interior(u):
    x, y, cond = interior()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, x, 2) - gradients(uxy, y, 4), cond)

def l_down_yy(u):
    x, y, cond = down_yy()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, y, 2), cond)

def l_up_yy(u):
    x, y, cond = up_yy()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(gradients(uxy, y, 2), cond)

def l_down(u):
    x, y, cond = down()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

def l_up(u):
    x, y, cond = up()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

def l_left(u):
    x, y, cond = left()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

def l_right(u):
    x, y, cond = right()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

# Training
u = PINN()
opt = optim.Adam(params=u.parameters())
Epochs = 10000
for epoch in range(Epochs+1):
    opt.zero_grad()
    l = l_interior(u) + l_up_yy(u) + l_down_yy(u) + l_up(u) + l_down(u) + l_left(u) + l_right(u)
    l.backward()
    opt.step()
    if epoch % 5 == 0:
        print(f'Epoch: {epoch}, Loss: {l.item()}')

# Inference
xc = torch.linspace(0, 1, 100)
xx, yy = torch.meshgrid(xc, xc)
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
u_pred = u(xy)
u_pred_np=u_pred.detach().numpy()
u_exact = xx ** 2 * torch.exp(-yy)
u_exact_np=u_exact.detach().numpy()
print("Max abs error is: ", float(torch.max(torch.abs(u_pred - u_exact))))


plt.figure(figsize=(8, 6))
plt.scatter(xy[:, 0], xy[:, 1], c=u_pred_np.flatten(), cmap='coolwarm', s=50, marker='o')
plt.colorbar()  # 添加颜色条
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Prediction')

plt.figure(figsize=(8, 6))
plt.scatter(xy[:, 0], xy[:, 1], c=u_exact_np.flatten(), cmap='coolwarm', s=50, marker='o')
plt.colorbar()  # 添加颜色条
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Exact')
plt.show()
```

```python
"""
An example for PINN solving the following Burgers PDE
u_t+uu_x-(0.01/pi)*u_xx=0
"""

# Domain and sampling
def interior(n=1000):
    t = torch.rand(n, 1)
    x = 2* torch.rand(n, 1) - 1
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), t.requires_grad_(True), cond

def left_t(n=100):
    x = 2* torch.rand(n, 1) - 1
    t = torch.zeros_like(x)
    cond = - torch.sin(torch.pi*x)
    return x.requires_grad_(True), t.requires_grad_(True), cond

def left_x(n=100):
    t = torch.rand(n, 1)
    x = - torch.ones_like(t)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), t.requires_grad_(True), cond

def right_x(n=100):
    t = torch.rand(n, 1)
    x = torch.ones_like(t)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), t.requires_grad_(True), cond

# Loss defination
def l_interior(u):
    x, t, cond = interior()
    uxy = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxy, t, 1) + uxy * gradients(uxy, x, 1) - (torch.tensor(0.01)/torch.pi) * gradients(uxy, x, 2), cond)

def l_left_t(u):
    x, t, cond = left_t()
    uxy = u(torch.cat([x, t], dim=1))
    return loss(uxy, cond)

def l_left_x(u):
    x, t, cond = left_x()
    uxy = u(torch.cat([x, t], dim=1))
    return loss(uxy, cond)

def l_right_x(u):
    x, t, cond = right_x()
    uxy = u(torch.cat([x, t], dim=1))
    return loss(uxy, cond)

# Training
u = PINN()
opt = optim.Adam(params=u.parameters())
Epochs = 10000
for epoch in range(Epochs + 1):
    opt.zero_grad()
    l = l_interior(u) + l_left_t(u) + l_left_x(u) + l_right_x(u)
    l.backward()
    opt.step()
    if epoch % 5 == 0:
        print(f'Epoch: {epoch}, loss: {l.item()}')
        
# Plot
xc = torch.linspace(-1, 1, 100)
tc = torch.linspace(0, 1, 100)
xx, tt = torch.meshgrid(xc, tc)
xx = xx.reshape(-1, 1)
tt = tt.reshape(-1, 1)
xt = torch.cat([xx, tt], dim=1)
u_pred = u(xt)
u_pred_np = u_pred.detach().numpy()


plt.figure(figsize=(8, 6))
plt.scatter(xt[:, 1], xt[:, 0], c=u_pred_np.flatten(), cmap='coolwarm', s=50, marker='o')
plt.colorbar()  # 添加颜色条
plt.xlabel('T')
plt.ylabel('X')
plt.title('Prediction')
plt.show() 

xx = torch.linspace(-1, 1, 100)
tt = 0.75 * torch.ones_like(xx)
xx = xx.reshape(-1, 1)
tt = tt.reshape(-1, 1)
xt = torch.cat([xx, tt], dim=1)
u_pred = u(xt)
u_pred_np = u_pred.detach().numpy()

plt.figure(figsize=(8, 6))
plt.plot(xt[:, 0], u_pred_np, marker='o')
plt.xlabel('x')
plt.ylabel('u')
plt.title('t=0.75')
plt.show()
```

*: 第二个实例沿用了第一个的网络结构, 包括微分的计算, 因此这一部分不再重复, 直接修改方程对应的loss信息和画图的细节即可.

## PINN优势

- **无网格化**

  PINN是一种无网格方法, 不需要事先定义初始网络也就不依赖于网格生成的质量, 面对复杂区域问题时更加灵活;

- **无监督学习**

  PINN主要依赖方程进行训练, 是一种典型的无监督学习方法, 不需要大量的标记数据, 降低数据收集和标注的成本;

- 高维表现良好

  与传统的基于网格的方法相比, PINN在高维空间中的精度和效率更高;

- 问题扩展

  PINN通常在复杂混合问题中表现良好，并可以扩展到解决逆问题, 如参数反演以及发现问题, 如参数调整, 表明PINN可广泛应用到各领域的不同场景.

从解决问题的角度看, 前两项体现了PINN潜在的巨大优势: 一方面或多或少弥补了传统解PDE的有限元方法的不足; 另一方面规避了神经网络大模型大数据的要求, 但其也存在着一定的局限性.

## PINN局限性

- **计算成本与工作量**

  虽然不需要大量标注数据训练, 但已经训练好的网络也还是不具备泛化能力, 换句话说PINN对于不同的初始条件和边界条件都需要重新训练. 如果问题的条件发生变化或需要解决不同场景的问题实例, 就需要重新搭建和训练网络, 可能会导致较大的工作量;

- Poor convergence

- 难以应对更复杂的问题, 如多尺度, 高频问题等

### Reference Link

[PINN及其“变种”-CSDN博客](https://blog.csdn.net/qq_58325633/article/details/135216767)

[PINN——加入物理约束的神经网络 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/544561165)

[深度学习求解偏微分方程（4）PINN局限与拓展 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/666566853)

[方程自己解（1）——物理信息神经网络（PINN）-CSDN博客](https://blog.csdn.net/jerry_liufeng/article/details/120727393)

[PINN.py · GitHub](https://github.com/zwqwo/PINN_scratch/blob/main/PINN.py)

### 小结

**最后的啰嗦**: PINN就是在神经网络通用近似理论的基础上, 通过加入偏导等微分算子给数值模拟加入了物理约束, 使得整个网络具有模拟物流规则的作用. 

**Key**: 通用近似理论, 物理信息传递(微分算子+损失函数构建)+自动微分(AD)



# 自适应损失平衡物理信息神经网络(LBPINN)

原有PINN的性能容易受到多重损失函数的加权组合的影响.

$\Downarrow$

探究损失函数权重对PINN扩展性能的影响 $\rightarrow$ 自适应地学习损失函数权重

:star:基于最大化多任务深度学习问题的不确定性高斯似然来加权多个损失函数.

1) 最小化多限制的损失函数 $\rightarrow$ 多目标优化问题

2) 模型架构限制 $\rightarrow$ 预测不确定性

- 建立高斯概率模型, 通过每个损失项的自适应权重定义损失函数;
- 基于最大似然估计更新每轮训练过程中的自适应权重.

## 基本原理

假设神经网络输出满足高斯概率模型, 均值为$\hat{u}$, 方差为$\varepsilon_d$, 在深度学习模型中一般可将输出的方差视作不确定性, $\varepsilon_d$被称为不确定性参数, 由此
$$
p(u|\hat{u}(x,t;\theta))=N(\hat{u}(x,t;\theta),\varepsilon_d^2).
$$
在训练过程中, 不确定性参数可由似然函数进行优化, 通常, 优化目标为似然函数最大化
$$
-\log p(u|\hat{u}(x,t;\theta)) \propto \frac{1}{2\varepsilon_d^2}\|u-\hat{u}(x,t;\theta)\|^2+\log \varepsilon_d = \frac{1}{2\varepsilon_d^2}L_{data}(\theta)+\log \varepsilon_d.
$$
进一步拓展到多个输出网络,  通常假设输出都服从高斯分布且独立同分布, 可得到
$$
\begin{aligned}
p(u,g|\hat{u}(x,t;\theta)& = p(u|\hat{u}(x,t;\theta))\cdot p(u,g|\hat{u}(x,t;\theta)\\
& = N(\hat{u}(x,t;\theta),\varepsilon_d^2)\cdot N(\hat{u}(x,t;\theta),\varepsilon_b^2).
\end{aligned}
$$
由此, PINN多项损失可被视作多输出模型, 考虑不确定性下的PINN优化目标为
$$
\begin{equation}
L(\varepsilon;\theta;N)=\frac{1}{2\varepsilon_f^2}L_{PDE}(\theta;N_f)+\frac{1}{2\varepsilon_b^2}L_{BC}(\theta;N_b)+\frac{1}{2\varepsilon_i^2}L_{PDE}(\theta;N_i)+\frac{1}{2\varepsilon_d^2}L_{PDE}(\theta;N_d)+\log \varepsilon_f \varepsilon_b \varepsilon_i \varepsilon_d.
\end{equation}
$$
最后通过梯度优化算法同时优化$\theta;\varepsilon_f, \varepsilon_b, \varepsilon_d$, 训练模型.















