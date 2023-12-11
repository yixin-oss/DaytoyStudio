---
title: Matlab数据处理--数据降维
tags: 
- Matlab
- 数据分析
categories: Matlab与数据分析
---



“维数灾难”：随着维数的增加，数据分析技术会变得异常困难，是许多数据分析技术的瓶颈.

目前，常用的降维技术有两种，主成分分析、因子分析.

<!--more-->

## 一、主成分分析

有的问题变量之间存在相关性，它们包含的信息有所重叠，将变量进行线性组合后形成数量较少的新变量，新变量之间不相关，称为主成分.主成分反映原来变量的大量信息且所含信息不重叠，这种方法叫主成分分析.

主成分分析用较少的变量代替了原来较多的变量，实现了有效的降维，可以使问题简化.

Steps:

- 对原数据进行标准化转换.

- 计算样本的相关系数矩阵.

- 计算相关系数矩阵的特征值和特征向量.

- 计算主成分贡献率和累积贡献率，选择重要主成分.主成分的贡献率越大，说明包含原始信息量越大.

- 计算主成分载荷和主成分得分.

  **[r,p]=corrcoef(x)** 计算样本的相关系数矩阵

  ### 1.pcacov指令

  **[r,p]=corrcoef(x)** 计算样本相关系数矩阵

  **coeff=pcacov(v)** v是样本的协方差矩阵或相关系数矩阵，coeff是p个主成分的系数矩阵，第i列是第i个主成分的系数向量.

  **[coeff,latent]=pcacov(v)** latent是p个主成分的方差构成的向量.

  **[coeff,latent,explained]=pcacov(v)** explained是p个主成分向量的贡献率.

  ### 2.princomp指令

  根据样本的观测值矩阵进行主成分分析.

  **[coeff,score]=princomp(x)** x是主成分的系数矩阵，第i列是第i个主成分的系数向量，score是主成分得分矩阵，每行代表一个样本，每列代表一个主成分的得分.

  **[coeff,score,latent]=princomp(x)**  latent指样本协方差矩阵的特征向量.

  **[coeff,score,latent,tsquare]=princomp(x)** tsquare是样本的Hotelling T^2统计值，表示某样本和样本观测矩阵中心之间的距离，可以用它寻找远离中心的局端数据.

  **per=100*latent/sum(latent)** 主成分的贡献率

  ## 二、因子分析

  目的：寻找隐含在现有变量里的若干更基本的有代表性的变量并提取出来.这些更基本的变量叫公共因子.

  **Steps:**

  - 求样本的相关矩阵

  - 求特征值和特征向量.

  - 计算方差贡献率和累积方差贡献率.

  - 确定因子.

  - 进行因子旋转，使因子变量更具有解释性.

  - 计算因子得分.

    **[lambda,psi,T]=factoran(x,m,paraml,vall,param2,val2)** lambda是因子载荷值；psi是方差值构成的向量；T是旋转矩阵；x是样本数据；m是公共因子数量

    **[lambda,psi,T,stats,F]=factoran(x,m)** stats是相关信息统计；F是得分矩阵.

    ```
    r=corrcoef(x)%相关系数矩阵
    [lambda,psi,T]=factoran(r,3,'xtype','covariance','delta',0,'rotate','none')%设三个公共因子
    ctb=100*sum(lambda.^2)/size(x,2) %计算贡献率
    cumctb=cumsum(ctb) %计算累积贡献率
    ```

    