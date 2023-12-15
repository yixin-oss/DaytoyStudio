---
title: python机器学习之鸢尾花分类问题
tags: 
- python
- 机器学习
categories: python学习笔记

---

<font size=4 face="楷体">**已知：**鸢尾花有三个品种：setosa，versicolor,virginica，给出鸢尾花花瓣长度、宽度及花萼的长度宽度作为测量数据，测量结果单位都是cm.</font>

<font size=4 face="楷体">**目标：**构建机器学习模型，从已知品种的鸢尾花测量数据中进行学习，预测新鸢尾花的品种.</font>

<!-- more -->

<font size=4 face="楷体">**分析：**有已知品种的测量数据，这是一个监督学习问题，数据集中每朵花分属于三个类别，这是三分类问题.</font>

<font size=4 face="楷体">以下代码均在 **Jupyter Notebook**  中编写及运行.</font>

<font size=5 face="楷体"> 必要的库调用</font>

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
```

<font size=4 face="楷体">mglearn库可在Jupyter Notebook 代码块中直接运行以下命令行下载，其它的库下载方式同理，下载成功后即可调用.</font>

```
pip install mglearn
```

<font size=5 face="楷体">数据准备</font>

<font size=4 face="楷体">数据集包含在scikit-learn的datasets模块中，可调用load_iris函数加载数据.
    </font>

```python
from sklearn.datasets import load_iris
iris_dataset=load_iris()
#这里返回的是Bunch对象，与字典类似，包含键和值.
```

<font size=4 face="楷体">利用print和.format方法可以查看数据集的相关信息.</font>

```python
print('Keys of iris_dataset:\n{}'.format(iris_dataset.keys()))
```

```
Keys of iris_dataset:
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
```

<font size=4 face="楷体">DESCR键对应的值是数据集的简要说明.</font>

<font size=4 face="楷体">targets_names键对应的值是字符串数组，包含预测花的品种:</font>

```python
print('Target names:{}'.format(iris_dataset['target_names']))
```

```
Target names: ['setosa' 'versicolor' 'virginica']
```

<font size=4 face="楷体">feature_names键对应值是字符串列表，对每个特征进行说明:</font>

```python
print(iris_dataset['feature_names'])
```

```
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```

<font size=4 face="楷体">数据包含在target和data字段中，data是测量数据，格式为Numpy数组:</font>

```python
print(type(iris_dataset['data']))
```

```
<class 'numpy.ndarray'>
```

<font size=4 face="楷体">data数组每行对应一朵花，列代表每朵花四个测量数据:</font>

```python
iris_dataset['data'].shape
```

```
(150, 4)
```

<font size=4 face="楷体">数组中共包含150朵不同的花测量数据.</font>

<font size=4 face="楷体">查看前5朵花数据：</font>

```python
print(iris_dataset['data'][:5])
```

```
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
```

<font size=4 face="楷体">target数组包含每朵花测量的品种，是一维Numpy数组,每朵花对应其中一个数据：</font>

```python
print(iris_dataset['target'].shape)
```

```
(150,)
```

```python
print(iris_dataset['target'])
```

```
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
```

<font size=4 face="楷体">品种已转换成0~2整数.</font>

<font size=5 face="楷体">训练数据与测试数据</font>

<font size=4 face="楷体">将收集好的带标签数据按比例分成两部分:**训练数据**（构建机器学习模型）、**测试数据**（评估模型性能）</font>

<font size=4 face="楷体">由于原始数据点是按标签排序的，如果只取数据后一部分数据作测试，无法评估模型好坏，因此要将数据打乱，确保测试数据包含所有类别的数据.</font>

<font size=4 face="楷体">这里用到了scikit-learn库中的train_test_split函数，将数据集打乱拆分，将75%数据作为训练集，其余25%数据为测试集.</font>

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0)
```

<font size=4 face="楷体">这里利用random_state参数指定随机数升成器种子，确保每次运行函数输出固定不变.</font>

<font size=4 face="楷体">查看输出数据的形状</font>

```python
print('X_train shape:{}'.format(X_train.shape))
print('y_train shape:{}'.format(y_train.shape))
```

```
X_train shape:(112, 4)
y_train shape:(112,)
```

```python
print(X_test.shape)
print(y_test.shape)
```

```
(38, 4)
(38,)
```

<font size=5 face="楷体">构建模型:k近邻算法</font>

<font size=4 face="楷体">含义：考虑训练集中与新数据最近的任意k个邻居，而不是只考虑最近的一个.用这些邻居中数量最多的类别作出预测.
</font>

<font size=4 face="楷体">现在只考虑1个邻居的情况.</font>

<font size=4 face="楷体">scikit-learn中所有机器学习模型都是在各自类中实现的，k近邻算法在neighbors模块的KNeighborsClassifiter类中实现.
    将类实化为对象需要设置模型参数.
</font>

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
```

<font size=4 face="楷体">knn对象对算法进行了封装，既包括用训练数据构建模型的算法，也包括对新数据点进行预测的算法，还包括算法从训练数据提取的信息.</font>

```python
#基于训练集构建模型，需要调用knn对象的fit方法
knn.fit(X_train,y_train)
```

<font size=4 face="楷体">运行结果：</font>

```
KNeighborsClassifier(n_neighbors=1)
```

<font size=5 face="楷体">作出预测</font>

<font size=4 face="楷体">对一朵新的已知测量数据的鸢尾花进行预测:</font>

```
x_new=np.array([[5,2.9,1,0.2]])
```

<font size=4 face="楷体">调用knn对象predict方法进行预测：</font>

```python
prediction=knn.predict(x_new)
print('Prediction:{}'.format(prediction))
print('Predicted target name:{}'.format(iris_dataset['target_names'][prediction]))
```

<font size=4 face="楷体">预测结果：</font>

```
Prediction:[0]
Predicted target name:['setosa']
```

<font size=5 face="楷体">评估模型</font>

<font size=4 face="楷体">对测试数据进行预测并与实际标签对比.</font>

```python
y_pred=knn.predict(X_test)
print(y_pred)
```

<font size=4 face="楷体">运行结果:</font>

```
[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
 2]
```

<font size=4 face="楷体">使用knn对象的score对象计算测试集精度：</font>

```
score=knn.score(X_test,y_test)
print(score)
```

<font size=4 face="楷体">运行结果</font>

```
0.9736842105263158
```

<font size=4 face="楷体">这个模型测试集的精度约为0.97，就是说对于测试集中鸢尾花的预测有97%是正确的.高精度意味着模型可信.</font>

<font size=5 face="楷体">汇总</font>

<font size=4 face="楷体">整个训练和评估过程所必需的代码：</font>

```
from sklearn.datasets import load_iris
iris_dataset=load_iris()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
score=knn.score(X_test,y_test)
print("Test set score:{}".format(score))
```

