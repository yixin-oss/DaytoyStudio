---
title: Latex绘制流程图
tags: 
- Latex
- 流程图
categories: Latex写作
---

## 实现效果

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211205192409508.png" alt="甲方乙方流程图"  />

<!--more-->

## Codes

```latex
\documentclass[UTF8,a4paper]{ctexart}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows}

\begin{document}
\pagestyle{empty} % 无页眉页脚

\tikzstyle{startstop} = [rectangle,rounded corners, minimum width=3cm,minimum height=1cm,text centered, draw=black,fill=red!30]
\tikzstyle{io} = [trapezium, trapezium left angle = 70,trapezium right angle=110,minimum width=3cm,minimum height=1cm,text centered,draw=black,fill=blue!30]
\tikzstyle{process} = [rectangle,minimum width=3cm,minimum height=1cm,text centered,text width =3cm,draw=black,fill=orange!30]
\tikzstyle{decision} = [diamond,minimum width=3cm,minimum height=1cm,text centered,draw=black,fill=green!30]
\tikzstyle{arrow} = [thick,->,>=stealth]

\begin{tikzpicture}[node distance=2cm]
\node (start) [startstop] {生活所迫};
\node (input1) [io,below of=start] {卑微乙方出方案};
\node (process1) [process,below of=input1] {修改方案};
\node (decision1) [decision,below of=process1,yshift=-0.5cm] {甲方爸爸};
\node (process2a) [process,below of=decision1,yshift=-0.5cm] {删除所有修改};
\node (process2b) [process,right of =decision1,xshift=2cm] {通宵修改方案};
\node (out1) [io,below of=process2a] {最初的方案};
\node (stop) [startstop,below of=out1] {卒};

\draw [arrow] (start) -- (input1);
\draw [arrow] (input1) -- (process1);
\draw [arrow] (process1) -- (decision1);
\draw [arrow] (decision1) -- node[anchor=east] {yes} (process2a);
\draw [arrow] (decision1) -- node[anchor=south] {no} (process2b);
\draw [arrow] (process2b) |- (process1);
\draw [arrow] (process2a) -- (out1);
\draw [arrow] (out1) -- (stop);
\end{tikzpicture}

\end{document} 
```

## 代码解析

### Using Package

```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric,arrows}
```

### tikzstyle定义node和箭头属性

#### 节点

```latex
\tikzstyle{process} = [rectangle,minimum width=3cm,minimum height=1cm,text centered,text width =3cm,draw=black,fill=orange!30]
```

```latex
# 节点形状
rectangle:矩形，可加圆角(rounded corners)
trapezium:平行四边形
diamond:菱形
# 尺寸
minimum width
minimum height
# 文本
text centered:文本居中
# 文本宽度
text width=3cm:文本超过3cm时会自动换行
# 边框
draw
# 填充颜色
fill
```

#### 箭头

```latex
\tikzstyle{arrow} = [thick,->,>=stealth]
```

```latex
# 线粗：
thick:粗
thin:细
# 箭头
->:反向箭头
<-:正向箭头
<->:双向箭头
# 虚线
dashed
# 箭头形状
>=stealth
```

### 创建节点

```latex
\node (process1) [process,below of=input1] {修改方案};
```

```latex
# name
(process1):这个节点的name，后面需要用这个name调用这个节点。
# 属性
decision：需要调用的节点的属性
# 位置
below of=input1：定义节点的位置
left of:
right of:
# 偏移,对位置进行微调
yshift:
xshift:
# title
{修改方案}:结果显示的标题
```

### 画箭头

```latex
\draw [arrow] (decision1) -- node[anchor=east] {yes} (process2a);
\draw [arrow] (decision1) -- node[anchor=south] {no} (process2b);
```

```latex
# 属性
[arrow]:需要调用的箭头的属性
(decision1)：箭头的初始位置
(process2a)：箭头的末尾位置
# 线型
--：直线
|-：先竖线后横线
-|：向横线后竖线
# 文字：如果需要在箭头上添加文字
{yes}:需要添加的文字
# 文字的位置,上南下北左东右西(与地图方位不一致)
[anchor=east]：
[anchor=south]：
[anchor=west]：
[anchor=north]：
[anchor=center]：
```

## Reference

[Latex画流程图](https://www.jianshu.com/p/2d01d5eaaa77)

[Creating Flowcharts with TikZ](https://link.jianshu.com/?t=https://www.sharelatex.com/blog/2013/08/29/tikz-series-pt3.html)