---
title: Dijkstra算法
tags:
- 赋权有向图
- 最短路径
categories: 图优化算法
mathjax: true
---

## 简介

**Dijkstra算法**, 是由[荷兰]计算机科学家Edsger Wybe Dijkstra在1956年发现的算法, 它使用类似广度优先搜索的方法解决**赋权图的单源最短路径问题**.  本算法从一个起点出发找到该点到图中其他顶点的最短路径, 产生一个最短路径树. 需要注意的是, Dijkstra算法不能处理带有负权边的图.

## 算法流程

**基本思路：**每次取出未访问顶点中距离最小的点, 用该顶点更新其他顶点的距离.

**输入：**赋权有向图$G=(V,E,W),$ 其中$V=\{v_1,...,v_n\},E=\{e_{i,j}:=\{v_i,v_j\}\},W=w_{i,j}(e_{i,j})$分别代表顶点, 边及权重.

**输出：**从起点$s$到$\forall v_i\in V-\{s\}$的最短路径.

1. 初始点集合$S=\{s\}$；

2. 计算初始点$s$到其余各点的直接距离$dist[s,v_i],\forall v_i\in V-S$;

3. 选出满足距离最小$min_{v_k\in V}dist[s,v_k]$的点$v_k$, 并将该点加入到集合$S$中, 即$S\cup{v_k}\rightarrow S$，更新$V-S$中各顶点的$dist$值, i.e.,

   如果$dist[s,v_k]+w_{k,i}<dist[s,v_i]$, 则$dist[s,v_i]=dist[s,v_k]+w_{k,i}, \forall v_i\in V-S$;

4. 返回步骤3, 直至$S=V$算法终止.

### 伪代码

| Algorithm: Dijkstra                                          |
| ------------------------------------------------------------ |
| **Input**: $Directed \quad graph\quad  G=(V,E,W) \quad with\quad  weight$<br/>**Output**: $All \quad the\quad shortest\quad paths\quad for\quad the\quad source\quad vertex\quad s \quad to\quad other\quad vertex$<br/>1: $S\leftarrow {s}$<br/>2: $dist[s,s]\leftarrow 0$ <br/>3: $for \quad v_i\in V-S \quad do$<br/>4: $dist[s,v_i]\leftarrow w(s,v_i)\quad (when \quad v_i \quad not \quad found,\quad dist[s,v_i]\leftarrow \infty)$<br/>5: $while \quad V-S\neq \varnothing\quad do$<br/>6: $find \quad min_{v_k\in V}dist[s,v_k]\quad from \quad the \quad set \quad V-S$<br/>7: $S\leftarrow S\cup\{v_k\}$<br/>8: $for \quad v_k \in V-S \quad do$<br/>9: $if \quad dist[s,v_k]+w_{k,i}< dist[s,v_i]\quad then$<br/>10: $dist[s,v_i]\leftarrow dist[s,v_k]+w_{k,i}$ |

## 实例计算

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/image-20230404171442386.png" alt="image-20230404171442386" style="zoom:67%;" />

​		选择$v_1$为起点开启算法的整个流程, $S={v_1}$, 则接下来要计算从$v_1$出发, 到达其余各点的直接距离$dist[v_2,v_1]\sim dist[v_6,v_1]$, 注意这里所说的直接距离是指只通过$v_1$到达其他各点路径的长度, 这是第一个for循环在做的事情. 显然, 只通过顶点$v_1$是无法到达顶点$v_3,v_4,v_5$的, 因此我们有
$$
dist[v_1,v_2]=w_{1,2}=10,\quad dist[v_1,v_3]=\infty,\\
dist[v_1,v_4]=\infty, \quad dist[v_1,v_5]=\infty,\\
dist[v_1,v_6]=w_{1,6}=3, \quad dist[v_1,v_1]=0.
$$
到这里, 伪代码的第4步结束, 现在开始第5步检查集合$V-S=\varnothing$是否成立, 显然不是，因为
$$
V-S=\{v_2,v_3,v_4,v_5,v_6\}.
$$
则进行第6步, 选出经过第一个for循环之后, 集合$V-S$中相对于其他点而言到$v_1$距离最短的顶点$v_j$， 显然是$v_6$. 则执行第7步, 
$$
S=S\cup\{v_6\}=\{v_1,v_6\},
$$
说明从起点$v_1$出发到顶点$v_6$的最短路径已找到.  由于集合$S$新加入点, 计算路径时允许通过新的点, 这样就可能会改变原有的路径长度, 因此要执行8-10步对距离进行更新, 以$dist[v_1,v_2]$的更新为例, 原来从$v_1$到$v_2$只能走直接的距离$dist[v_1,v_2]=w_{1,2}=10$， 而现在将$v_6$加入到集合$S$后, 我们多了一条路线$v_1\rightarrow v_6 \rightarrow v_2$, 两条路线都可以的话我们需要选择距离相对短的走法, 而
$$
dist[v_1,v_2]=10 > dist[v_1,v_6]+w_{6,2}=5,
$$
所以选择后者. 其余变化的顶点的分析是类似的, 最终我们得到
$$
dist[v_1,v_2]=5,\quad dist[v_1,v_3]=\infty,\\
dist[v_1,v_4]=dist[v_1,v_6]+w_{6,4}=9, \quad dist[v_1,v_5]=dist[v_1,v_6]+w_{6,5}=4,\\
dist[v_1,v_6]=w_{1,6}=3,\quad dist[v_1,v_1]=0.
$$
现在，算法已经从头到尾执行了一遍, 然后需要回到第5步判断while循环条件确定算法是否继续, 此时
$$
V-S=\{v_2,v_3,v_4,v_5\}\neq \varnothing,
$$
所以算法继续执行, 但计算过程与上述分析仍然是类似的, 这里不再赘述, 我们给出最终的结果:
$$
dist[v_1,v_2]=dist[v_1,v_6]=5,\quad dist[v_1,v_3]=dist[v_1,v_6]+w_{6,2}+w_{2,3}=12,\\
dist[v_1,v_4]=dist[v_1,v_6]+w_{6,4}=9, \quad dist[v_1,v_5]=dist[v_1,v_6]+w_{6,5}=4,\\
dist[v_1,v_6]=w_{1,6}=3,\quad dist[v_1,v_1]=0.\\
S=\{v_1,v_2,v_3,v_4,v_5,v_6\}
$$
此时$V=S$, 算法结束，我们得到了起点$v_1$到其余各点的最短距离.

## Code

```matlab
function [minimum,path] = Dijkstra(w,start,terminal)

% Dijkstra算法
% 计算起始点到终点的最短路径
% w是所求图的带权连接矩阵
% start, terminal分别是路径的起点和终点

n=size(w,1);
length_node=ones(1,n)*inf;
length_node(start)=0;
father_node(start)=start;
set(1)=start;
Now_best=start;
% n为所求图的顶点个数, length_node存放到各点的最短路径，father_node表示父亲顶点用于还原路径
% 初始化时将除start以外的顶点的length_node设置为无穷大
% 数组set存放已经搜好的顶点, 初始化时只有start
while length(set)<n
    Next_best=0;
    k=inf;
    % 遍历所有顶点, 将不再集合set中的顶点选出来
    for i=1:n
        flag=0;
        for j=1:length(set)
            if i==set(j)
                flag=1;
            end
        end
        % 判断是否有中继节点使得它们之间的距离更短, 如果有, 更新距离及顶点
        if flag==0
            if length_node(i)>(length_node(Now_best)+w(Now_best,i))
                length_node(i)=length_node(Now_best)+w(Now_best,i);
                father_node(i)=Now_best;
            end
            % 找到目前路径最短的顶点放入顶点集
            if k>length_node(i)
                k=length_node(i);
                Next_best=i;
            end
        end
    end
    set=[set Next_best];
    Now_best=Next_best;
end

minimum=length_node(terminal);
path(1)=terminal;
i=1;
% 按倒序结果推出最短路径
while path(i)~=start
    path(i+1)=father_node(path(i));
    i=i+1;
end
path(i)=start;
% 翻转得到最短路径
path=path(end:-1:1);
end
```

```matlab
% Dijkstra_test
w=[0,7,9,inf,inf,14;
   7,0,10,15,inf,inf;
   9,10,0,11,inf,2;
   inf,15,11,0,6,inf;
   inf,inf,inf,6,0,9;
   14,inf,2,inf,9,0];
start=1;terminal=5;
[minimum,path]=Dijkstra(w,start,terminal);
disp(minimum)
disp(path)
```

```matlab
20

     1     3     6     5
```

## Appendix

```matlab
%% matlab做有权无向图

% 函数graph(s,t,w): 可在s和t中的对应节点之间以w的权重创建边, 并生成一个图
s=[1,2,3,4];
t=[2,3,1,1];
w=[3,8,9,2];
G=graph(s,t,w);
figure;
plot(G,'EdgeLabel',G.Edges.Weight,'linewidth',2)
set(gca,'XTick',[],'YTick',[]);

%% 查找有权图的最短路径

s=[9 9 1 1 2 2 2 7 7 6 6 5 5 4];
t=[1 7 7 2 8 3 5 8 6 8 5 3 4 3];
w=[4 8 3 8 2 7 4 1 6 6 2 14 10 9];
G=graph(s,t,w);
figure;
plot(G,'EdgeLabel',G.Edges.Weight,'linewidth',2)
set(gca,'Xtick',[],'YTick',[]);
[P,d]=shortestpath(G,9,4); % P里面存储的是最短路径上面的节点

% 在图中高亮我们的最短路径
myplot=plot(G,'EdgeLabel',G.Edges.Weight,'linewidth',2); %首先将图赋给一个变量
highlight(myplot,P,'EdgeColor','r');

%求出任意两点的最短路径矩阵
D=distances(G)
D(1,2) % 1 -> 2的最短路径
D(9,4) % 9 -> 4的最短路径

% 找出给定范围内的所有点 nearest(G,s,d)
% 返回图形G中与节点s的距离在d之内的所有节点
[nodeIds,dist]=nearest(G,2,10)
```

```matlab
D =

     0     6    13    20    10     9     3     4     4
     6     0     7    14     4     6     3     2    10
    13     7     0     9    11    13    10     9    17
    20    14     9     0    10    12    17    16    24
    10     4    11    10     0     2     7     6    14
     9     6    13    12     2     0     6     6    13
     3     3    10    17     7     6     0     1     7
     4     2     9    16     6     6     1     0     8
     4    10    17    24    14    13     7     8     0

ans =

     6

ans =

    24

nodeIds =

     8
     7
     5
     1
     6
     3
     9

dist =

     2
     3
     4
     6
     6
     7
    10
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/weight_graph.png" alt="weight_graph" style="zoom: 50%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/shortest_path.png" style="zoom: 50%;" />

## Reference

```
1.《算法》课程,屈婉玲教授.
2. 知乎专栏 https://zhuanlan.zhihu.com/p/129373740
3. CSDN博客 https://blog.csdn.net/weixin_46308081/article/details/119254473
4. bilibili视频 https://www.bilibili.com/video/BV1rS4y1B7Du/?spm_id_from=333.337.search-card.all.click&vd_source=51da3c4dcc63551dbc947486e34c89c4
```

















