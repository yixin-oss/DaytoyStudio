---
title: 二叉堆的Python实现
tags:
- Python
- 二叉堆
- 完全二叉树
- 堆排序
categories: 数据结构与算法
mathjax: true
---

## 优先队列与二叉堆

**优先队列（Priority Queue）：**

- 出队仍遵循队列原则，即**队首出队**；
- 入队时需要考虑内部数据项之间由**优先级**决定的次序.

对于优先队列的实现，一个很自然的想法是借助于排序算法，但为了降低算法的复杂度，这里考虑用**二叉堆（Binary Heap）**实现，将入队和出队的复杂度保持在**O(logn)**.

- 二叉堆的逻辑结构像二叉树，用非嵌套列表实现；
- 最小的key排在队首，称为**最小堆（Min Heap）**.

<!--more-->

**ADT BinaryHeap操作定义：**

| BinaryHeap() |     创建一个空二叉堆对象     |
| :----------: | :--------------------------: |
|  insert(k)   |      将新key加入到堆中       |
|  findMin()   |    返回堆中最小项，仍保留    |
|   delMin()   | 返回堆中最小项，并从堆中删除 |
|  isEmpty()   |        返回堆是否为空        |
|    size()    |      返回堆中key的个数       |
| buildHeap()  |    从一个key列表创建新堆     |

为了保证复杂度在**对数水平**，必须采用二叉树结构，且要尽可能保证**平衡**(即根左右子树拥有相同数量的节点). 下面介绍“退而求其次”的**完全二叉树**.

## 完全二叉树

### 定义

叶节点最多只出现在最底层和次底层，且最底层的叶节点连续集中在最左边，每个内部节点都有两个子节点，**最多可有1个节点例外**.

### 实例

![完全二叉树](https://s2.loli.net/2022/05/23/t6nUxNdC1ZiBel9.jpg)

### 性质

如果节点下标为**p**，那么其左子节点下标为**2p**，右子节点下标为**2p+1**，父节点下标为**p//2**.

![完全二叉树性质](https://s2.loli.net/2022/05/23/Tb2BXpWucC89H5o.png)

根据该性质可以快速确定其节点的位置.

## 堆次序Heap Order

- 任何一个节点x，其父节点p中的key均**小于**x中的key；
- 符合“堆”性质的二叉树，其中任何一条路径，均是一个**已排序**数列，且**根节点的key最小**.

![堆次序](https://s2.loli.net/2022/05/23/qylxCEeGDkPafTn.png)

## 二叉堆的Python实现

### 二叉堆初始化

采用一个列表保存堆数据，但要注意根节点下标从1开始.

```python
class BinHeap:
    def __init__(self):
        self.heaplist = [0]
        self.currentsize = 0
```

### insert(key)方法

需将新key沿着路径**上浮**到正确位置，但并不影响其他路径节点的**堆次序**.

```python
def percUp(self, i):
    while i // 2 > 0:
        if self.heaplist[i] < self.heaplist[i // 2]:
            tmp = self.heaplist[i // 2]
            self.heaplist[i // 2] = self.heaplist[i]  # 与父节点交换
            self.heaplist[i] = tmp
        i = i // 2  # 沿路径向上

def insert(self, k):
    self.heaplist.append(k)  # 添加到末尾
    self.currentsize += 1
    self.percUp(self.currentsize)  # 新key上浮
```

### delMin()方法

移走整个堆中最小的key（即根节点heaplist[1]），为保持完全二叉树性质，只能将最后一个节点代替根节点，但这样显然会破坏**堆次序**.

**策略：** 新的根节点沿一条路径**下沉**，直到比两个子节点都小，如果比子节点大，选择较小的子节点交换下沉.

```python
def percDown(self, i):
    while (i * 2) <= self.currentsize:
        mc = self.minChild(i)
        if self.heaplist[i] > self.heaplist[mc]:
            tmp = self.heaplist[i]
            self.heaplist[i] = self.heaplist[mc]  # 交换下沉
            self.heaplist[mc] = tmp
        i = mc  # 沿路径向下

def minChild(self, i):
    if 2 * i + 1 > self.currentsize:
        return 2 * i  # 唯一子节点
    else:
        if self.heaplist[i * 2] < self.heaplist[2 * i + 1]:
            return 2 * i
        else:
            return 2 * i + 1  # 返回较小的

def delMin(self):
    retval = self.heaplist[1]  # 移走堆顶
    self.heaplist[1] = self.heaplist[self.currentsize]
    self.currentsize -= 1
    self.heaplist.pop()
    self.percDown(1)  # 新顶下沉
    return retval
```

### buildHeap(list)方法

- 从无序表生成“堆”；
- 用“下沉”法将总代价控制为**O(n)**；
- 因叶节点无需下沉，故从最后节点的父节点开始.

```python
def buildHeap(self, alist):
    i = len(alist) // 2
    self.currentsize = len(alist)
    self.heaplist = [0] + alist[:]
    print(len(self.heaplist, i))
    while i > 0:
        print(self.heaplist, i)
        self.percDown(i)
        i = i - 1
    print(self.heaplist, i)
```

## 堆排序

### 堆的定义

- 大顶堆：每个节点的值都大于或等于其左右子节点的值；
- 小顶堆：每个节点的值都小于或等于其左右子节点的值.

堆排序是利用完全二叉树特性的一种选择排序. 虽然堆中记录无序，但在小顶堆中堆顶的key值最小，在大顶堆中堆顶的key值最大. 因此，堆排序是首先按key值大小排成堆，将堆顶元素与末尾元素交换位置，再将前（n-1）个元素排成堆，将堆顶元素与倒数第二交换，以此类推，即可得到按key值排序的有序序列.

### 节点下标关系

**注**：这里没有在列表前端加0.

- 父节点找左子节点的下标：i[左子] = 2i[父] + 1
- 父节点找右子节点的下标：i[右子] = 2i[父] + 2
- 子节点找父节点：i[父] = (i[子] - 1) // 2
- 用列表来储存，顺序从上到下，从左到右

### 向下调整

在堆排序过程中，当堆顶元素和末尾元素交换位置后，根节点和其子节点的key不满足堆的定义，需进行调整变堆.

**注：**这里的代码用的是**小顶堆降序**处理.

```python
def shit(li, first, finaly):
    '''

    :param li: 列表
    :param first: 指向根节点的下标
    :param finaly: 指向最后一个节点的下标
    :return:
    '''
    i = first  # i指向根节点
    j = 2 * i + 1  # j指向左节点
    count = li[first]  # count储存待调整的值
    while j <= finaly:  # 当j大于最后一个元素下标时，退出循环
        if j + 1 <= finaly and li[j + 1] < li[j]:  # 比较左右节点的大小
            j = j + 1  # j指向右节点
        if li[j] < count:  # 比较子节点是否大于根节点
            li[i] = li[j]  
            i = j
            j = 2 * i + 1  # j再次指向左子节点
        else:
            j = finaly + 1  # 退出循环

    li[i] = count  #如果j已经超过最后一个节点，那么count放到了叶节点

```

### 算法步骤

- 将待排序序列建成完全二叉树；
- 将完全二叉树建堆；
- 输出堆顶元素并用筛选法调整堆，直到二叉树只剩下一个节点.

```python
def heap_sort(li):
    # 先建堆
    for i in range(((len(li) - 1) - 1) // 2, -1, -1):
        shit(li, i, len(li) - 1)
    # 开始排序
    for i in range(len(li) - 1, -1, -1):  # i表示堆最后一个元素
        li[i], li[0] = li[0], li[i]  #最后一个元素和第一个交换位置
        shit(li, 0, i - 1)  # i-1表示新的finaly
```

### 实例

```python
list = [9,4,5,3,8,1]
heap_sort(list)
print(list)
```

```
[9, 8, 5, 4, 3, 1]
```

### 算法性能分析

- 时间复杂度

假设在堆排序过程中产生的二叉树高度为k，则$k=[log_2n]+1$，一次筛选过程中key的比较次数最多$2(k-1)$次，交换次数最多为$k$次，所以总的比较次数不超过
$$
2([log_2(n-1)]+[log_2(n-2)]+...+log_22)<2nlog_2n
$$
建初始堆的比较次数不超过$4n$次，故在最坏情况下堆排序算法时间复杂度$O(nlogn)$.

- 空间复杂度O(1)
- 不稳定的排序算法

