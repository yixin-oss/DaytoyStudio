---
title: wordcloud库
tags: python
categories: python学习

---

## 一、wordcloud库基本介绍

优秀的词云展示第三方库.

词云以词语为单位，更直观和艺术的展示文字.

### 1.安装

```
pip install wordcloud
```

<!--more-->

### 2.wordcloud库常规方法

**w=wordcloud.WordCloud()**生成一个词云对象.

- wordcloud库把词云当作一个WordCloud对象.
- wordcloud.WordCloud()代表一个文本对应词云.
- 可以根据文本中词云出现的频率等参数绘制词云.
- 绘制词云的形状、尺寸和颜色都可以设定.

|        方法         |                  描述                  |
| :-----------------: | :------------------------------------: |
|   w.generate(txt)   |     向WordCloud对象w中加载文本txt      |
| w.to_file(filename) | 将词云输出为图像文件，.png or .jpg格式 |

### 3.基本步骤

1. 配置对象参数
2. 加载词云文本
3. 输出词云文件

**示例**

```
import wordcloud
c = wordcloud.WordCloud()
c.generate('wordcloud by Python')
c.to_file('pywordcloud.png')
```

**四步：**

1. 以空格为分隔符号，将文本分隔成单词.
2. 统计单词出现次数并过滤短单词.
3. 根据统计配置字号.
4. 布局：颜色环境尺寸.

**配置对象参数**

**w = wordcloud.WoedCloud(<参数>)**

|      参数       |                            描述                            |
| :-------------: | :--------------------------------------------------------: |
|      width      |          指定词云对象生成图片的宽度，默认400像素           |
|     height      |          指定词云对象生成图片的高度，默认200像素           |
|  min_font_size  |             指定词云中字体的最小字号，默认4号              |
|  max_font_size  |         指定词云中字体的最大字号，根据高度自动调节         |
|    font_step    |            指定词云中字体字号的步进间隔，默认1             |
|    font_path    |   指定字体文件的路径，默认None   e.g."msyh.ttc"微软雅黑    |
|    max_words    |            指定词云显示的最大单词数量，默认200             |
|   stop_words    |      指定词云排除词列表   e.g. stop_words={'Python'}       |
|      mask       |      指定词云形状，默认为长方形，需要引用imread()函数      |
| backgroud_color | 指定词云图片背景颜色，默认黑色 e.g.backgroud_color='white' |

```python
from scipy.misc import imread
mk = imread('pic.png')
w = wordcloud.WordCloud(mask=mk)
```

### 4.应用实例

**英文**

```
import wordcloud
txt = 'life is short, you need python'
w = wordcloud.WordCloud( \
        backgroud_color = 'white')#生成一个词云对象，背景设为白色
w.generate(txt)
w.to_file("pywordcloud.png")
```

**中文**

```python
#需要对中文先进行分词，引入jieba库
import jieba
import wordcloud
txt = "程序设计语言是计算机能够理解和\
识别用户操作意图的一种交互体系，它按照\
特定规则组织计算机指令，使计算机能够自\
动进行各种运算处理。"
w = wordcloud.WordCloud( width=1000,\
    font_path="msyh.ttc",height=700)
w.genrtate(''.join(jieba.lcut(txt)))
w.to_file("pythoncloud.png")
```

**Remark.** 中文需要先分词并组成空格分隔字符串.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/pythoncloud.png" style="zoom: 50%;" />

## 二、实例：政府工作报告词云

### 1.问题分析

- 需求：对于政府工作报告等政策文件，如何直观理解？
- 体会直观价值：生成词云&优化词云
- 两份重要文件
  - [习总书记在中国共产党第十九次全国代表大会上的报告](https://python123.io/resources/pye/新时代中国特色社会主义.txt)
  - [中共中央 国务院关于实施乡村振兴战略的意见](https://python123.io/resources/pye/关于实施乡村振兴战略的意见.txt)

**政府报告等文件—>>有效展示的词云**

### 2.基本思路

- 读取文件、分词整理
- 设置并输入词云
- 观察结果，优化迭代

### 3.代码展示

```python
import jieba
import wordcloud
f = open("新时代中国特色社会主义.txt", 'r', encoding='utf-8')
t = f.read()
f.close()
ls =jieba.lcut(t)
for i in range(len(ls)-1,-1,-1):
    if len(ls[i])==1: #移除单字
        ls.remove(ls[i])
txt = " ".join(ls)
w = wordcloud.WordCloud(font_path="msyh.ttc",width=1000, height=700, background_color="white")
w.generate(txt)
w.to_file("新时代中国特色社会主义.png")
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/新时代中国特色社会主义.png" style="zoom: 50%;" />

```python
import jieba
import wordcloud
f = open("乡村振兴战略意见.txt", 'r', encoding='utf-8')
t = f.read()
f.close()
ls =jieba.lcut(t)
for i in range(len(ls)-1,-1,-1):
    if len(ls[i])==1:
        ls.remove(ls[i])
txt = " ".join(ls)
w = wordcloud.WordCloud(font_path="msyh.ttc",width=1000, height=700, background_color="white")
w.generate(txt)
w.to_file("乡村振兴战略意见.png")
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/乡村振兴战略意见.png" style="zoom:50%;" />

### 4.优化

#### （1）减少词云数目

```
w = wordcloud.WordCloud(font_path="msyh.ttc",width=1000, height=700, background_color="white",max_words=15)
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/乡村振兴战略意见15词.png" style="zoom:50%;" />

#### （2）改变词云背景

```
#金色背景
w = wordcloud.WordCloud(font_path="msyh.ttc",width=1000, height=700, background_color="gold",max_words=15)
```

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/乡村振兴战略意见gold背景.png" style="zoom:50%;" />

