---
title: python turtle库"一箭穿心"代码实现
tags: python
categories: python学习笔记
---

利用python中的turtle库实现“一箭穿心”简单绘图

<!--more-->

### 代码实现

源代码如下：

```
import turtle as t
t.color('red','pink')
t.hideturtle()#隐藏画笔形状
t.speed('fast')
t.begin_fill()
t.width(4)
t.left(135)
t.circle(50,180)
t.fd(100)
t.left(90)
t.fd(100)
t.circle(50,180)
t.pu()
t.goto(50,-30)
t.left(270)
t.pd()
t.circle(50,180)
t.fd(100)
t.left(90)
t.fd(100)
t.circle(50,180)
t.end_fill()
t.color('black')
t.pu()
t.goto(230,-100)
t.pd()
#尾1.1
t.left(90)
t.fd(40)
t.fd(-40)
#尾1.2
t.left(80)
t.fd(40)
t.fd(-40)
#尾间
t.left(135)
t.fd(40)
#尾2.1
t.left(135)
t.fd(40)
t.fd(-40)
#尾2.2
t.left(90)
t.fd(40)
t.fd(-40)
#箭身
t.left(135)
t.fd(145)
t.pu()
t.fd(135)
t.pd()
t.fd(100)
#箭矢
t.left(30)
t.fd(40)
t.right(60)
t.fd(40)
t.right(120)
t.fd(40)
t.right(60)
t.fd(40)
t.done()
```

整体的思路就是运用了turtle库中的各种指令，下面附注一些对turtle库的几处简要说明，代码比较容易，可以尝试编写哦.

### 指令说明

|              指令               |                             说明                             |
| :-----------------------------: | :----------------------------------------------------------: |
| turtle.color(’color1‘,‘color2’) |     color1画笔颜色,color2填充颜色,若无填充可省略'color2'     |
|       turtle.width(size)        |                      画笔宽度，参数size                      |
|       turtle.hideturtle()       |                     hide，即隐藏画笔形状                     |
|           turtle.pu()           |         put up，抬起画笔，之后画笔行进轨迹将不再显示         |
|           turtle.pd()           |             put down，放下画笔，画笔轨迹继续显示             |
|          turtle.fd(d)           |                  forward，向正前方前进距离d                  |
|          turtle.bd(d)           |                 backward，向正后方移动距离d                  |
|       turtle.left(angle)        |               画笔前进方向逆时针转动角度angle                |
|       turtle.right(angle)       |               画笔前进方向顺时针转动角度angle                |
|        turtle.goto(x,y)         |        画笔移动到坐标位置(x,y)，注意移动轨迹也会显示         |
|     turtle.circle(r,angle)      | 以当前画笔方向左侧某处为圆心进行曲线运行，r为曲线半径，angle是曲线圆心角度数（注意运行后画笔方向随之改变） |
|       turtle.begin_fill()       |                           开始填充                           |
|        turtle.end_fill()        |                           结束填充                           |
|      turtle.speed(’speed‘)      |       画笔速度，'speed'可选择'fast','fastest','slow'等       |

### 坐标体系

#### 1.绝对坐标

<img src="https://raw.githubusercontent.com/yixin-oss/Image/master/imgxy.png"/>

绝对坐标其实就是指平面直角坐标系，利用坐标表示平面中点的位置，与指令turtle.goto(x,y)对应，可以控制画笔移动到某一坐标位置，由于移动轨迹会显示，故可搭配turtle.pu()，turtle.pd()一起使用.

#### 2.海龟坐标

<img src="https://raw.githubusercontent.com/yixin-oss/Image/master/imgturtle%E5%9D%90%E6%A0%87.png"/>

海龟坐标就是站在海龟的角度考虑方向问题，海龟可不认识坐标点哦，只需要考虑前进(forward)还是后退(backward)，向左(left)还是向右(right)，注意水平向右是海龟的默认初始方向，left和right指以当前海龟的方向向左转或者向右转，如果海龟方向比较奇怪的时候可能分不清左右(我个人如此)，故建议用顺(right)逆(left)时针来记(可看表的指针).

### RGB色彩体系

提到色彩最重要的当然就是画笔颜色和填充颜色，白+彩虹七色就不用多说了，下面介绍几种特殊颜色

|   词汇   |     颜色     |
| :------: | :----------: |
|   pink   |     粉色     |
| magenta  |     洋红     |
|   cyan   |     青色     |
| seashell | 海贝色(很浅) |
|   gold   |     金色     |
|  brown   |     棕色     |
|  tomato  |    番茄色    |

当然还可以根据红蓝绿三个通道颜色组合设计自己想要的颜色，每种颜色的取值默认为小数值，举例用法：turtle.pencolor((0.63,0.12,0.94)) (紫色).

### 赞助

如果对本文有好感，可以点击下方打赏赞助我买包辣条嘛，一两块钱就可以哦，咦嘻嘻~