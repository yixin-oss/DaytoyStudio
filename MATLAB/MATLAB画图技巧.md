- 取消$surf(x,y,z)$中曲面上的网格线

```matlab
shading interp
```

- 设置曲面的颜色

```matlab
map=[0 0.5 0]
colormap(map) % 绿色
```

- 设置$figure$图片的背景色

```matlab
backcolor=[0.94 0.99 0.94]; % 蜜露橙
set(gca,'Color',backcolor)
```

[更多颜色RGB](https://blog.csdn.net/qq_42537872/article/details/127960146)

- 设置坐标区的颜色条(色阶)

```matlab
colorbar('vert') % 竖条
% colorbar('hori') 横条
```

- 设置$quiver$函数的箭头长度及颜色

```matlab
quiver(...,scale,'b') % 根据因子scale拉伸
```

- 设置坐标轴的粗细

```matlab
set(gca,'Linewidth',1.2)
```

- 画水平线

```matlab
x=get(gca,'xlim');
y=0.1;
plot(x,[y y],'b-')
```

- 画竖直线同理

```matlab
y=get(gca,'ylim');
x=0.1;
plot([x x],y,'b-')
```

- 颜色设计

```matlab
colormap('jet') %冷暖调分明的鲜艳色
```

- 坐标轴调整

```matlab
axis tight % 调整坐标和输入数据范围一致
```

- 思维绘图函数

```matlab
surf(X,Y,Z,F) % 在三维绘图指令中加入新的参量就变成思维绘图函数了
```

- MATLAB作图淡色系

```matlab
[1 0.58 0.8] % 浅红色

[0.58 0.8 1] % 淡蓝色

[0.8 1 0.58] % 浅绿色

[1 0.8 0.58] % 浅橙色

[0.53 0.15 0.34] % 草莓色

[0.8 0.58 1] % 浅紫色

[0.58 1 0.8] % 墨绿色
```

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/color.png)