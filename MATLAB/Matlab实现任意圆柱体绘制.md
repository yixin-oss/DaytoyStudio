---
title: Matlab实现任意圆柱体绘制
tags:
- Matlab
- Cylinder
categories: 机械加工
---

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/喵.jpg" style="zoom: 33%;" />

<!--more-->

## 代码

```matlab
function [Cylinder EndPlate1 EndPlate2] = cylinder3(X1,X2,r,n,cyl_color,closed,lines)

%

% This function constructs a cylinder connecting two center points

%两个中心点连线的圆柱体

% Usage :

% [Cylinder EndPlate1 EndPlate2] = cylinder3(X1+20,X2,r,n,'r',closed,lines)

%

% Cylinder-------Handle of the cylinder

% EndPlate1------Handle of the Starting End plate

% EndPlate2------Handle of the Ending End plate

% X1 and X2 are the 3x1 vectors of the two points

% r is the radius of the cylinder

% n is the no. of elements on the cylinder circumference (more--> refined)

% cyl_color is the color definition like 'r','b',[0.52 0.52 0.52]

% closed=1 for closed cylinder or 0 for hollow open cylinder

% lines=1 for displaying the line segments on the cylinder 0 for only surface

%

% Typical Inputs

% X1=[10 10 10];

% X2=[35 20 40];

% r=1;

% n=20;

% cyl_color='b';

% closed=1;

%

%see more information please go to www.matlabsky.cn

% NOTE: There is a MATLAB function "cylinder" to revolve a curve about an axis. 

% This "Cylinder" provides more customization(定制) like direction and etc

% Calculating the length of the cylinder

length_cyl=norm(X2-X1);

% Creating a circle in the YZ plane

t=linspace(0,2*pi,n)';

x2=r*cos(t);

x3=r*sin(t);

% Creating the points in the X-Direction

x1=[0 length_cyl];

% Creating (Extruding) the cylinder points in the X-Directions

xx1=repmat(x1,length(x2),1);

xx2=repmat(x2,1,2);

xx3=repmat(x3,1,2);

% Drawing two filled cirlces to close the cylinder

if closed==1

hold on

EndPlate1=fill3(xx1(:,1),xx2(:,1),xx3(:,1),'r');

EndPlate2=fill3(xx1(:,2),xx2(:,2),xx3(:,2),'r');

end

% Plotting the cylinder along the X-Direction with required length starting from Origin

%从坐标原点沿x轴绘制给定长度的圆柱

Cylinder=mesh(xx1,xx2,xx3);

% Defining Unit vector along the X-direction

unit_Vx=[1 0 0];

% Calulating the angle between the x direction and the required direction of cylinder through dot product

%通过向量内积计算x轴与圆柱所需方向的夹角

angle_X1X2=acos( dot( unit_Vx,(X2-X1) )/( norm(unit_Vx)*norm(X2-X1)) )*180/pi;

% Finding the axis of rotation (single rotation) to roate the cylinder in X-direction to the required arbitrary direction through cross product

%通过向量外积找到旋转轴，x轴绕旋转轴旋转angle_X1X2度到所需的任意方向

axis_rot=cross([1 0 0],(X2-X1) );

% Rotating the plotted cylinder and the end plate circles to the required angles

%将画好的圆柱和两端圆旋转到所需方向

if angle_X1X2~=0 % Rotation is not needed if required direction is along X

rotate(Cylinder,axis_rot,angle_X1X2,[0 0 0])

if closed==1

rotate(EndPlate1,axis_rot,angle_X1X2,[0 0 0])

rotate(EndPlate2,axis_rot,angle_X1X2,[0 0 0])

end

end

% Till now cylinder has only been aligned with the required direction, but position starts from the origin. 

%so it will now be shifted to the right position

%将确定方向的圆柱平移到正确的位置

if closed==1

set(EndPlate1,'XData',get(EndPlate1,'XData')+X1(1))

set(EndPlate1,'YData',get(EndPlate1,'YData')+X1(2))

set(EndPlate1,'ZData',get(EndPlate1,'ZData')+X1(3))

set(EndPlate2,'XData',get(EndPlate2,'XData')+X1(1))

set(EndPlate2,'YData',get(EndPlate2,'YData')+X1(2))

set(EndPlate2,'ZData',get(EndPlate2,'ZData')+X1(3))

end

set(Cylinder,'XData',get(Cylinder,'XData')+X1(1))

set(Cylinder,'YData',get(Cylinder,'YData')+X1(2))

set(Cylinder,'ZData',get(Cylinder,'ZData')+X1(3))

% Setting the color to the cylinder and the end plates

%设置圆柱和两端圆的颜色

set(Cylinder,'FaceColor',cyl_color)

if closed==1

set([EndPlate1 EndPlate2],'FaceColor',cyl_color)

else

EndPlate1=[];

EndPlate2=[];

end

% If lines are not needed making it disapear

if lines==0

set(Cylinder,'EdgeAlpha',0)

end
axis off
```

## 示例

圆柱两个中心点坐标取为$[10\quad 10\quad 10],\quad [35 \quad 20\quad  40]$，颜色设置为蓝色.

```matlab
[Cylinder EndPlate1 EndPlate2] = cylinder3([10 10 10],[35 20 40],1,20,'b',1,0)
```

![Cylinder](https://gitee.com/yixin-oss/blogImage/raw/master/img/cylinder.jpg)

## Reference

[Matlab画圆柱体](https://blog.csdn.net/weixin_39575212/article/details/116089163)