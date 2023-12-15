---
title: Typora配置图片自动上传到图床
tags: 
- Typora
- Gitee
- PicGo
categories: hexo博客搭建
---

平常写博客或者做笔记时，都有插入图片的时候，由于Typora图片只能保存在本地，变动一下文件就会出现访问失效的问题，所以需要图床的存在. 图床，就是自动把本地图片转换为链接的工具，就目前的使用种类而言，PicGo是一款比较优秀的图床工具.

下边使用**PicGo+Gitee**实现markdown图床.

<!--more-->

## PicGo的设置1(Windows版本)

**插件设置**-->**搜索gitee**-->**安装插件**

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125095720589.png" style="zoom: 67%;" />

## Gitee码云图床

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125100454032.png" style="zoom: 67%;" />

6点击创建.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125100736637.png" style="zoom: 67%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125100845514.png"  style="zoom:67%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125100937929.png" alt="image-20211125100937929" style="zoom:67%;" />

**令牌码拷贝下来备用！**

## PicGo设置2

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125101218243.png"  style="zoom: 67%;" />

```markdown
owner: 码云用户名
repo: 仓库名称
path: img
token: 码云的私人令牌
message: 不写
```

## 配置Typora

点击**文件**-->**偏好设置**-->**图像**-->**设置PicGo文件路径**

- 设置成上传图片
- 勾选三个选项
- 上传服务：PicGo(app)
- PicGo路径选择为安装PicGo.exe的路径
- 验证图片上传选项：配置完成测试是否成功，去码云图床仓库查看是否有图片上传

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125101636090.png"  style="zoom:67%;" />

## 注意事项

Typora使用下面这个url和PicGo连接的，PicGo的设置也要对应，默认一般是
$$
using\quad http://127.0.0.1:36677/upload
$$
<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125102252660.png"  style="zoom:67%;" />

![](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20211125102311391.png)

不过PicGo的Sever监听端口会经常变动（比如电脑重启后），就需要修改不然Typora图片也会上传失败.