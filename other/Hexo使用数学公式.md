---
title: Hexo使用数学公式
tags: hexo
categories: hexo博客搭建
mathjax: true
---

## 在Typora添加数学公式

Typora作为Markdown编辑器，在写技术文档时少不了输入各种Latex数学公式，下面介绍具体的设置、使用格式和常见的符号.

<!--more-->

1.进入**文件->偏好设置->Markdown**，将“Markdown扩展语法“全部勾选，然后重启Typora.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1624945882(1).jpg" style="zoom:80%;" />



2.使用格式

|              行间式               |                            行外式                            |
| :-------------------------------: | :----------------------------------------------------------: |
| 文字间插入公式，只需公式前后加上$ | 公式单独一行居中显示.右键插入公式块或换行输入$$再回车或快捷键**ctrl+shift+M** |

效果预览：

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1624945949(1).jpg"  />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/1624946022(1).jpg" style="zoom:80%;" />

3.常用符号及规则

- 上下标、矢量、分式、根号、累加累乘

|       数学表达式       |      Latex代码       |
| :--------------------: | :------------------: |
|         $x^2$          |         x^2          |
|         $x_1$          |         x_1          |
|        $\vec F$        |        \vec F        |
|     $\frac{1}{2}$      |     \frac{1}{2}      |
|       $\sqrt{2}$       |       \sqrt{2}       |
| $\sum_{i=1}^{n}a_{i}$  | \sum_{i=1}^{n}a_{i}  |
| $\prod_{i=1}^{n}a_{i}$ | \prod_{i=1}^{n}a_{i} |



- 极限、无穷

|          数学表达式          |         Latex代码          |
| :--------------------------: | :------------------------: |
| $\lim_{a\rightarrow+\infty}$ | \lim_{a\rightarrow+\infty} |
|           $\infty$           |           \infty           |

- 关系运算符

| 数学表达式 | Latex代码 |
| :--------: | :-------: |
|   $\leq$   |   \leq    |
|   $\neq$   |   \neq    |
|   $\geq$   |   \geq    |

- 希腊字母

|  数学表达式   |  Latex代码  |
| :-----------: | :---------: |
|   $\alpha$    |   \alpha    |
|    $\beta$    |    \beta    |
|   $\gamma$    |   \gamma    |
|   $\delta$    |   \delta    |
| $\varepsilon$ | \varepsilon |
|   $\lambda$   |   \lambda   |
|    $\phi$     |    \phi     |
|     $\xi$     |     \xi     |
|    $\psi$     |    \psi     |
|   $\omega$    |   \omega    |

快速获取更多希腊字母的Latex代码可以进入[手写识别网站](http://detexify.kirelabs.org/classify.html)

## 在Hexo中渲染MathJax数学公式

由于Hexo默认使用“hexo-renderer-marked”引擎渲染网页，使得一些特殊的markdown符号和html标签存在语义冲突，如下划线‘‘_’‘在Latex公式中表示下标，而在html中引擎会将其处理为$<em>$标签，导致数学公式在博客文章中出现各种错误.

### 处理方法

#### 更换渲染引擎

```powershell
npm uninstall hexo-renderer-marked --save #卸载原引擎
npm install hexo-renderer-kramed --save #安装新引擎
```

在安装新引擎时，还可能会出现由于超时导致安装失败的问题，可以先把npm换到淘宝镜像，再安装新引擎.

```powershell
npm config set registry https://registry.npm.taobao.org #永久使用（推荐）
```

#### 变量修改

到博客根目录下，找到node_modules\kramed\lib\rules\inline.js，把espace和em变量的值做相应修改.

```js
escape: /^\\([`*\[\]()#$+\-.!_>])/
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/
```

#### 开启mathjax

进入到主题目录，找到**_config.yml**，把mathjax默认的false修改为true，具体如下：

```yaml
# Math Formulas Render Support
math:
  # Default (true) will load mathjax / katex script on demand.
  # That is it only render those page which has `mathjax: true` in Front-matter.
  # If you set it to false, it will load mathjax / katex srcipt EVERY PAGE.
  per_page: true

  # hexo-renderer-pandoc (or hexo-renderer-kramed) required for full MathJax support.
  mathjax:
    enable: true
    # See: https://mhchem.github.io/MathJax-mhchem/
    mhchem: false
```

#### 文章设置

在文章的Front-matter中打开mathjax开关(开篇手动输入‘---‘即可出现该模块)

```markdown
---
title: Hexo使用数学公式
tags: hexo
categories: hexo博客搭建
mathjax: true 
---
```

这样设置的好处是只在用到数学公式的页面才加载 Mathjax，不需要渲染数学公式的页面的访问速度则不会受到影响.

