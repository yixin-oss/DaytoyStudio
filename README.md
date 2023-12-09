# vitepress-blog

## 文档使用方法

### 安装依赖

1. 安装 Node 

   [https://nodejs.cn/download/](https://nodejs.cn/download/)

   Node 需要大于等于 16.16.0 版本，使用最新的即可

   ![image-20231205160637163](https://gitee.com/lilyn/pic/raw/master/lagoulearn-img/image-20231205160637163.png)

2. 全局安装 pnpm

   ```bash
   npm i -g pnpm
   ```

3. 启动项目

   ```bash
   pnpm dev
   ```

### 修改项目配置

比较重要的几个文件

- index.md：用来修改首页内容信息
- config.ts 最主要就是 sidebar 和 nav 两个配置，我给抽取到 sidebar/index.ts 下面了
- sidebar/index.ts 生成页面需要的 sidebar 和 nav

![image-20231205160852143](https://gitee.com/lilyn/pic/raw/master/lagoulearn-img/image-20231205160852143.png)

sidebar 下面的路由自己管理

- base 是文件名
- items 我是分模块管理的

![image-20231205162711385](https://gitee.com/lilyn/pic/raw/master/lagoulearn-img/image-20231205162711385.png)

如果文档不多可以不用分模块管理

- 可以参考：[https://github.com/vuejs/vitepress/blob/main/docs/.vitepress/config.ts](https://github.com/vuejs/vitepress/blob/main/docs/.vitepress/config.ts)

创建 `sidebarXXX.ts` 文件，主要修改 text 和 items

- 复制粘贴创建新文件就行

![image-20231205162812670](https://gitee.com/lilyn/pic/raw/master/lagoulearn-img/image-20231205162812670.png)

items 可以使用脚本生成

- 修改对应 filePath，运行生成对应的 items

![image-20231205163000811](https://gitee.com/lilyn/pic/raw/master/lagoulearn-img/image-20231205163000811.png)

## 自动化部署

新建 git 项目

### 开启Pages

GitHub Actions 配置文件，参考 vitepress 官方教程：[https://vitepress.dev/guide/deploy#github-pages](https://vitepress.dev/guide/deploy#github-pages)

点击 Pages，Source 选择 GitHub Actions，操作如下图：

![image-20231205105845133](https://gitee.com/lilyn/pic/raw/master/lagoulearn-img/image-20231205105845133.png)

### 设置base

vitepress 中的 base 需要设置为项目名。修改 `package.json`

- BASE 改为你 GitHub 该项目的项目名，我的项目名为 `vitepress-blog` -> `BASE=/vitepress-blog/`

```json
"scripts": {
  // 专门针对 github pages 目录设置的变量
  "build": "cross-env BASE=/vitepress-blog/ vitepress build",
  // 部署到其他地方
  "build:blog": "vitepress build"
}
```

