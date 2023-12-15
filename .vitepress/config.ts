import { defineConfig } from 'vitepress'
import { sidebar, nav } from './sidebar'
import { name, keywords } from './meta'

const base = process.env.BASE || './'

export default defineConfig({
  title: name,
  locales: {
    root: { label: '简体中文', lang: 'zh-CN' }
  },
  base,
  markdown: {
    math: true,
    lineNumbers: true
  },
  lastUpdated: true,
  themeConfig: {
    logo: '/heart.png',
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    search: {
        provider: 'local'
    },
    ignoreDeadLinks: true,
    returnToTopLabel: '返回顶部',
    outlineTitle: '导航栏',
    darkModeSwitchLabel: '外观',
    sidebarMenuLabel: '归档',
    lastUpdatedText: '最后一次更新于',
    outline: 'deep',
    socialLinks: [
      { icon: 'github', link: 'https://gitee.com/yixin-oss/daytoy-vitepress' },
      {
        icon: {
                    svg: '<svg viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" width="22" height="22"><path d="M640 426.666667c0 11.044571-0.487619 22.016-1.389714 32.816762a235.324952 235.324952 0 0 0 69.485714-93.720381 319.463619 319.463619 0 0 1 123.904 252.903619C832 795.40419 688.737524 938.666667 512 938.666667S192 795.40419 192 618.666667c0-90.038857 37.180952-171.398095 97.03619-229.522286a616.521143 616.521143 0 0 0 146.285715-302.104381c121.734095 64.414476 204.678095 192.341333 204.678095 339.626667z" fill="#808080"></path></svg>',
                },
        link: 'https://blog.csdn.net/yixon_oss?type=blog',
        ariaLabel: 'csdn博客'
      }
    ],
    footer: {
      message: '热爱可抵岁月漫长',
      copyright: `版权所有 © 2023-2024 Daytoy Studio`
    },
    nav,
    sidebar
  },
  head: [
    ['meta', { name: 'referrer', content: 'never' }],
    ['meta', { name: 'keywords', content: keywords }],
    ['meta', { name: 'author', content: 'Daytoy' }],

    ['link', { rel: 'icon', type: 'image/png+xml', href: '/cat.png' }]
  ]
})
