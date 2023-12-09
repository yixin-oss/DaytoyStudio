import { defineConfig } from 'vitepress'
import { sidebar, nav } from './sidebar'
import { name, keywords } from './meta'

const base = process.env.BASE || '/'

export default defineConfig({
  title: name,
  locales: {
    root: { label: '简体中文', lang: 'zh-CN' }
  },
  base: '/dayvitepress/',
  markdown: {
    math: true,
    lineNumbers: true
  },
  themeConfig: {
    logo: '/avatar.png',
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
      { icon: 'github', link: 'https://github.com/llwodexue/vitepress-blog' }
    ],
    footer: {
      message: '常备不懈，才能有备无患',
      copyright: `2023 © Daytoy Studio`
    },
    nav,
    sidebar
  },
  head: [
    ['meta', { name: 'referrer', content: 'never' }],
    ['meta', { name: 'keywords', content: keywords }],
    ['meta', { name: 'author', content: 'lyn' }],

    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/dolphin.svg' }]
  ]
})
