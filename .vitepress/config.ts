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
    lastUpdatedText: '最后更新于',
    outline: 'deep',
    socialLinks: [
      { icon: 'github', link: 'https://gitee.com/yixin-oss/daytoy-vitepress' },
      {
        icon: {
                    svg: '<svg viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" width="22" height="22"><path d="M640 426.666667c0 11.044571-0.487619 22.016-1.389714 32.816762a235.324952 235.324952 0 0 0 69.485714-93.720381 319.463619 319.463619 0 0 1 123.904 252.903619C832 795.40419 688.737524 938.666667 512 938.666667S192 795.40419 192 618.666667c0-90.038857 37.180952-171.398095 97.03619-229.522286a616.521143 616.521143 0 0 0 146.285715-302.104381c121.734095 64.414476 204.678095 192.341333 204.678095 339.626667z" fill="#808080"></path></svg>',
                },
        link: 'https://blog.csdn.net/yixon_oss?type=blog',
        ariaLabel: 'csdn博客'
      },
      {
        icon: {
                    svg: '<svg viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" width="22" height="22"><path d="M975.24 527.69c37.4-36.45 50.6-89.95 34.47-139.62-16.14-49.67-58.27-85.19-109.95-92.7l-170.72-24.81a24.622 24.622 0 0 1-18.55-13.48l-76.35-154.7C611.03 55.56 564.23 26.46 512 26.46c-52.22 0-99.03 29.1-122.14 75.93l-76.35 154.7a24.645 24.645 0 0 1-18.55 13.48l-170.72 24.81c-51.68 7.51-93.81 43.03-109.95 92.7S11.36 491.25 48.76 527.7l123.53 120.41a24.631 24.631 0 0 1 7.09 21.81l-29.16 170.03c-8.83 51.47 11.93 102.52 54.19 133.22 23.88 17.35 51.8 26.16 79.93 26.16 21.64 0 43.41-5.22 63.51-15.79l152.7-80.28a24.658 24.658 0 0 1 22.93 0l152.7 80.28c20.11 10.57 41.86 15.79 63.51 15.79 28.12 0 56.06-8.81 79.93-26.16 42.25-30.7 63.01-81.74 54.19-133.22l-29.16-170.03a24.661 24.661 0 0 1 7.08-21.81l123.51-120.42z" fill="#808080"></path></svg>',
                },
        link: 'https://llmysnow.top/',
        ariaLabel: '友情链接'
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
