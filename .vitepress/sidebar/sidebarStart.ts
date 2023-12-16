import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '写在前面',
      collapsed: false,
      items: [ { text: '欢迎来到Daytoy Studio', link: '欢迎.md' }]
    },
    {
      text: '实用网站',
      collapsed: false,
      items: [ { text: '特色网站分享', link: '网站分享.md' }]
    }
  ]
}