import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '写在前面',
      collapsed: false,
      items: [ { text: '欢迎来到Daytoy Studio', link: '欢迎来到Daytoy Studio.md' }]
    }
  ]
}