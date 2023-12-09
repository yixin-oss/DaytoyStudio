import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'pnpm',
      collapsed: false,
      items: [{ text: 'pnpm', link: 'pnpm.md' }]
    },
    {
      text: 'math',
      collapsed: false,
      items: [{ text: 'Bezier曲线曲面绘制', link: 'Bezier曲线曲面绘制.md' }]
    }
  ]
}
