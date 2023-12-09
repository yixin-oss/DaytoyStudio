import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'Latex写作',
      collapsed: false,
      items: [{ text: 'Latex绘制流程图', link: 'Latex绘制流程图.md' },
  { text: 'Latex随手记', link: 'Latex随手记.md' }]
    }
  ]
}
