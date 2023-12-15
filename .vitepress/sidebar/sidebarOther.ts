import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '🔔pnpm',
      collapsed: false,
      items: [{ text: 'pnpm', link: 'pnpm.md' }]
    },
    {
      text: '📖Latex',
      collapsed: false,
      items: [{ text: 'Latex绘制流程图', link: 'Latex绘制流程图.md' },
  { text: 'Latex随手记', link: 'Latex随手记.md' }]
    },
    {
      text: '🔎Hexo',
      collapsed: false,
      items: [{ text: 'Hexo使用数学公式', link: 'Hexo使用数学公式.md' }]
    },
    {
      text: '💡Typora',
      collapsed: false,
      items: [{ text: 'Typora配置图片自动上传到图床', link: 'Typora配置图片自动上传到图床.md' }]
    }
  ]
}
