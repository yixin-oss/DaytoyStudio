import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '🍺生活札记',
      collapsed: false,
      items: [{ text: '不被定义', link: '不被定义.md' },
      { text: '遇到同频的人有多幸运', link: '遇到同频的人有多幸运.md' },
      { text: '读博那些事', link: '读博那些事.md' },
      { text: '八字短句集', link: '八字短句集.md' }
      ]
    },
    {
      text: '🍴美食纪',
      collapsed: false,
      items: [{ text: '好好吃饭就是好好生活', link: '美食纪.md' }]
    },
    {
      text: '🌿沿途风光',
      collapsed: false,
      items: [{ text: '背起行囊走四方——旅行随笔', link: '旅行随笔.md' },{ text: '长沙行', link: '长沙行.md' }]
    },
    {
      text: '🎨AI绘画',
      collapsed: false,
      items: [{ text: 'Midjourney', link: 'AI绘画.md' }]
    }
  ]
}
