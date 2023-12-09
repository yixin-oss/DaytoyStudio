import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: 'CAM',
      collapsed: false,
      items: [{ text: '流线型加工路径', link: '流线型加工路径.md' }]
    }
  ]
}
