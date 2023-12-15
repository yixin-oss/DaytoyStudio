import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
     {
      text: '🔥CAD',
      collapsed: false,
      items: [{ text: '自由曲面重构——数据点参数化', link: '自由曲面重构--数据点参数化.md' },
      { text: 'MATLAB中实体模型的可视化', link: 'MATLAB中实体模型的可视化.md' }]
    },
    {
      text: '🔥CAM',
      collapsed: false,
      items: [{ text: '流线型加工路径', link: '流线型加工路径.md' }]
    },
    {
      text: '🔥文献整理',
      collapsed: false,
      items: [{ text: '文献整理', link: '文献整理.md' }]
    }
  ]
}

