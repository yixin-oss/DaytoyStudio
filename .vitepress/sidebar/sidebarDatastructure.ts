import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '基本数据结构',
      collapsed: false,
      items: [{ text: '栈与队列', link: '栈与队列.md' }]
    },
    {
      text: '动态规划',
      collapsed: false,
      items: [{ text: '动态规划案例分析', link: '动态规划案例分析.md' }]
    },
    {
      text: '查找与排序',
      collapsed: false,
      items: [{ text: '二分查找及应用', link: '二分查找及应用.md' },
  { text: '排序算法及其应用', link: '排序算法及其应用.md' }]
    },
    {
      text: '搜索与回溯',
      collapsed: false,
      items: [{ text: '搜索与回溯算法应用实例', link: '搜索与回溯算法应用实例.md' }]
    },
    {
      text: '其他',
      collapsed: false,
      items: [{ text: '二叉堆的Python实现', link: '二叉堆的Python实现.md' },
      { text: '双指针及其应用', link: '双指针及其应用.md' }]
    }
  ]
}
