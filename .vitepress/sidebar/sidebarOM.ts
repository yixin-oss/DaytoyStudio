import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '约束优化方法',
      collapsed: false,
      items: [{ text: '约束优化方法—惩罚函数法', link: '约束优化方法—惩罚函数法.md' },
      { text: '约束优化方法—乘子法', link: '约束优化方法—乘子法.md' }]
    },
    {
      text: '二次规划',
      collapsed: false,
      items: [{ text: '二次规划—Lagrange方法', link: '二次规划—Lagrange方法.md' }]
    },
    {
      text: 'Newton',
      collapsed: false,
      items: [{ text: '非线性方程(组)求解', link: '非线性方程(组)求解.md' },
  { text: 'Newton法求解无约束非线性规划问题', link: 'Newton法求解无约束非线性规划问题.md' },
  { text: 'Newton下山法求解非线性方程组', link: 'Newton下山法求解非线性方程组.md' }]
    },
    {
      text: 'Dijkstra',
      collapsed: false,
      items: [{ text: 'Dijkstra算法', link: 'Dijkstra算法.md' }]
    },
    {
      text: '其他',
      collapsed: false,
      items: [{ text: '隐式QR迭代算法', link: '隐式QR迭代算法.md' }]
    }
  ]
}