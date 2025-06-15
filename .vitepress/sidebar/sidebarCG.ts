import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '📐数值逼近',
      collapsed: false,
      items: [{ text: '最小二乘法', link: '最小二乘法.md' }]
    },
    {
      text: '✨微分几何',
      collapsed: false,
      items: [{ text: '理论基础', link: '理论基础.md' } ,{ text: 'Dupin指标线', link: 'Dupin指标线.md' }]
    },
    {
      text: '🎐编程练习',
      collapsed: false,
      items: [{ text: '计算几何中的编程练习', link: '计算几何中的编程练习.md' }]
    },
    {
      text: '🎯样条曲线曲面',
      collapsed: false,
      items: [{ text: 'Bezier曲线曲面绘制', link: 'Bezier曲线曲面绘制.md' },
       { text: '双三次B样条曲面绘制及微分量计算', link: '双三次B样条曲面绘制及微分量计算.md' },
  { text: 'Matlab样条工具箱及曲线拟合', link: 'Matlab样条工具箱及曲线拟合.md' },
  { text: '基于MATLAB的B样条曲线插值算法', link: '基于MATLAB的B样条曲线插值算法.md' },
  { text: 'de Casteljau算法与de Boor算法', link: 'de Casteljau算法与de Boor算法.md'}

      ]
    },
    {
      text: '💯试题练习',
      collapsed: false,
      items: [{ text: '计算几何测试题', link: '计算几何测试题.md' },
  { text: '计算几何复习题', link: '计算几何复习题.md' },
  { text: '计算几何测试题答案', link: '计算几何测试题答案.md' }]
    }
  ]
}
