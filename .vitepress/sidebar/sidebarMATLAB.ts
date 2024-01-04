import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '数据处理',
      collapsed: false,
      items: [{ text: 'Matlab数据处理——数据降维', link: 'Matlab数据处理--数据降维.md' },
      {text: '拉普拉斯特征映射(Laplacian-Eigenmaps)', link: '拉普拉斯特征映射(Laplacian-Eigenmaps).md'
  }
]
    },
    {
      text: 'Fourier变换',
      collapsed: false,
      items: [{ text: '快速傅里叶变换(FFT)及应用实例', link: '快速傅里叶变换(FFT)及应用实例.md' }]
    },
    {
      text: '随机模拟',
      collapsed: false,
      items: [ { text: '蒙特卡洛(Monte Carlo)模拟及应用', link: '蒙特卡洛模拟及应用.md' }]
    },
   {
      text: '智能计算',
      collapsed: false,
      items: [{ text: '稀疏自编码器重构数据点的Matlab实现', link: '稀疏自编码器重构数据点的Matlab实现.md' }]
   },
   {
      text: '画法几何',
      collapsed: false,
      items: [{ text: 'Matlab实现任意圆柱体绘制', link: 'Matlab实现任意圆柱体绘制.md' },
      { text: 'MATLAB画图技巧', link: 'MATLAB画图技巧.md' }]
   }
  ]
}
