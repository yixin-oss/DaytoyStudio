import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '数据降维',
      collapsed: false,
      items: [{ text: 'Matlab数据处理--数据降维', link: 'Matlab数据处理--数据降维.md' }]
    },
    {
      text: 'Fourier变换',
      collapsed: false,
      items: [{ text: '快速傅里叶变换(FFT)及应用实例', link: '快速傅里叶变换(FFT)及应用实例.md' }]
    },
   {
      text: '稀疏自编码器',
      collapsed: false,
      items: [{ text: '稀疏自编码器重构数据点的Matlab实现', link: '稀疏自编码器重构数据点的Matlab实现.md' }]
   }
  ]
}
