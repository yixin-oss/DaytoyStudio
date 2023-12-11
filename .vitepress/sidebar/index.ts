import { type DefaultTheme } from 'vitepress'
//import sidebarGuide from './sidebarGuide'
import sidebarOther from './sidebarOther'
import sidebarCAM from './sidebarCAM'
import sidebarLatex from './sidebarLatex'
import sidebarDatastructure from './sidebarDatastructure'
import sidebarMATLAB from './sidebarMATLAB'

const transNav = (base: string, arrFn: () => DefaultTheme.SidebarItem[]) => {
  const nav = arrFn().map(i => {
    const link = i.items![0].link
    return { text: i!.text || '', link: `${base}${link}` }
  })
  return nav
}
// TODO 需要自己生成
const sidebar = {
  '/CAM/': { base: '/CAM/', items: sidebarCAM() },
  '/other/': { base: '/other/', items: sidebarOther() },
  '/Latex/': { base: '/Latex/', items: sidebarLatex() },
  '/DataStructure/': { base: '/DataStructure/', items: sidebarDatastructure() },
  '/MATLAB/': { base: '/MATLAB/', items: sidebarMATLAB() }
}
const nav: DefaultTheme.NavItem[] = [
  { text: 'Start', items: transNav('/CAM/', sidebarCAM) },
  { text: '其他', items: transNav('/other/', sidebarOther) },
  { text: 'Latex', items: transNav('/Latex/', sidebarLatex) },
  { text: '数据结构与算法', items: transNav('/DataStructure/', sidebarDatastructure) },
  { text: 'MATLAB', items: transNav('/MATLAB/', sidebarMATLAB) }
]

export { sidebar, nav }
