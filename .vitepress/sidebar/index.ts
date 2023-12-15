import { type DefaultTheme } from 'vitepress'
//import sidebarGuide from './sidebarGuide'
import sidebarOther from './sidebarOther'
import sidebarCAM from './sidebarCAM'
//import sidebarLatex from './sidebarLatex'
import sidebarDatastructure from './sidebarDatastructure'
import sidebarMATLAB from './sidebarMATLAB'
import sidebarLife from './sidebarLife'
import sidebarCG from './sidebarCG'
import sidebarOM from './sidebarOM'
import sidebarPython from './sidebarPython'
import sidebarPython2 from './sidebarPython2'
import sidebarStart from './sidebarStart'

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
  '/CG/': { base: '/CG/', items: sidebarCG() },
  '/other/': { base: '/other/', items: sidebarOther() },
  //'/Latex/': { base: '/Latex/', items: sidebarLatex() },
  //'/DataStructure/': { base: '/DataStructure/', items: sidebarDatastructure() },
  '/MATLAB/': { base: '/MATLAB/', items: sidebarMATLAB() },
  '/Life/': { base: '/Life/', items: sidebarLife() },
  '/OM/': { base: '/OM/', items: sidebarOM() },
  '/Python/': { base: '/Python/', items: sidebarPython() },
  '/Start/': { base: '/Start/', items: sidebarStart() }
}
const nav: DefaultTheme.NavItem[] = [
  { text: '🔥CAD/CAM', items: transNav('/CAM/', sidebarCAM) },
  { text: '🎓计算几何', items: transNav('/CG/', sidebarCG) },
  { text: '🎇优化方法', items: transNav('/OM/', sidebarOM) },
  //{ text: '💱数据结构与算法', items: transNav('/DataStructure/', sidebarDatastructure) },
  { text: '⭐️MATLAB', items: transNav('/MATLAB/', sidebarMATLAB) },
  { text: '🎈Python', items: transNav('/Python/', sidebarPython2) },
  //{ text: '📖Latex', items: transNav('/Latex/', sidebarLatex) },
  { text: '🍊Life', items: transNav('/Life/', sidebarLife) },
  { text: '👻Others', items: transNav('/other/', sidebarOther) }
]

export { sidebar, nav }
