import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
     {
      text: '🔥CAD',
      collapsed: false,
      items: [{ text: '自由曲面重构——数据点参数化', link: '自由曲面重构--数据点参数化.md' },
      { text: '三角网格曲面协调映射参数化', link: '三角网格曲面协调映射参数化.md' },
      { text: 'MATLAB中实体模型的可视化', link: 'MATLAB中实体模型的可视化.md' },
      { text: 'UG创建参数曲面加工模型', link: 'UG创建参数曲面加工模型.md' }]
    },
    {
      text: '🔥CAM',
      collapsed: false,
      items: [{ text: '流线型加工路径', link: '流线型加工路径.md' },
      {
        text: 'Tool path planning method for parametric surfaces based on piecewise Coons stream function reconstruction on hierarchical T-meshes',
        link: 'Tool path planning method for parametric surfaces based on piecewise Coons stream function reconstruction on hierarchical T-meshes.md'
      },
      { text: '余切拉普拉斯算子与向量场离散散度定理推导', link: '余切拉普拉斯算子与向量场离散散度定理推导.md' },
      {
        text: '基于热理论的测地距离计算(Geodesics in Heat)',
        link: '基于热理论的测地距离计算(Geodesics in Heat).md'
      }
    ]
    },
    {
      text: '🔥文献整理',
      collapsed: false,
      items: [{ text: '文献整理', link: '文献整理.md' }]
    }
  ]
}

