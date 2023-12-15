import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '📌函数基础',
      collapsed: false,
      items: [  { text: 'Python函数及变量的定义和使用', link: 'Python函数及变量的定义和使用.md' }]
    },
    {
      text: '💱数据结构与算法',
      collapsed: false,
      items: [ { text: '栈与队列', link: '栈与队列.md' },
                { text: '树及其应用', link: '树及其应用.md' },
                { text: '动态规划案例分析', link: '动态规划案例分析.md' },
                { text: '二分查找及应用', link: '二分查找及应用.md' },
                { text: '排序算法及其应用', link: '排序算法及其应用.md'},
                { text: '搜索与回溯算法应用实例', link: '搜索与回溯算法应用实例.md' },
                { text: '二叉堆的Python实现', link: '二叉堆的Python实现.md' },
                { text: '双指针及其应用', link: '双指针及其应用.md' }
                ]
    },
    {
      text: '🚩人工智能',
      collapsed: false,
      items: [{ text: 'python机器学习之鸢尾花分类问题', link: 'python机器学习之鸢尾花分类问题.md' },
              { text: 'Windows10安装TensorFlow', link: 'Windows10安装TensorFlow.md' },
              {text: '人工智能：Tensorflow2笔记(一)', link: '人工智能：Tensorflow2笔记(一).md'},
              { text: '人工智能-Tensorflow2笔记(二)', link: '人工智能-Tensorflow2笔记(二).md' },
              { text: '人工智能-Tensorflow2笔记(三)', link: '人工智能-Tensorflow2笔记(三).md' },
              { text: 'DBSCAN聚类算法', link: 'DBSCAN聚类算法.md' }
             ]
    },
    {
      text: '💝turtle库',
      collapsed: false,
      items: [{ text: 'python turtle库“一箭穿心”代码实现', link: 'python turtle库“一箭穿心”代码实现.md'}]
    },
    {
      text: '📋Wordcloud库',
      collapsed: false,
      items: [{ text: 'wordcloud库', link: 'wordcloud库.md'}]
    }
  ]
}