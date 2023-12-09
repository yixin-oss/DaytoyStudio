import { type DefaultTheme } from 'vitepress'

export default function (): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '指引',
      collapsed: false,
      items: [
        { text: 'what-is-vitepress', link: 'index.md' },
        { text: 'asset-handling', link: 'asset-handling.md' },
        { text: 'cms', link: 'cms.md' },
        { text: 'custom-theme', link: '用户主题.md' },
        { text: 'data-loading', link: 'data-loading.md' },
        { text: 'deploy', link: 'deploy.md' },
        { text: 'frontmatter', link: 'frontmatter.md' }
      ]
    }
  ]
}
