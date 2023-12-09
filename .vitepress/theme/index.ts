import DefaultTheme from 'vitepress/theme'
import { useRoute } from 'vitepress'
import { onMounted, watch, nextTick } from 'vue'
import mediumZoom from 'medium-zoom'
import './style/vars.css'
import './style/global.css'

export default {
  ...DefaultTheme,
  setup() {
    const route = useRoute()
    const initZoom = () => {
      mediumZoom('.main img', { background: 'var(--vp-img-bg)' })
    }
    onMounted(() => {
      initZoom()
    })
    watch(
      () => route.path,
      () => nextTick(() => initZoom())
    )
  }
}
