import 'leaflet/dist/leaflet.css'
import 'leaflet-draw/dist/leaflet.draw.css'

import { createApp } from 'vue'
import App from './App.vue'
import './style.css'
import router from './router'
import VueVirtualScroller from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'

createApp(App)
  .use(router)
  .use(VueVirtualScroller)
  .mount('#app')
