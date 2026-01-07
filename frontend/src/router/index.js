import { createRouter, createWebHistory } from 'vue-router'
import WelcomePage from '../pages/WelcomePage.vue'
import QueryPage from '../pages/QueryPage.vue'
import RankPage from '../pages/RankPage.vue'
import LikePage from '../pages/LikePage.vue'

const routes = [
  { path: '/', name: 'home', component: WelcomePage },
  { path: '/query', name: 'query', component: QueryPage },
  { path: '/rank', name: 'rank', component: RankPage },
  { path: '/like', name: 'like', component: LikePage}
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router