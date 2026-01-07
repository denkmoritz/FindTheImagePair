<template>
  <div>
    <!-- Intro -->
    <section class="mx-32 mt-16 mb-8 text-sm leading-8 text-justify">
      <p>
        This is the actual querying part. Please enter the desired range for the circles, e.g., 7 - 12 m is reasonable.
      </p>
    </section>

    <!-- Query form -->
    <p class="mx-32 mt-8 mb-8 text-m leading-8 text-justify">
      Enter the distance for the desired range (in meters):
    </p>

    <form class="mx-32 mb-6 flex items-center space-x-4" @submit.prevent="runQuery">
      <input v-model="inner" type="text" placeholder="Inner Circle" class="border border-gray-300 rounded w-32 text-center" />
      <input v-model="outer" type="text" placeholder="Outer Circle" class="border border-gray-300 rounded w-32 text-center" />
      <button type="submit" class="border border-gray-300 hover:bg-teal-700 font-semibold rounded shadow w-28 text-center">
        Query
      </button>

      <div v-if="count !== null" class="text-sm/8">
        {{ count }} pairs found for {{ inner }} - {{ outer }} m
      </div>
    </form>

    <!-- Controls -->
    <div class="mx-32 mb-2 flex items-center gap-4">
      <label class="text-sm">Pair width:</label>
      <select v-model.number="tileWidth" class="border rounded px-2 py-1 text-sm">
        <option :value="640">640 px</option>
        <option :value="760">760 px</option>
        <option :value="880">880 px</option>
        <option :value="1040">1040 px</option>
      </select>

      <label class="text-sm">Gap:</label>
      <select v-model.number="tileGap" class="border rounded px-2 py-1 text-sm">
        <option :value="16">16</option>
        <option :value="24">24</option>
        <option :value="32">32</option>
        <option :value="48">48</option>
      </select>

      <label class="text-sm">Aspect:</label>
      <select v-model="aspect" class="border rounded px-2 py-1 text-sm">
        <option value="4 / 3">4:3</option>
        <option value="3 / 2">3:2</option>
        <option value="16 / 9">16:9</option>
        <option value="1 / 1">1:1</option>
      </select>
    </div>

    <!-- STICKY progress (always visible) -->
    <div class="sticky-progress">
      <div class="mx-32">
        <ProgressBar :reviewed="reviewed" :total="total || count || 0" />
      </div>
    </div>

    <!-- Virtualized gallery -->
    <div class="mx-32 mb-10">
      <DynamicScroller
        v-if="items.length"
        ref="scroller"
        :items="items"
        :min-item-size="minItemSize"
        key-field="id"
        class="v-list"
        :style="{ '--gap': tileGap + 'px', '--tile-w': tileWidth + 'px' }"
        @scroll-end="onScrollEnd"
      >
        <template #default="{ item }">
          <DynamicScrollerItem :item="item" :active="true">
            <div class="row">
              <div class="tile-wrap">
                <PairTile
                  :pair="item"
                  :aspect="aspect"
                  @rate="onRate"
                  @map="onMapRequest"
                  v-observe="visible => onVisible(item, visible)"
                />
              </div>
            </div>
          </DynamicScrollerItem>
        </template>
      </DynamicScroller>

      <p v-else class="text-sm text-gray-500">No pairs yet — run “Query”.</p>

      <div v-if="loading" class="mt-4 text-sm opacity-70">Loading…</div>
      <div v-else-if="!cursor && items.length && total && items.length >= total" class="mt-4 text-sm opacity-70">
        End of results.
      </div>
    </div>

    <!-- MAP MODAL -->
    <dialog ref="mapDialog" class="map-dialog" @click="onMapBackdrop">
      <div class="map-stage">
        <button @click="closeMap" class="map-close" aria-label="Close">×</button>
        <MapView ref="mapView" />
      </div>
    </dialog>
  </div>
</template>

<script>
import PairTile from '../components/PairTile.vue'
import ProgressBar from '../components/ProgressBar.vue'
import MapView from '../components/MapView.vue'

export default {
  name: 'MapPage',
  components: { MapView, PairTile, ProgressBar },

  directives: {
    /**
     * IO that uses the scroller container as root (or viewport if none)
     * Marks items "seen" when >= 60% visible.
     * Also updates the bound callback on component updates so it stays in sync
     * with recycled DOM from vue-virtual-scroller.
     */
    observe: {
      mounted(el, binding) {
        const getRoot = () => el.closest('.vue-recycle-scroller') || null
        const cb = (entries) => {
          for (const e of entries) {
            const visible = e.isIntersecting && e.intersectionRatio >= 0.6
            el.__onVisible__ && el.__onVisible__(visible)
          }
        }
        el.__onVisible__ = binding.value
        el.__io__ = new IntersectionObserver(cb, { root: getRoot(), threshold: [0, 0.6, 1] })
        el.__io__.observe(el)
      },
      updated(el, binding) {
        // keep latest callback when item is recycled
        el.__onVisible__ = binding.value
      },
      unmounted(el) {
        el.__io__ && el.__io__.disconnect()
        delete el.__io__
        delete el.__onVisible__
      }
    }
  },

  data() {
    return {
      inner: null,
      outer: null,
      count: null,

      items: [],
      total: null,
      cursor: null,
      loading: false,

      tileWidth: 880,  // wider default
      tileGap: 24,
      aspect: '4 / 3',

      selectedPair: null,

      _queue: [],
      _flushTimer: null
    }
  },

  computed: {
    reviewed() {
      return this.items.filter(p => p.seen || p.rating !== null).length
    },
    minItemSize() {
      // estimate height for virtualization: 2 images side-by-side
      const parts = String(this.aspect).split('/')
      const w = parseFloat(parts[0])
      const h = parseFloat(parts[1])
      const ar = (isFinite(w) && isFinite(h) && h !== 0) ? (w / h) : (4 / 3)
      const imageHeight = (this.tileWidth / 2) / ar
      const chrome = 140 // padding + stars + margins
      return Math.round(imageHeight + chrome)
    }
  },

  methods: {
    // -------- server calls ----------
    async runQuery() {
      try {
        const inner = parseFloat(this.inner)
        const outer = parseFloat(this.outer)
        const response = await fetch('http://localhost:8000/query/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ inner_buffer: inner, outer_buffer: outer })
        })
        if (!response.ok) throw new Error(`Request failed: ${response.status}`)
        const data = await response.json()
        this.count = data.count
        await this.loadFirst()
      } catch (err) {
        console.error('Query failed:', err)
      }
    },

    async loadFirst() {
      this.items = []
      this.total = null
      this.cursor = null
      await this.loadMore()
    },

    async loadMore() {
      if (this.loading) return
      if (this.total !== null && !this.cursor && this.items.length >= this.total) return
      this.loading = true
      try {
        const body = { limit: 50 }
        if (this.cursor) body.cursor = this.cursor
        const res = await fetch('http://localhost:8000/pairs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        })
        if (!res.ok) throw new Error(`/pairs failed: ${res.status}`)
        const data = await res.json()
        const mapped = (data.items || []).map(it => ({
          id: it.id,
          left: it.left,
          right: it.right,
          rating: it.rating ?? null,
          seen: !!it.seen,
          starred: !!it.starred
        }))
        this.items.push(...mapped)
        this.total = data.total ?? this.total
        this.cursor = data.nextCursor ?? null

        this.$nextTick(() => this.$refs.scroller?.forceUpdate?.())
      } finally {
        this.loading = false
      }
    },

    onScrollEnd() {
      if (this.cursor) this.loadMore()
    },

    // -------- interactions ----------
    onVisible(item, visible) {
      if (!visible) return
      const idx = this.items.findIndex(p => p.id === item.id)
      if (idx >= 0 && !this.items[idx].seen) {
        this.items[idx] = { ...this.items[idx], seen: true }
        this.queueInteraction({ pairId: item.id, seen: true })
      }
    },

    async onRate({ id, rating }) {
      const idx = this.items.findIndex(p => p.id === id)
      if (idx < 0) return
      const prev = this.items[idx]
      this.items[idx] = { ...prev, rating, seen: true }
      this.queueInteraction([{ pairId: id, rating, seen: true }])
    },

    queueInteraction(payload) {
      const arr = Array.isArray(payload) ? payload : [payload]
      this._queue.push(...arr)
      clearTimeout(this._flushTimer)
      this._flushTimer = setTimeout(this.flushInteractions, 400)
    },

    async flushInteractions() {
      if (!this._queue.length) return
      const batch = this._queue.splice(0, this._queue.length)
      try {
        await fetch('http://localhost:8000/interactions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(batch)
        })
      } catch (e) {
        // naive retry
        this._queue.unshift(...batch)
      }
    },

    // -------- map modal ----------
    onMapRequest(pair) {
      this.selectedPair = pair
      const idx = this.items.findIndex(p => p.id === pair.id)
      if (idx >= 0 && !this.items[idx].seen) {
        this.items[idx] = { ...this.items[idx], seen: true }
        this.queueInteraction({ pairId: pair.id, seen: true })
      }
      const dlg = this.$refs.mapDialog
      if (dlg && !dlg.open) dlg.showModal()
      this.$nextTick(() => this.$refs.mapView?.showPair?.(pair))
      document.addEventListener('keydown', this.onEsc)
    },

    closeMap() {
      const dlg = this.$refs.mapDialog
      if (dlg?.open) dlg.close()
      this.$refs.mapView?.clear?.()
      this.selectedPair = null
      document.removeEventListener('keydown', this.onEsc)
    },

    onEsc(e) { if (e.key === 'Escape') this.closeMap() },
    onMapBackdrop(e) { if (e.target === this.$refs.mapDialog) this.closeMap() }
  }
}
</script>

<style scoped>
/* Sticky progress: always visible */
.sticky-progress {
  position: sticky;
  top: 0;
  z-index: 50;
  padding: 10px 0 8px;
  background: rgba(12, 22, 32, 0.75);      /* subtle backdrop so text stays readable */
  -webkit-backdrop-filter: blur(4px);
  backdrop-filter: blur(4px);
  border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* Virtualized list layout: one centered tile per row */
.v-list { margin: 0; }
.row { width: 100%; padding: var(--gap) 0; }
.tile-wrap {
  width: var(--tile-w);
  margin: 0 auto;
  padding: 0 var(--gap);
  box-sizing: border-box;
}

/* Kill the magnifier/hover on images */
.tile-wrap :deep(.imgs) { cursor: default; pointer-events: none; }
.tile-wrap :deep(.map) { pointer-events: auto; }

/* vue-virtual-scroller internal wrapper behaves like block rows */
.v-list :deep(.vue-recycle-scroller__item-view) { display: block; }

/* Map dialog (unchanged) */
.map-dialog {
  width: min(1200px, 96vw);
  margin: auto;
  padding: 0;
  border: none;
  background: transparent;
}
.map-dialog::backdrop { background: rgba(0,0,0,.35); backdrop-filter: blur(6px); }
.map-stage { position: relative; background: #111; padding: 0; border-radius: 14px; overflow: hidden; width: 100%; height: auto; }
.map-stage #map { width: 100%; height: 100%; }
.map-close {
  position: absolute; top: 8px; right: 8px; width: 36px; height: 36px; border-radius: 50%;
  border: none; background: rgba(255,255,255,0.12); color: #fff; font-size: 20px;
  line-height: 36px; text-align: center; cursor: pointer; box-shadow: 0 2px 10px rgba(0,0,0,.35);
  backdrop-filter: blur(2px); z-index: 10;
}
.map-close:hover { background: rgba(255,255,255,0.18); }
.map-close:focus-visible { outline: 2px solid rgba(255,255,255,0.45); outline-offset: 2px; }
</style>