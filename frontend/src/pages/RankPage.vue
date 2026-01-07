<template>
  <div class="page-margins">
    <section>
      <p v-if="inner && outer">
        Ranking pairs for <strong>{{ inner }} – {{ outer }} m</strong>
        <span v-if="count !== null"> · {{ count }} pairs found</span>
        <span v-if="city"> · City: {{ city }}</span>
      </p>
      <p v-else>No query parameters detected.</p>
      
      <!-- Download status -->
      <p v-if="downloading" class="text-blue-400 mt-2">
        x{{ downloadStatus || 'Downloading images...' }}
      </p>
      <p v-else-if="downloadStatus" class="text-green-400 mt-2">
        ✓ {{ downloadStatus }}
      </p>
    </section>
    
    <!-- Fullscreen Map View -->
    <div v-if="showMap" class="map-overlay">
      <div class="map-controls">
        <h3>Map View - {{ city }}</h3>
        <button @click="closeMap" class="close-map-btn">✕ Close Map</button>
      </div>
      <div class="map-content">
        <MapView ref="mapView" />
      </div>
    </div>
    
    <div>
      <div v-if="items.length">
        <PairTile
          v-for="item in items"
          :key="item.id"
          :pair="item"
          @like="onLike"
          @map="onMapRequest"
        />
      </div>
      <p v-else>Loading pairs…</p>
      <button 
        v-if="cursor && !loading" 
        @click="loadMore"
        class="load-more-btn"
      >
        Load More
      </button>
      <p v-if="loading">Loading...</p>
    </div>
  </div>
</template>

<script>
import PairTile from '../components/PairTile.vue'
import MapView from '../components/MapView.vue'

const IMAGE_BASE = 'http://localhost:8000/image'

export default {
  name: 'RankPage',
  components: { PairTile, MapView },
  data() {
    return {
      inner: this.$route.query.inner ?? null,
      outer: this.$route.query.outer ?? null,
      count: this.$route.query.count ? Number(this.$route.query.count) : null,
      city: this.$route.query.city ?? null,
      items: [],
      total: null,
      cursor: null,
      loading: false,
      downloading: false,
      downloadStatus: null,
      showMap: false
    }
  },
  created() { 
    this.autoDownloadAndLoad()
  },
  methods: {
    async autoDownloadAndLoad() {
      if (this.city) {
        await this.triggerDownload()
      }
      await this.loadFirst()
    },
    async triggerDownload() {
      this.downloading = true
      this.downloadStatus = 'Checking for missing images...'
      
      try {
        const res = await fetch('http://localhost:8000/download/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ city: this.city })
        })
        
        if (!res.ok) {
          console.error('Download failed:', res.status)
          this.downloadStatus = 'Download failed, some images may be missing'
          return
        }
        
        const data = await res.json()
        console.log('Download complete:', data)
        
        if (data.downloaded > 0) {
          this.downloadStatus = `Downloaded ${data.downloaded} new images`
        } else if (data.skipped_existing > 0) {
          this.downloadStatus = 'All images already downloaded'
        } else {
          this.downloadStatus = null
        }
        
        setTimeout(() => { this.downloadStatus = null }, 3000)
      } catch (err) {
        console.error('Download error:', err)
        this.downloadStatus = 'Download error, some images may be missing'
      } finally {
        this.downloading = false
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
        if (this.city) body.city = this.city
        
        const res = await fetch('http://localhost:8000/pairs/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        })
        
        if (!res.ok) throw new Error(`/pairs failed: ${res.status}`)
        
        const data = await res.json()
        
        if (!this.city && data.city) {
          this.city = data.city
        }
        
        const mapped = (data.items || []).map(it => {
          const cityParam = this.city ? `&city=${this.city}` : ''
          return {
            id: it.id,
            left: { 
              src: `${IMAGE_BASE}?uuid=${it.left.uuid}${cityParam}`, 
              alt: '',
              uuid: it.left.uuid,
              lat: it.left.lat,
              lng: it.left.lng
            },
            right: { 
              src: `${IMAGE_BASE}?uuid=${it.right.uuid}${cityParam}`, 
              alt: '',
              uuid: it.right.uuid,
              lat: it.right.lat,
              lng: it.right.lng
            },
            liked: it.liked ?? false,
            seen: false,
            distance: it.distance
          }
        })
        
        this.items.push(...mapped)
        this.total = data.total ?? this.total
        this.cursor = data.nextCursor ?? null
      } finally {
        this.loading = false
      }
    },
    async onLike({ id, liked }) {
      const idx = this.items.findIndex(p => p.id === id)
      if (idx < 0) return
      const pair = this.items[idx]
      
      // Optimistic update
      this.items[idx] = { ...pair, liked, seen: true }
      
      // Send to backend with city
      try {
        const res = await fetch('http://localhost:8000/like/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: id,
            liked: liked,
            city: this.city  // Include city in the request
          })
        })
        
        if (!res.ok) {
          console.error('Failed to save like:', res.status)
          // Revert on error
          this.items[idx] = { ...pair, liked: !liked, seen: true }
        }
      } catch (err) {
        console.error('Error saving like:', err)
        // Revert on error
        this.items[idx] = { ...pair, liked: !liked, seen: true }
      }
    },
    async onMapRequest(pair) {
      console.log('Map requested for pair', pair)
      console.log('Coordinates - Left:', pair.left.lat, pair.left.lng)
      console.log('Coordinates - Right:', pair.right.lat, pair.right.lng)
      
      this.showMap = true
      
      await this.$nextTick()
      
      // Give the map a moment to render
      setTimeout(() => {
        if (this.$refs.mapView) {
          console.log('Calling showPair on mapView')
          this.$refs.mapView.showPair(pair)
        } else {
          console.error('mapView ref not found')
        }
      }, 100)
    },
    closeMap() {
      this.showMap = false
      if (this.$refs.mapView) {
        this.$refs.mapView.clear()
      }
    }
  }
}
</script>

<style scoped>
.load-more-btn {
  display: block;
  margin: 20px auto;
  padding: 10px 20px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}
.load-more-btn:hover {
  background: #0056b3;
}

.map-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
  background: white;
  display: flex;
  flex-direction: column;
}

.map-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background: #2c3e50;
  border-bottom: 2px solid #34495e;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.map-controls h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: white;
}

.close-map-btn {
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  transition: all 0.2s;
}

.close-map-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  transform: scale(1.05);
}

.map-content {
  flex: 1;
  position: relative;
  overflow: hidden;
}
</style>