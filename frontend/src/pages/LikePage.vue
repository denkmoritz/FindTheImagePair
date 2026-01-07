<template>
  <div class="page-margins">
    <section class="mt-10 mb-4 text-sm leading-8">
      <h1 class="text-2xl font-bold mb-2">Liked Pairs</h1>
      <p class="text-gray-200">
        All your liked image pairs from the current session.
      </p>
    </section>

    <!-- Filters Section -->
    <div class="filter-section mb-6">
      <!-- City Filter -->
      <div class="filter-group mb-4">
        <label class="text-gray-200 font-semibold mb-2 block">Filter by City</label>
        <div class="flex gap-2 flex-wrap">
          <button 
            @click="selectedCities = []" 
            :class="['city-btn', { active: selectedCities.length === 0 }]"
          >
            All Cities
          </button>
          <button 
            v-for="city in availableCities" 
            :key="city"
            @click="toggleCity(city)"
            :class="['city-btn', { active: selectedCities.includes(city) }]"
          >
            {{ city.charAt(0).toUpperCase() + city.slice(1) }}
          </button>
        </div>
      </div>

      <!-- Distance Filter -->
      <div class="filter-group">
        <label class="text-gray-200 font-semibold mb-2 block">Filter by Distance (meters)</label>
        <div class="flex gap-4 items-center flex-wrap">
          <div class="flex items-center gap-2">
            <label class="text-gray-300 text-sm">Min:</label>
            <input 
              v-model.number="minDistance" 
              type="number" 
              step="0.1"
              class="distance-input"
              placeholder="0"
            />
          </div>
          <div class="flex items-center gap-2">
            <label class="text-gray-300 text-sm">Max:</label>
            <input 
              v-model.number="maxDistance" 
              type="number" 
              step="0.1"
              class="distance-input"
              placeholder="∞"
            />
          </div>
          <button 
            @click="clearFilters" 
            class="clear-btn"
          >
            Clear All Filters
          </button>
        </div>
      </div>

      <p class="text-gray-400 text-sm mt-3">
        Showing {{ filteredLikes.length }} of {{ likes.length }} pairs
      </p>
    </div>

    <div v-if="loading" class="text-gray-300 mt-4">Loading…</div>
    <p v-else-if="!likes.length" class="text-gray-300 mt-4">No liked pairs found.</p>
    <p v-else-if="!filteredLikes.length" class="text-gray-300 mt-4">No pairs match the current filters.</p>
    
    <div v-else class="pairs-grid">
      <div 
        v-for="like in filteredLikes" 
        :key="`${like.uuid_1}|${like.uuid_2}`"
        class="like-card"
      >
        <div class="like-card-images">
          <img 
            :src="`http://localhost:8000/image?uuid=${like.uuid_1}${like.city ? '&city=' + like.city : ''}`" 
            alt="Image 1" 
          />
          <img 
            :src="`http://localhost:8000/image?uuid=${like.uuid_2}${like.city ? '&city=' + like.city : ''}`" 
            alt="Image 2" 
          />
        </div>
        <div class="like-card-meta">
          <div class="meta-row">
            <span class="meta-label">Distance:</span>
            <span class="meta-value">{{ like.distance ? like.distance.toFixed(2) : 'N/A' }} m</span>
          </div>
          <div class="meta-row" v-if="like.city">
            <span class="meta-label">City:</span>
            <span class="meta-value city-tag">{{ like.city.charAt(0).toUpperCase() + like.city.slice(1) }}</span>
          </div>
          <div class="meta-row" v-if="like.created_at">
            <span class="meta-label">Liked:</span>
            <span class="meta-value text-sm">{{ formatDate(like.created_at) }}</span>
          </div>
          <button 
            @click="unlikePair(like)" 
            class="unlike-btn"
          >
            Remove Like
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'LikePage',
  data() {
    return {
      likes: [],
      loading: false,
      minDistance: null,
      maxDistance: null,
      selectedCities: [] // Empty means show all
    }
  },
  computed: {
    availableCities() {
      // Get unique cities from likes
      const cities = [...new Set(this.likes.map(like => like.city).filter(Boolean))]
      return cities.sort()
    },
    filteredLikes() {
      return this.likes.filter(like => {
        // City filter
        if (this.selectedCities.length > 0) {
          if (!like.city || !this.selectedCities.includes(like.city)) {
            return false
          }
        }

        // Distance filter
        if (like.distance === null || like.distance === undefined) {
          return true // Include pairs without distance data
        }
        
        const dist = like.distance
        const passesMin = this.minDistance === null || this.minDistance === '' || dist >= this.minDistance
        const passesMax = this.maxDistance === null || this.maxDistance === '' || dist <= this.maxDistance
        
        return passesMin && passesMax
      })
    }
  },
  created() {
    this.fetchLikes()
  },
  methods: {
    async fetchLikes() {
      this.loading = true
      try {
        const res = await fetch('http://localhost:8000/liked/')
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`)
        const data = await res.json()
        this.likes = data.items || []
      } catch (e) {
        console.error('Error fetching likes:', e)
        alert('Failed to load liked pairs.')
      } finally {
        this.loading = false
      }
    },
    async unlikePair(like) {
      if (!confirm('Are you sure you want to remove this like?')) return
      
      try {
        const res = await fetch('http://localhost:8000/like/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: `${like.uuid_1}|${like.uuid_2}`,
            liked: false,
            city: like.city
          })
        })
        
        if (!res.ok) throw new Error(`Failed to unlike: ${res.status}`)
        
        // Remove from local array
        this.likes = this.likes.filter(l => 
          !(l.uuid_1 === like.uuid_1 && l.uuid_2 === like.uuid_2)
        )
      } catch (e) {
        console.error('Error unliking pair:', e)
        alert('Failed to remove like.')
      }
    },
    toggleCity(city) {
      const index = this.selectedCities.indexOf(city)
      if (index === -1) {
        this.selectedCities.push(city)
      } else {
        this.selectedCities.splice(index, 1)
      }
    },
    clearFilters() {
      this.minDistance = null
      this.maxDistance = null
      this.selectedCities = []
    },
    formatDate(timestamp) {
      if (!timestamp) return 'N/A'
      const date = new Date(timestamp)
      return date.toLocaleString('en-GB', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
  }
}
</script>

<style scoped>
.filter-section {
  background: rgba(30, 40, 50, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 20px;
}

.filter-group {
  margin-bottom: 16px;
}

.filter-group:last-child {
  margin-bottom: 0;
}

.city-btn {
  background: rgba(20, 30, 40, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  padding: 8px 16px;
  color: rgba(255, 255, 255, 0.7);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.city-btn:hover {
  background: rgba(40, 50, 60, 0.8);
  border-color: rgba(255, 255, 255, 0.3);
  color: white;
}

.city-btn.active {
  background: rgba(56, 189, 248, 0.3);
  border-color: rgba(56, 189, 248, 0.6);
  color: #38bdf8;
}

.distance-input {
  background: rgba(20, 30, 40, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  padding: 8px 12px;
  color: white;
  width: 120px;
  font-size: 14px;
  appearance: textfield;
  -moz-appearance: textfield;
}

.distance-input::-webkit-outer-spin-button,
.distance-input::-webkit-inner-spin-button {
  -webkit-appearance: none;
  appearance: none;
  margin: 0;
}

.distance-input:focus {
  outline: none;
  border-color: rgba(56, 189, 248, 0.6);
  box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1);
}

.clear-btn {
  background: rgba(100, 116, 139, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  padding: 8px 16px;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.clear-btn:hover {
  background: rgba(100, 116, 139, 0.7);
  border-color: rgba(255, 255, 255, 0.3);
}

.pairs-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
  gap: 24px;
  margin-top: 24px;
}

.like-card {
  background: rgba(20, 30, 40, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
  transition: all 0.3s ease;
}

.like-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
  border-color: rgba(255, 255, 255, 0.15);
}

.like-card-images {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 12px;
}

.like-card-images img {
  width: 100%;
  aspect-ratio: 4/3;
  object-fit: cover;
  border-radius: 8px;
  background: rgba(40, 50, 60, 0.5);
}

.like-card-meta {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.meta-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
}

.meta-label {
  color: rgba(255, 255, 255, 0.6);
  font-weight: 500;
}

.meta-value {
  color: rgba(255, 255, 255, 0.9);
  font-weight: 600;
}

.city-tag {
  background: rgba(56, 189, 248, 0.2);
  border: 1px solid rgba(56, 189, 248, 0.4);
  padding: 2px 8px;
  border-radius: 4px;
  color: #38bdf8;
  font-size: 13px;
}

.unlike-btn {
  margin-top: 8px;
  padding: 8px 16px;
  background: rgba(220, 38, 38, 0.2);
  border: 1px solid rgba(220, 38, 38, 0.5);
  border-radius: 6px;
  color: #ff6b6b;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.unlike-btn:hover {
  background: rgba(220, 38, 38, 0.3);
  border-color: rgba(220, 38, 38, 0.6);
  transform: translateY(-1px);
}
</style>