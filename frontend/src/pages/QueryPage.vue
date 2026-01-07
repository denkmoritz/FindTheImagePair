<template>
  <div class="page-margins">
    <!-- Intro -->
    <section class="mt-16 mb-8 text-m leading-8 text-justify">
      <p>
        Choose a city and a buffer range (e.g., 7 – 12 m). Optionally draw a circle on the map to limit the search area.
        If you don't draw anything, the entire city dataset will be used.
      </p>
    </section>

    <!-- City selector -->
    <div class="mb-6">
      <p class="mb-3 text-m leading-8">Select a city:</p>
      <div v-if="loadingCities" class="flex gap-3 flex-wrap">
        <button
          v-for="i in 6"
          :key="i"
          class="px-6 py-3 rounded-lg border-2 border-gray-600 bg-gray-800/50 text-gray-400 animate-pulse"
          disabled
        >
          Loading...
        </button>
      </div>
      <div v-else-if="cities.length > 0" class="flex gap-3 flex-wrap">
        <button
          v-for="city in cities"
          :key="city.name"
          @click="selectCity(city)"
          :class="[
            'px-6 py-3 rounded-lg border-2 font-semibold transition-all',
            selectedCity?.name === city.name
              ? 'border-teal-500 bg-teal-500/20 text-teal-100'
              : 'border-gray-600 bg-gray-800/50 hover:border-gray-500'
          ]"
        >
          {{ city.displayName }}
        </button>
      </div>
      <div v-else class="text-red-400">
        No cities available. Please check the backend configuration.
      </div>
    </div>

    <!-- Area picker -->
    <div v-if="selectedCity" class="mb-6">
      <AreaPickerMap 
        v-model="area" 
        :key="selectedCity.name"
        :initialCenter="selectedCity.center"
        :initialZoom="selectedCity.zoom"
      />
    </div>

    <!-- Query form -->
    <div v-if="selectedCity">
      <p class="mt-2 mb-4 text-m leading-8 text-justify">
        Distance range (meters):
      </p>
      <form class="mb-6 flex items-center flex-wrap gap-4" @submit.prevent="runQuery">
        <input 
          v-model="inner" 
          type="text" 
          placeholder="Inner Circle" 
          class="border border-gray-300 rounded w-32 mr-4 text-center"
        />
        <input 
          v-model="outer" 
          type="text" 
          placeholder="Outer Circle" 
          class="border border-gray-300 rounded w-32 mr-4 text-center"
        />
        <button 
          :disabled="loading" 
          type="submit" 
          class="border border-gray-300 hover:bg-teal-700 font-semibold rounded shadow w-20 text-center"
        >
          {{ loading ? 'Querying…' : 'Query' }}
        </button>
        <div v-if="count !== null" class="text-sm/8">
          {{ count }} pairs found for {{ inner }} – {{ outer }} m
          <span v-if="area" class="opacity-70">in selected area</span>
          <span v-else class="opacity-70">in {{ selectedCity.displayName }}</span>
        </div>
      </form>
      <div>
        <button
          v-if="count !== null && count > 0"
          @click="goRank"
          class="mb-8 border border-gray-300 hover:bg-teal-700 font-semibold rounded shadow px-4 py-2"
        >
          Start ranking
        </button>
        <p v-else-if="count === 0" class="mb-8 text-sm text-gray-500">
          No pairs for that query — adjust range or area.
        </p>
      </div>
    </div>
  </div>
</template>

<script>
import AreaPickerMap from '../components/AreaPickerMap.vue'

// Default coordinates for common cities
const CITY_COORDINATES = {
  'berlin': { center: [52.5200, 13.4050], zoom: 12 },
  'sao_paulo': { center: [-23.5505, -46.6333], zoom: 12 },
  'washington': { center: [38.9072, -77.0369], zoom: 12 },
  'sydney': { center: [-33.8688, 151.2093], zoom: 12 },
  'cape_town': { center: [-33.9249, 18.4241], zoom: 12 },
  'taipei': { center: [25.0330, 121.5654], zoom: 12 }
}

export default {
  name: 'QueryPage',
  components: { AreaPickerMap },
  data() {
    return {
      cities: [], // Will be populated from API
      selectedCity: null,
      inner: null,
      outer: null,
      area: null,
      count: null,
      loading: false,
      loadingCities: false
    }
  },
  async created() {
    await this.loadCities()
  },
  methods: {
    async loadCities() {
      this.loadingCities = true
      try {
        const res = await fetch('http://localhost:8000/available-cities/')
        if (!res.ok) {
          console.error('Failed to load cities:', res.status)
          return
        }
        
        const citiesData = await res.json()
        
        // Transform API response to match our component format
        this.cities = citiesData.map(city => {
          const cityName = city.city // Base city name from API
          const coords = CITY_COORDINATES[cityName] || { center: [0, 0], zoom: 10 }
          
          return {
            name: cityName, // Use base name for queries
            displayName: this.formatCityName(cityName), // Formatted for display
            tableName: city.table_name, // Full table name from API
            center: coords.center,
            zoom: coords.zoom
          }
        })
        
        console.log('Loaded cities:', this.cities)
      } catch (error) {
        console.error('Error loading cities:', error)
      } finally {
        this.loadingCities = false
      }
    },
    
    formatCityName(cityName) {
      // Convert "cape_town" to "Cape Town", "sao_paulo" to "São Paulo", etc.
      return cityName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
        .replace(/Sao /, 'São ')
    },
    
    selectCity(city) {
      this.selectedCity = city
      this.area = null // Reset area when city changes
      this.count = null // Reset count when city changes
    },
    
    async runQuery() {
      if (!this.selectedCity) {
        alert('Please select a city first')
        return
      }
      
      this.loading = true
      try {
        const toNumOrNull = (v) => {
          const n = v === '' || v == null ? null : Number(v)
          return Number.isFinite(n) ? n : null
        }
        
        const payload = {
          city: this.selectedCity.name, // Send base city name to backend
          inner_buffer: toNumOrNull(this.inner),
          outer_buffer: toNumOrNull(this.outer),
          ...(this.area ? { area: this.area } : {})
        }
        
        const res = await fetch('http://localhost:8000/query/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}))
          console.error('/query failed', res.status, detail)
          throw new Error(`Request failed: ${res.status}`)
        }
        
        const data = await res.json()
        this.count = data.count ?? null
      } finally {
        this.loading = false
      }
    },
    
    goRank() {
      this.$router.push({
        name: 'rank',
        query: {
          city: this.selectedCity.name,
          inner: this.inner,
          outer: this.outer,
          count: this.count,
          ...(this.area ? {
            lng: this.area.center[0],
            lat: this.area.center[1],
            r: Math.round(this.area.radius_m)
          } : {})
        }
      })
    }
  }
}
</script>