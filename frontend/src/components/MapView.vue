<template>
  <div class="map-wrapper">
    <div ref="mapEl" class="map-element"></div>

    <!-- Modal with blurred background -->
    <div
      v-if="showModal"
      class="fixed inset-0 backdrop-blur-sm bg-white/30 flex justify-center items-center text-black"
      style="z-index: 10000;"
    >
      <div class="bg-white p-6 rounded-lg shadow-lg max-w-5xl w-full">
        <h2 class="text-xl font-semibold mb-6 text-center">Image Pair</h2>
        <div class="flex gap-6 justify-center">
          <img :src="imageA" alt="Image A" class="max-w-md rounded border" />
          <img :src="imageB" alt="Image B" class="max-w-md rounded border" />
        </div>
        <div class="flex justify-center mt-6">
          <button
            @click="showModal = false"
            class="px-6 py-2 bg-gray-200 text-black rounded hover:bg-gray-300"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref, nextTick } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

/* ------- Modal state ------- */
const showModal = ref(false)
const imageA = ref('')
const imageB = ref('')

/* ------- Leaflet setup ------- */
const DefaultIcon = L.icon({
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
})
L.Marker.prototype.options.icon = DefaultIcon

const mapEl = ref(null)
const map = ref(null)
const overlays = ref([]) // markers + line

function clear() {
  console.log('Clearing map overlays')
  overlays.value.forEach(layer => {
    if (map.value?.hasLayer(layer)) map.value.removeLayer(layer)
  })
  overlays.value = []
}

async function showPair(pair) {
  console.log('MapView showPair called with:', pair)
  
  if (!map.value) {
    console.error('Map not initialized yet')
    return
  }

  // Ensure map is properly sized
  await nextTick()
  setTimeout(() => {
    if (map.value) {
      map.value.invalidateSize()
      console.log('Map size invalidated')
    }
  }, 50)

  // Get coordinates
  const lat1 = getNum(pair.left?.lat ?? pair.lat_1)
  const lon1 = getNum(pair.left?.lng ?? pair.lon_1 ?? pair.left?.lon)
  const lat2 = getNum(pair.right?.lat ?? pair.lat_2)
  const lon2 = getNum(pair.right?.lng ?? pair.lon_2 ?? pair.right?.lon)

  console.log('Parsed coordinates:', { lat1, lon1, lat2, lon2 })

  if (![lat1, lon1, lat2, lon2].every(n => typeof n === 'number' && !Number.isNaN(n))) {
    console.error('Invalid coordinates:', { lat1, lon1, lat2, lon2 })
    alert('Cannot display map: coordinates are missing or invalid')
    return
  }

  // Clear previous overlays
  clear()

  // Add markers
  console.log('Adding marker 1 at:', [lat1, lon1])
  const m1 = L.marker([lat1, lon1])
    .addTo(map.value)
    .bindPopup('<b>Left Image</b><br>Click to view')
  
  console.log('Adding marker 2 at:', [lat2, lon2])
  const m2 = L.marker([lat2, lon2])
    .addTo(map.value)
    .bindPopup('<b>Right Image</b><br>Click to view')

  // Add line connecting them
  console.log('Adding line between markers')
  const line = L.polyline(
    [[lat1, lon1], [lat2, lon2]], 
    { weight: 3, color: '#3388ff', opacity: 0.7 }
  ).addTo(map.value)

  overlays.value.push(m1, m2, line)
  console.log('Added overlays:', overlays.value.length)

  // Fit bounds to show both markers
  const bounds = L.latLngBounds([[lat1, lon1], [lat2, lon2]])
  console.log('Fitting bounds:', bounds)
  map.value.fitBounds(bounds, { padding: [100, 100], maxZoom: 18 })

  // Set up modal images
  if (pair.uuid && pair.relation_uuid) {
    imageA.value = `http://localhost:8000/image/${pair.uuid}`
    imageB.value = `http://localhost:8000/image/${pair.relation_uuid}`
  } else {
    imageA.value = pair.left?.src || ''
    imageB.value = pair.right?.src || ''
  }

  // Click markers to open modal
  m1.on('click', () => {
    console.log('Marker 1 clicked')
    showModal.value = true
  })
  m2.on('click', () => {
    console.log('Marker 2 clicked')
    showModal.value = true
  })
  
  console.log('showPair completed successfully')
}

function setPairs(locations) {
  if (!map.value || !Array.isArray(locations) || !locations.length) return
  clear()
  const layers = []
  locations.forEach(pair => {
    const lat1 = getNum(pair.lat_1), lon1 = getNum(pair.lon_1)
    const lat2 = getNum(pair.lat_2), lon2 = getNum(pair.lon_2)
    if ([lat1, lon1, lat2, lon2].some(n => Number.isNaN(n))) return

    const m1 = L.marker([lat1, lon1]).addTo(map.value).bindTooltip('Click to view pair', { direction: 'top' })
    m1.on('click', () => {
      imageA.value = `http://localhost:8000/image/${pair.uuid}`
      imageB.value = `http://localhost:8000/image/${pair.relation_uuid}`
      showModal.value = true
    })
    const m2 = L.marker([lat2, lon2]).addTo(map.value)
    const line = L.polyline([[lat1, lon1], [lat2, lon2]], { weight: 2 }).addTo(map.value)
    layers.push(m1, m2, line)
  })
  overlays.value = layers
  const all = locations.flatMap(p => [[p.lat_1, p.lon_1], [p.lat_2, p.lon_2]])
  map.value.fitBounds(L.latLngBounds(all), { padding: [30, 30], maxZoom: 16 })
}

function getNum(x) { 
  const n = Number(x)
  return Number.isFinite(n) ? n : NaN
}

onMounted(() => {
  nextTick(() => {
    if (!mapEl.value) {
      console.error('Map element not found')
      return
    }
    
    console.log('Initializing map')
    // Initialize with world view - will zoom to pair when showPair is called
    map.value = L.map(mapEl.value, {
      center: [0, 0],
      zoom: 2,
      zoomControl: true
    })
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19
    }).addTo(map.value)

    console.log('Map initialized successfully')

    // Ensure proper sizing
    setTimeout(() => {
      if (map.value) {
        map.value.invalidateSize()
      }
    }, 300)
  })
})

defineExpose({
  showPair,
  clear,
  setPairs
})
</script>

<style scoped>
.map-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  width: 100%;
  height: 100%;
}

.map-element {
  width: 100%;
  height: 100%;
}
</style>

<style>
.leaflet-container { 
  font: inherit;
  width: 100%;
  height: 100%;
}
</style>