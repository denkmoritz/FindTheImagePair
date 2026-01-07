<template>
  <div class="area-picker">
    <div class="toolbar">
      <button class="btn" @click="startDrawing" :disabled="drawing">Draw circle</button>
      <button class="btn" @click="clearCircle" :disabled="!circleLayer && !drawing">Clear area</button>
      <span class="hint" v-if="!drawing && !circleLayer">Tip: Click “Draw circle”, then click twice on the map.</span>
      <span class="hint" v-else-if="drawing">
        Click to set center, move to adjust radius, click again to finish.
        <span v-if="centerLat != null && hoverLat != null"> · r ≈ {{ Math.round(previewRadiusM) }} m</span>
      </span>
      <span class="hint" v-else>
        Center: {{ fmt(centerLat) }}, {{ fmt(centerLng) }} · Radius: {{ Math.round(radiusM) }} m
      </span>
    </div>
    <div ref="mapEl" class="map"></div>
  </div>
</template>

<script>

// eslint-disable-next-line
import L from 'leaflet'

export default {
  name: 'AreaPickerMap',
  props: {
    // Optional initial area: { center:[lng,lat], radius_m }
    modelValue: { type: Object, default: null },
    // Initial map view; defaults to BERLIN
    initialCenter: { type: Array, default: () => [52.5200, 13.4050] }, // [lat, lng]
    initialZoom: { type: Number, default: 12 }
  },
  emits: ['update:modelValue','change'],
  data() {
    return {
      map: null,
      circleLayer: null,
      centerLat: null, centerLng: null, radiusM: null,
      drawing: false,
      centerLatLngObj: null,
      tempCircle: null,
      hoverLat: null, hoverLng: null,
      previewRadiusM: 0
    }
  },
  mounted() {
    this.map = L.map(this.$refs.mapEl, { worldCopyJump: true })
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap'
    }).addTo(this.map)

    // Start in Berlin (or provided props)
    this.map.setView(this.initialCenter, this.initialZoom)

    // If initial area is provided, render it
    if (this.modelValue && this.modelValue.center && this.modelValue.radius_m) {
      const [lng, lat] = this.modelValue.center
      const circle = L.circle([lat, lng], { radius: this.modelValue.radius_m })
      circle.addTo(this.map)
      this.circleLayer = circle
      this.centerLat = lat; this.centerLng = lng; this.radiusM = this.modelValue.radius_m
      this.map.fitBounds(circle.getBounds(), { padding: [20,20] })
    }

    window.addEventListener('keydown', this.onKeydown)
  },
  beforeUnmount() { window.removeEventListener('keydown', this.onKeydown) },
  methods: {
    startDrawing() {
      if (this.drawing) return
      this.drawing = true
      if (this.circleLayer) { this.map.removeLayer(this.circleLayer); this.circleLayer = null }
      this._emit(null)
      this.map.once('click', this.onFirstClick)
      this.map.on('mousemove', this.onMouseMove)
      this.map.once('contextmenu', this.cancelDrawing)
    },
    onFirstClick(e) {
      this.centerLatLngObj = e.latlng
      this.centerLat = e.latlng.lat
      this.centerLng = e.latlng.lng
      this.tempCircle = L.circle(this.centerLatLngObj, { radius: 1, color: '#22c55e' }).addTo(this.map)
      this.map.once('click', this.onSecondClick)
    },
    onMouseMove(e) {
      if (!this.drawing || !this.centerLatLngObj) return
      this.hoverLat = e.latlng.lat; this.hoverLng = e.latlng.lng
      this.previewRadiusM = this.map.distance(this.centerLatLngObj, e.latlng)
      if (this.tempCircle) this.tempCircle.setRadius(this.previewRadiusM)
    },
    onSecondClick(e) {
      if (!this.centerLatLngObj) return
      const finalRadius = this.map.distance(this.centerLatLngObj, e.latlng)
      if (this.tempCircle) { this.map.removeLayer(this.tempCircle); this.tempCircle = null }
      this.circleLayer = L.circle(this.centerLatLngObj, { radius: finalRadius }).addTo(this.map)
      this.radiusM = finalRadius
      this._emit({ center: [this.centerLng, this.centerLat], radius_m: this.radiusM })
      this.map.off('mousemove', this.onMouseMove)
      this.drawing = false
      this.centerLatLngObj = null
      this.hoverLat = this.hoverLng = null
    },
    cancelDrawing() {
      this.map.off('mousemove', this.onMouseMove)
      this.map.off('click', this.onSecondClick)
      if (this.tempCircle) { this.map.removeLayer(this.tempCircle); this.tempCircle = null }
      this.drawing = false
      this.centerLatLngObj = null
      this.hoverLat = this.hoverLng = null
    },
    clearCircle() {
      this.cancelDrawing()
      if (this.circleLayer) { this.map.removeLayer(this.circleLayer); this.circleLayer = null }
      this.centerLat = this.centerLng = this.radiusM = null
      this._emit(null)
    },
    _emit(val) { this.$emit('update:modelValue', val); this.$emit('change', val) },
    onKeydown(e) { if (e.key === 'Escape' && this.drawing) this.cancelDrawing() },
    fmt(n) { return n == null ? '' : n.toFixed(5) }
  }
}
</script>

<style scoped>
.area-picker { border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; overflow: hidden; background: #0f172a22; }
.toolbar { display: flex; gap: 12px; align-items: center; padding: 10px 12px; background: rgba(0,0,0,.25); flex-wrap: wrap; }
.map { width: 100%; height: 500px; }
.btn { padding: 6px 10px; border: 1px solid #3b3b3b; border-radius: 8px; cursor: pointer; background: rgba(255,255,255,0.06); }
.btn:disabled { opacity: .6; cursor: not-allowed; }
.hint { font-size: 12px; opacity: .8; }
</style>