<template>
  <div class="pair-tile">
    <div class="tile-card">
      <div class="imgs" @click="$emit('open', pair)">
        <img 
          :src="pair.left.src" 
          alt="Left image"
          @error="onImgError('left', $event)"
          @load="onImgLoad('left')"
        />
        <img 
          :src="pair.right.src" 
          alt="Right image"
          @error="onImgError('right', $event)"
          @load="onImgLoad('right')"
        />
      </div>
      <div class="meta">
        <button
          class="like-btn"
          :class="{ liked: pair.liked }"
          @click.stop="toggleLike"
          :title="pair.liked ? 'Unlike' : 'Like'"
        >
          {{ pair.liked ? '❤️ Liked' : '🤍 Like' }}
        </button>
        <button class="map" @click.stop="$emit('map', pair)">Map</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PairTile',
  props: {
    pair: { type: Object, required: true }
  },
  emits: ['like', 'open', 'map'],
  methods: {
    toggleLike() {
      this.$emit('like', { id: this.pair.id, liked: !this.pair.liked })
    },
    onImgError(side, event) {
      console.error(`Failed to load ${side} image:`, this.pair[side].src)
      console.error('Error event:', event)
    },
    onImgLoad(side) {
      console.log(`Successfully loaded ${side} image:`, this.pair[side].src)
    }
  }
}
</script>

<style scoped>
.pair-tile { 
  padding: 16px 0;
}

.tile-card {
  background: rgba(20, 30, 40, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
  transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
}

.tile-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
  border-color: rgba(255, 255, 255, 0.15);
}

.imgs { 
  display: grid; 
  grid-template-columns: 1fr 1fr; 
  gap: 8px; 
  width: 100%;
  margin-bottom: 12px;
}

.imgs img { 
  width: 100%; 
  height: auto;
  min-height: 200px;
  aspect-ratio: 4/3; 
  object-fit: cover; 
  border-radius: 8px;
  background: rgba(40, 50, 60, 0.5);
  display: block;
}

.meta { 
  display: flex; 
  justify-content: space-between; 
  gap: 8px; 
}

.like-btn, .map { 
  cursor: pointer;
  padding: 8px 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  background: rgba(30, 40, 50, 0.7);
  color: #fff;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.like-btn:hover, .map:hover {
  background: rgba(40, 50, 60, 0.9);
  border-color: rgba(255, 255, 255, 0.3);
  transform: translateY(-1px);
}

.like-btn.liked { 
  background: rgba(220, 38, 38, 0.2);
  border-color: rgba(220, 38, 38, 0.5);
  color: #ff6b6b;
}

.like-btn.liked:hover {
  background: rgba(220, 38, 38, 0.3);
  border-color: rgba(220, 38, 38, 0.6);
}
</style>