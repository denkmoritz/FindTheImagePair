<template>
  <section class="gallery">
    <div class="gallery__controls" v-if="showControls">
      <input
        v-model="q"
        class="gallery__search"
        type="search"
        placeholder="Search by caption or tag…"
        aria-label="Search images"
      />
    </div>

    <div ref="gridEl" class="gallery__grid" :style="gridStyle">
      <article
        v-for="(pair, i) in filteredPairs"
        :key="pair.id ?? i"
        class="card-wrapper"
        :aria-label="pair.caption || 'Image pair'"
      >
        <!-- Image + caption card -->
        <div class="card" :style="{'--row-h': rowHeight + 'px'}">
          <div class="card__media">
            <figure class="pair">
              <img
                class="pair__img"
                :src="pair.left.src"
                :alt="pair.left.alt || ''"
                loading="lazy"
              />
              <img
                class="pair__img"
                :src="pair.right.src"
                :alt="pair.right.alt || ''"
                loading="lazy"
              />
            </figure>
          </div>

          <footer
            class="card__meta"
            v-if="pair.caption || (pair.tags && pair.tags.length)"
          >
            <p class="card__caption" v-if="pair.caption">{{ pair.caption }}</p>
            <ul class="card__tags" v-if="pair.tags && pair.tags.length">
              <li v-for="t in pair.tags" :key="t" class="card__tag">#{{ t }}</li>
            </ul>
          </footer>
        </div>

        <!-- Actions BELOW card -->
        <div class="card__actions">
          <button class="card__btn" @click="zoom(i)">Zoom</button>
          <button class="card__btn" @click="map(pair)">Map</button>
        </div>
      </article>
    </div>

    <!-- Lightbox (Zoom modal) -->
    <dialog ref="dlg" class="lightbox" @click="onDialogBackdrop">
      <button
        class="lightbox__close"
        @click="closeLightbox"
        aria-label="Close"
      >×</button>
      <div class="lightbox__stage" v-if="activeIndex !== null">
        <img
          :src="filteredPairs[activeIndex].left.src"
          :alt="filteredPairs[activeIndex].left.alt || ''"
        />
        <img
          :src="filteredPairs[activeIndex].right.src"
          :alt="filteredPairs[activeIndex].right.alt || ''"
        />
        <p
          class="lightbox__caption"
          v-if="filteredPairs[activeIndex].caption"
        >
          {{ filteredPairs[activeIndex].caption }}
        </p>
        <div class="lightbox__nav">
          <button @click.stop="prev">‹</button>
          <button @click.stop="next">›</button>
        </div>
      </div>
    </dialog>
  </section>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { gsap } from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'
gsap.registerPlugin(ScrollTrigger)

const props = defineProps({
  pairs: { type: Array, required: true },
  columns: { type: Number, default: 1 },
  gap: { type: Number, default: 16 },
  showControls: { type: Boolean, default: false },
  rowHeight: { type: Number, default: 260 }
})

const emit = defineEmits(['map'])
function map(pair) { emit('map', pair) }

const q = ref('')
const filteredPairs = computed(() => {
  if (!q.value) return props.pairs
  const term = q.value.toLowerCase()
  return props.pairs.filter(p =>
    (p.caption && p.caption.toLowerCase().includes(term)) ||
    (p.tags && p.tags.some(t => t.toLowerCase().includes(term)))
  )
})

const gridEl = ref(null)
const gridStyle = computed(() => ({
  '--gap': props.gap + 'px',
  '--columns': String(props.columns || 1)
}))

let ctx = null
async function runIntroAnim() {
  await nextTick()
  const scope = gridEl.value
  if (!scope || !scope.isConnected) return
  const cards = Array.from(scope.querySelectorAll('.card-wrapper'))
  if (!cards.length) return

  if (ctx) ctx.revert()
  ctx = gsap.context(() => {
    gsap.set(cards, { autoAlpha: 0, y: 16 })
    gsap.to(cards, {
      autoAlpha: 1,
      y: 0,
      duration: 0.5,
      ease: 'power2.out',
      stagger: { each: 0.04, from: 'start' },
      scrollTrigger: {
        trigger: scope,
        start: 'top bottom-=10%',
        end: 'bottom top+=10%',
        toggleActions: 'play none none reverse'
      }
    })
  }, scope)
}

onMounted(runIntroAnim)
watch(filteredPairs, async () => { await runIntroAnim() })
onBeforeUnmount(() => { if (ctx) ctx.revert(); ctx = null })

/* Lightbox controls */
const dlg = ref(null)
const activeIndex = ref(null)
function zoom(i) { openLightbox(i) }
function openLightbox(i) {
  activeIndex.value = i
  if (dlg.value && !dlg.value.open) {
    dlg.value.showModal()
    document.addEventListener('keydown', onKey)
  }
}
function closeLightbox() {
  if (dlg.value && dlg.value.open) dlg.value.close()
  activeIndex.value = null
  document.removeEventListener('keydown', onKey)
}
function onKey(e) {
  if (e.key === 'Escape') closeLightbox()
  if (e.key === 'ArrowRight') next()
  if (e.key === 'ArrowLeft') prev()
}
function onDialogBackdrop(e) { if (e.target === dlg.value) closeLightbox() }
function next() {
  if (activeIndex.value === null) return
  activeIndex.value = (activeIndex.value + 1) % filteredPairs.value.length
}
function prev() {
  if (activeIndex.value === null) return
  activeIndex.value = (activeIndex.value - 1 + filteredPairs.value.length) % filteredPairs.value.length
}
</script>

<style scoped>
/* Layout */
.gallery { display: grid; gap: var(--gap, 16px); }
.gallery__controls { display: flex; justify-content: flex-end; }
.gallery__search {
  inline-size: min(420px, 100%);
  padding: .6rem .8rem;
  border: 1px solid hsl(0 0% 85%);
  border-radius: .6rem;
}

/* Grid */
.gallery__grid {
  display: grid;
  gap: var(--gap, 16px);
  grid-template-columns: repeat(var(--columns, 1), minmax(520px, 1fr));
}

/* Card wrapper keeps spacing for buttons */
.card-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: .6rem;
}

/* Card */
.card {
  background: white;
  border-radius: 14px;
  border: 1px solid hsl(0 0% 92%);
  box-shadow: 0 1px 0 hsl(0 0% 95%);
  overflow: hidden;
  transform-origin: center;
  will-change: transform, opacity;
  width: 100%;
}

.card__media {
  inline-size: 100%;
  block-size: var(--row-h, 260px);
}

.pair {
  margin: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  inline-size: 100%;
  block-size: 100%;
}
.pair__img {
  inline-size: 100%;
  block-size: 100%;
  object-fit: cover;
}

/* Meta */
.card__meta { padding: .75rem .9rem 0.5rem; }
.card__caption { margin: 0 0 .35rem; font-size: .95rem; color: hsl(0 0% 15%); }
.card__tags { display: flex; flex-wrap: wrap; gap: .4rem .5rem; list-style: none; margin: 0; padding: 0; }
.card__tag { font-size: .75rem; color: hsl(0 0% 35%); background: hsl(0 0% 96%); border: 1px solid hsl(0 0% 90%); padding: .15rem .5rem; border-radius: 999px; }

/* Actions */
.card__actions {
  display: flex;
  gap: .6rem;
  justify-content: center;
}
.card__btn {
  appearance: none;
  border: 1px solid hsl(210 10% 70%);
  background: hsl(210 90% 45%);
  color: white;
  padding: .38rem .8rem;
  border-radius: .5rem;
  font-size: .85rem;
  font-weight: 600;
  cursor: pointer;
}
.card__btn:hover { background: hsl(210 90% 40%); }
.card__btn:focus { outline: 2px solid hsl(210 90% 60%); outline-offset: 2px; }

/* Lightbox */
.lightbox {
  width: min(1200px, 96vw);
  margin: auto;
  padding: 0;
  border: none;
  background: transparent;
}
.lightbox::backdrop {
  background: rgba(0,0,0,.35);
  backdrop-filter: blur(6px);
}
.lightbox__stage {
  position: relative;
  background: #111;
  padding: 16px;
  border-radius: 14px;
  display: grid;
  gap: 10px;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: center;
  overflow: hidden;
}
.lightbox__stage img {
  max-width: 100%;
  max-height: 82vh;
  object-fit: contain;
  display: block;
  background: #000;
}
.lightbox__caption {
  color: #eee;
  margin: 4px 0 0;
  grid-column: 1/-1;
  text-align: center;
}
.lightbox__nav {
  position: absolute; inset: 0;
  display: flex; justify-content: space-between; align-items: center;
  pointer-events: none;
}
.lightbox__nav > button {
  pointer-events: auto;
  border: none;
  background: rgba(255,255,255,.15);
  color: white;
  font-size: 2rem;
  line-height: 1;
  padding: .2rem .6rem;
  border-radius: .5rem;
  margin: 0 .3rem;
}
.lightbox__close {
  position: absolute; top: 8px; right: 8px;
  width: 36px; height: 36px;
  border-radius: 50%; border: none;
  background: rgba(255,255,255,0.12);
  color: #fff;
  font-size: 20px; line-height: 36px; text-align: center;
  cursor: pointer;
  box-shadow: 0 2px 10px rgba(0,0,0,.35);
  backdrop-filter: blur(2px);
}
.lightbox__close:hover { background: rgba(255,255,255,0.18); }
.lightbox__close:focus-visible { outline: 2px solid rgba(255,255,255,0.45); outline-offset: 2px; }

/* Dark mode */
@media (prefers-color-scheme: dark) {
  .card { background: hsl(0 0% 9%); border-color: hsl(0 0% 20%); box-shadow: none; }
  .card__caption { color: hsl(0 0% 95%); }
  .card__tag { background: hsl(0 0% 16%); border-color: hsl(0 0% 24%); color: hsl(0 0% 80%); }
  .gallery__search { background: hsl(0 0% 10%); border-color: hsl(0 0% 25%); color: hsl(0 0% 92%); }
  .card__btn { background: hsl(210 90% 50%); border-color: hsl(210 20% 35%); }
  .card__btn:hover { background: hsl(210 90% 46%); }
}
</style>