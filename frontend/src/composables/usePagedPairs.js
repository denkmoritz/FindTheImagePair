import { ref, computed } from 'vue'

export function usePagedPairs() {
  const items = ref([])
  const total = ref(null)
  const cursor = ref(null)
  const loading = ref(false)

  const reachedEnd = computed(() => {
    if (total.value == null) return false
    return !cursor.value && items.value.length >= total.value && total.value > 0
  })

  async function loadFirst(limit = 50, userId = 'default') {
    items.value = []
    total.value = null
    cursor.value = null
    await loadMore(limit, userId)
  }

  async function loadMore(limit = 50, userId = 'default') {
    if (loading.value) return
    if (reachedEnd.value) return
    loading.value = true
    try {
      const body = {
        limit,
        user_id: userId, // backend accepts user_id OR userId; tweak if needed
      }
      if (cursor.value) body.cursor = cursor.value

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

      items.value.push(...mapped)
      total.value = data.total ?? total.value
      cursor.value = data.nextCursor ?? null
    } finally {
      loading.value = false
    }
  }

  return { items, total, cursor, loading, reachedEnd, loadFirst, loadMore }
}