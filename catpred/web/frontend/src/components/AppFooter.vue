<script setup lang="ts">
import { ref, onMounted } from 'vue'

const visitCount = ref<string | null>(null)

onMounted(async () => {
  try {
    const res = await fetch('https://catpred.goatcounter.com/counter/TOTAL.json')
    if (res.ok) {
      const data = await res.json()
      visitCount.value = data.count_unique ?? data.count ?? null
    }
  } catch {
    // silently ignore — counter just won't show
  }
})
</script>

<template>
  <footer class="footer">
    <div class="container footer-inner">
      <p class="footer-text">
        <span class="footer-brand">CatPred</span>
        <span class="footer-sep">&middot;</span>
        Developed in
        <a href="https://www.maranasgroup.com/" target="_blank" rel="noreferrer" class="footer-link">Maranas Group</a>
        at Penn State
      </p>
      <p v-if="visitCount" class="visit-count">{{ visitCount }} visits</p>
    </div>
  </footer>
</template>

<style scoped>
.footer {
  border-top: 1px solid var(--border);
  padding: 1.5rem 0;
}

.footer-inner {
  text-align: center;
}

.footer-text {
  color: var(--text-tertiary);
  font-size: 0.8rem;
}

.footer-brand {
  font-family: var(--font-serif);
  color: var(--text-secondary);
}

.footer-sep {
  margin: 0 0.25rem;
  opacity: 0.5;
}

.footer-link {
  color: var(--text-secondary);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.footer-link:hover {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

.visit-count {
  margin-top: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-tertiary);
  opacity: 0.7;
}
</style>
