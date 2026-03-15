<script setup lang="ts">
import { computed } from 'vue'
import type { Parameter, PredictionMode } from '../api/types'

const props = defineProps<{
  mode: PredictionMode
  predictKcat: boolean
  predictKm: boolean
  available: Set<Parameter>
}>()

const emit = defineEmits<{
  'update:mode': [value: PredictionMode]
  'update:predictKcat': [value: boolean]
  'update:predictKm': [value: boolean]
}>()

const substrateAvailable = computed(() => props.available.has('kcat') || props.available.has('km'))
const inhibitionAvailable = computed(() => props.available.has('ki'))

function selectMode(m: PredictionMode) {
  if (m === 'substrate' && !substrateAvailable.value) return
  if (m === 'inhibition' && !inhibitionAvailable.value) return
  emit('update:mode', m)
}

function toggleKcat() {
  const newVal = !props.predictKcat
  if (!newVal && !props.predictKm) return
  emit('update:predictKcat', newVal)
}

function toggleKm() {
  const newVal = !props.predictKm
  if (!newVal && !props.predictKcat) return
  emit('update:predictKm', newVal)
}
</script>

<template>
  <div class="mode-selector">
    <div class="mode-pills" role="radiogroup" aria-label="Prediction mode">
      <button
        type="button"
        role="radio"
        :aria-checked="mode === 'substrate'"
        :disabled="!substrateAvailable"
        :class="['mode-pill', 'mode-substrate', { active: mode === 'substrate' }]"
        :tabindex="mode === 'substrate' ? 0 : -1"
        @click="selectMode('substrate')"
      >kcat / Km</button>
      <button
        type="button"
        role="radio"
        :aria-checked="mode === 'inhibition'"
        :disabled="!inhibitionAvailable"
        :class="['mode-pill', 'mode-inhibition', { active: mode === 'inhibition' }]"
        :tabindex="mode === 'inhibition' ? 0 : -1"
        @click="selectMode('inhibition')"
      >Ki</button>
    </div>

    <div v-if="mode === 'substrate'" class="param-checks">
      <label class="check-label" :class="{ disabled: !available.has('kcat') }">
        <input
          type="checkbox"
          :checked="predictKcat"
          :disabled="!available.has('kcat') || (predictKcat && !predictKm)"
          @change="toggleKcat"
        />
        <span>kcat</span>
      </label>
      <label class="check-label" :class="{ disabled: !available.has('km') }">
        <input
          type="checkbox"
          :checked="predictKm"
          :disabled="!available.has('km') || (predictKm && !predictKcat)"
          @change="toggleKm"
        />
        <span>Km</span>
      </label>
    </div>
  </div>
</template>

<style scoped>
.mode-selector {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.mode-pills {
  display: flex;
  gap: 0.375rem;
}

.mode-pill {
  height: 34px;
  padding: 0 0.875rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-surface);
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 500;
  letter-spacing: 0.03em;
  color: var(--text-secondary);
  transition: all 0.15s;
}

.mode-pill:hover:not(:disabled) {
  border-color: var(--text-tertiary);
}

.mode-pill:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.mode-substrate.active {
  color: var(--color-kcat);
  border-color: var(--color-kcat);
  background: rgba(5, 150, 105, 0.08);
}

.mode-inhibition.active {
  color: var(--color-ki);
  border-color: var(--color-ki);
  background: rgba(139, 127, 199, 0.08);
}

.param-checks {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.check-label {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-secondary);
  cursor: pointer;
  user-select: none;
}

.check-label.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.check-label input[type="checkbox"] {
  width: 14px;
  height: 14px;
  accent-color: var(--accent);
  cursor: inherit;
}
</style>
