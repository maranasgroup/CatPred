<script setup lang="ts">
import { ref } from 'vue'
import type { SubstrateEntry } from '../api/types'
import { validateSmiles } from '../composables/useInputRows'

defineProps<{
  substrates: SubstrateEntry[]
  rowId: number
}>()

const emit = defineEmits<{
  addSubstrate: [rowId: number]
  removeSubstrate: [rowId: number, subId: number]
  updateSmiles: [rowId: number, subId: number, smiles: string]
  setPrimary: [rowId: number, subId: number]
}>()

const errors = ref<Record<number, string>>({})

function onBlur(sub: SubstrateEntry) {
  if (sub.smiles.trim()) {
    errors.value[sub.id] = validateSmiles(sub.smiles)
  } else {
    errors.value[sub.id] = ''
  }
}
</script>

<template>
  <div class="substrate-inputs">
    <label class="field-label">Substrates</label>
    <div class="substrate-list">
      <div v-for="sub in substrates" :key="sub.id" class="substrate-row">
        <label
          class="primary-radio"
          :title="sub.isPrimary ? 'Primary substrate (used for Km)' : 'Set as primary (for Km)'"
        >
          <input
            type="radio"
            :name="`primary-${rowId}`"
            :checked="sub.isPrimary"
            @change="emit('setPrimary', rowId, sub.id)"
          />
          <span class="primary-label">{{ sub.isPrimary ? 'Primary' : '' }}</span>
        </label>
        <div class="sub-field">
          <input
            type="text"
            :value="sub.smiles"
            placeholder="SMILES"
            :aria-label="`Substrate SMILES`"
            :aria-invalid="errors[sub.id] ? 'true' : undefined"
            @input="emit('updateSmiles', rowId, sub.id, ($event.target as HTMLInputElement).value)"
            @blur="onBlur(sub)"
          />
          <p v-if="errors[sub.id]" class="field-error" role="alert">{{ errors[sub.id] }}</p>
        </div>
        <button
          v-if="substrates.length > 1"
          type="button"
          class="remove-sub-btn"
          :aria-label="`Remove substrate`"
          @click="emit('removeSubstrate', rowId, sub.id)"
        >&times;</button>
      </div>
    </div>
    <button
      type="button"
      class="add-sub-btn"
      @click="emit('addSubstrate', rowId)"
    >+ Add substrate</button>
  </div>
</template>

<style scoped>
.field-label {
  display: block;
  margin-bottom: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}

.substrate-list {
  display: grid;
  gap: 0.375rem;
}

.substrate-row {
  display: flex;
  align-items: flex-start;
  gap: 0.375rem;
}

.primary-radio {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
  min-width: 70px;
  padding-top: 0.5rem;
  cursor: pointer;
}

.primary-radio input[type="radio"] {
  width: 14px;
  height: 14px;
  accent-color: var(--accent);
  cursor: pointer;
}

.primary-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  font-weight: 500;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;
}

.sub-field {
  flex: 1;
  min-width: 0;
}

.sub-field input {
  width: 100%;
  padding: 0.5rem 0.625rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  font-size: 0.875rem;
  line-height: 1.5;
  transition: border-color 0.15s;
}

.sub-field input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--focus-ring);
}

.sub-field input[aria-invalid="true"] {
  border-color: var(--danger);
}

.remove-sub-btn {
  flex-shrink: 0;
  width: 28px;
  height: 34px;
  margin-top: 0.125rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 1rem;
  color: var(--text-tertiary);
  transition: all 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.remove-sub-btn:hover {
  color: var(--danger);
  border-color: rgba(220, 38, 38, 0.3);
  background: rgba(220, 38, 38, 0.04);
}

.add-sub-btn {
  margin-top: 0.25rem;
  padding: 0.25rem 0.5rem;
  border: 1px dashed var(--border);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 500;
  color: var(--text-tertiary);
  transition: all 0.15s;
}

.add-sub-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(5, 150, 105, 0.04);
}

.field-error {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: var(--danger);
}
</style>
