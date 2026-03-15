<script setup lang="ts">
import type { PredictionMode } from '../api/types'
import InputRow from './InputRow.vue'
import BatchUpload from './BatchUpload.vue'
import type { InputRowEntry } from '../composables/useInputRows'

defineProps<{
  rows: InputRowEntry[]
  mode: PredictionMode
  disabled: boolean
}>()

const emit = defineEmits<{
  addRow: []
  removeRow: [id: number]
  updateField: [id: number, field: 'sequence' | 'pdbpath' | 'inhibitorSmiles', value: string]
  addSubstrate: [rowId: number]
  removeSubstrate: [rowId: number, subId: number]
  updateSubstrateSmiles: [rowId: number, subId: number, smiles: string]
  setPrimary: [rowId: number, subId: number]
  loadSample: []
  importCsv: [text: string]
  submit: []
}>()
</script>

<template>
  <form
    class="input-panel"
    novalidate
    :aria-busy="disabled"
    @submit.prevent="emit('submit')"
  >
    <h3 class="panel-title">Inputs</h3>

    <div class="rows-list">
      <InputRow
        v-for="(row, index) in rows"
        :key="row.id"
        :row="row"
        :mode="mode"
        :index="index"
        :show-header="rows.length > 1"
        :can-remove="rows.length > 1"
        @remove="(id) => emit('removeRow', id)"
        @update-field="(id, field, value) => emit('updateField', id, field, value)"
        @add-substrate="(rowId) => emit('addSubstrate', rowId)"
        @remove-substrate="(rowId, subId) => emit('removeSubstrate', rowId, subId)"
        @update-substrate-smiles="(rowId, subId, smiles) => emit('updateSubstrateSmiles', rowId, subId, smiles)"
        @set-primary="(rowId, subId) => emit('setPrimary', rowId, subId)"
      />
    </div>

    <BatchUpload @import="(text) => emit('importCsv', text)" />

    <div class="actions">
      <button
        type="button"
        class="btn btn-ghost"
        :disabled="disabled"
        @click="emit('addRow')"
      >Add row</button>
      <button
        type="button"
        class="btn btn-ghost"
        :disabled="disabled"
        @click="emit('loadSample')"
      >Load sample</button>
      <button
        type="submit"
        class="btn btn-primary"
        :disabled="disabled"
        :aria-busy="disabled"
      >
        <span v-if="disabled" class="spinner" aria-hidden="true"></span>
        {{ disabled ? 'Running...' : 'Run prediction' }}
      </button>
    </div>

    <p v-if="mode === 'substrate'" class="helper-text">
      Add substrates per entry. The <strong>primary</strong> substrate is used for Km;
      all substrates (joined) are used for kcat.
    </p>
    <p v-else class="helper-text">
      Enter the inhibitor compound SMILES, enzyme sequence, and a Sequence ID.
    </p>
  </form>
</template>

<style scoped>
.input-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem;
  box-shadow: var(--shadow-sm);
}

.panel-title {
  font-family: var(--font-serif);
  font-size: 1.25rem;
  font-weight: 400;
  color: var(--text);
  margin-bottom: 0.75rem;
}

.rows-list {
  display: grid;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.375rem;
  height: 40px;
  padding: 0 1rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 500;
  transition: all 0.15s;
  min-width: 44px;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-ghost {
  background: transparent;
  color: var(--text-secondary);
}

.btn-ghost:hover:not(:disabled) {
  background: var(--bg-muted);
  color: var(--text);
}

.btn-primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
  box-shadow: 0 1px 3px rgba(5, 150, 105, 0.2);
}

.btn-primary:hover:not(:disabled) {
  background: var(--accent-hover);
  border-color: var(--accent-hover);
}

.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.helper-text {
  margin-top: 0.75rem;
  font-size: 0.8rem;
  color: var(--text-tertiary);
  line-height: 1.5;
}

.helper-text strong {
  font-weight: 600;
  color: var(--text-secondary);
}
</style>
