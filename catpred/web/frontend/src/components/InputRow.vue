<script setup lang="ts">
import { ref } from 'vue'
import type { PredictionMode } from '../api/types'
import type { InputRowEntry } from '../composables/useInputRows'
import { validateSmiles, validateSequence, validatePdbpath } from '../composables/useInputRows'
import SubstrateInputs from './SubstrateInputs.vue'

defineProps<{
  row: InputRowEntry
  mode: PredictionMode
  index: number
  showHeader: boolean
  canRemove: boolean
}>()

const emit = defineEmits<{
  remove: [id: number]
  updateField: [id: number, field: 'sequence' | 'pdbpath' | 'inhibitorSmiles', value: string]
  addSubstrate: [rowId: number]
  removeSubstrate: [rowId: number, subId: number]
  updateSubstrateSmiles: [rowId: number, subId: number, smiles: string]
  setPrimary: [rowId: number, subId: number]
}>()

const inhibitorError = ref('')
const sequenceError = ref('')
const pdbpathError = ref('')

function onBlurInhibitor(e: Event) {
  const val = (e.target as HTMLInputElement).value
  inhibitorError.value = validateSmiles(val)
}

function onBlurSequence(e: Event) {
  const val = (e.target as HTMLTextAreaElement).value
  sequenceError.value = validateSequence(val)
}

function onBlurPdbpath(e: Event) {
  const val = (e.target as HTMLInputElement).value
  pdbpathError.value = validatePdbpath(val)
}
</script>

<template>
  <article class="row-item">
    <div v-if="showHeader" class="row-head">
      <h4 class="row-title">Entry {{ index + 1 }}</h4>
      <button
        v-if="canRemove"
        type="button"
        class="remove-btn"
        :aria-label="`Remove entry ${index + 1}`"
        @click="emit('remove', row.id)"
      >Remove</button>
    </div>

    <div class="row-fields">
      <div class="field">
        <label :for="`seq-${row.id}`" class="field-label">Enzyme sequence</label>
        <textarea
          :id="`seq-${row.id}`"
          :value="row.sequence"
          rows="2"
          placeholder="ACDEFGHIK"
          required
          :aria-invalid="sequenceError ? 'true' : undefined"
          :aria-describedby="sequenceError ? `seq-err-${row.id}` : undefined"
          @input="emit('updateField', row.id, 'sequence', ($event.target as HTMLTextAreaElement).value)"
          @blur="onBlurSequence"
        ></textarea>
        <p v-if="sequenceError" :id="`seq-err-${row.id}`" class="field-error" role="alert">{{ sequenceError }}</p>
      </div>

      <div class="field">
        <label :for="`pdbpath-${row.id}`" class="field-label">Sequence ID</label>
        <input
          :id="`pdbpath-${row.id}`"
          type="text"
          :value="row.pdbpath"
          placeholder="seq_001"
          required
          :aria-invalid="pdbpathError ? 'true' : undefined"
          :aria-describedby="pdbpathError ? `pdbpath-err-${row.id}` : undefined"
          @input="emit('updateField', row.id, 'pdbpath', ($event.target as HTMLInputElement).value)"
          @blur="onBlurPdbpath"
        />
        <p v-if="pdbpathError" :id="`pdbpath-err-${row.id}`" class="field-error" role="alert">{{ pdbpathError }}</p>
      </div>

      <!-- Substrate mode: multi-substrate inputs -->
      <SubstrateInputs
        v-if="mode === 'substrate'"
        :substrates="row.substrates"
        :row-id="row.id"
        @add-substrate="(rowId) => emit('addSubstrate', rowId)"
        @remove-substrate="(rowId, subId) => emit('removeSubstrate', rowId, subId)"
        @update-smiles="(rowId, subId, smiles) => emit('updateSubstrateSmiles', rowId, subId, smiles)"
        @set-primary="(rowId, subId) => emit('setPrimary', rowId, subId)"
      />

      <!-- Inhibition mode: single inhibitor SMILES -->
      <div v-else class="field">
        <label :for="`inhibitor-${row.id}`" class="field-label">Inhibitor (SMILES)</label>
        <input
          :id="`inhibitor-${row.id}`"
          type="text"
          :value="row.inhibitorSmiles"
          placeholder="CCO"
          required
          :aria-invalid="inhibitorError ? 'true' : undefined"
          :aria-describedby="inhibitorError ? `inhibitor-err-${row.id}` : undefined"
          @input="emit('updateField', row.id, 'inhibitorSmiles', ($event.target as HTMLInputElement).value)"
          @blur="onBlurInhibitor"
        />
        <p v-if="inhibitorError" :id="`inhibitor-err-${row.id}`" class="field-error" role="alert">{{ inhibitorError }}</p>
      </div>
    </div>
  </article>
</template>

<style scoped>
.row-item {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  padding: 0.75rem;
}

.row-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.row-title {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 500;
  color: var(--text-tertiary);
  letter-spacing: 0.02em;
}

.remove-btn {
  height: 28px;
  padding: 0 0.6rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  color: var(--danger);
  transition: background 0.15s, border-color 0.15s;
}

.remove-btn:hover {
  background: rgba(220, 38, 38, 0.06);
  border-color: rgba(220, 38, 38, 0.3);
}

.row-fields {
  display: grid;
  gap: 0.5rem;
}

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

input,
textarea {
  width: 100%;
  padding: 0.5rem 0.625rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  font-size: 0.875rem;
  line-height: 1.5;
  transition: border-color 0.15s;
}

input:focus,
textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--focus-ring);
}

input[aria-invalid="true"],
textarea[aria-invalid="true"] {
  border-color: var(--danger);
}

input[aria-invalid="true"]:focus,
textarea[aria-invalid="true"]:focus {
  box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.3);
}

textarea {
  resize: vertical;
  min-height: 48px;
}

.field-error {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: var(--danger);
}
</style>
