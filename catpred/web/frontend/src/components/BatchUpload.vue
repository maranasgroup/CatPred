<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  import: [csvText: string]
}>()

const isDragging = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

function onDragOver(e: DragEvent) {
  e.preventDefault()
  isDragging.value = true
}

function onDragLeave() {
  isDragging.value = false
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  isDragging.value = false
  const file = e.dataTransfer?.files[0]
  if (file) readFile(file)
}

function onFileChange(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) readFile(file)
}

function readFile(file: File) {
  if (!file.name.endsWith('.csv') && file.type !== 'text/csv') return
  const reader = new FileReader()
  reader.onload = () => {
    if (typeof reader.result === 'string') {
      emit('import', reader.result)
    }
  }
  reader.readAsText(file)
}

function onPaste(e: ClipboardEvent) {
  const text = e.clipboardData?.getData('text/plain')
  if (text && text.includes(',') && text.includes('\n')) {
    e.preventDefault()
    emit('import', text)
  }
}

function openFilePicker() {
  fileInput.value?.click()
}
</script>

<template>
  <div
    :class="['upload-zone', { dragging: isDragging }]"
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
    @paste="onPaste"
    tabindex="0"
    role="button"
    aria-label="Upload CSV file or paste CSV data"
    @click="openFilePicker"
    @keydown.enter="openFilePicker"
    @keydown.space.prevent="openFilePicker"
  >
    <input
      ref="fileInput"
      type="file"
      accept=".csv,text/csv"
      class="sr-only"
      @change="onFileChange"
    />
    <p class="upload-text">
      <strong>Drop CSV here</strong> or click to browse
    </p>
    <p class="upload-hint">Columns: SMILES, sequence, pdbpath</p>
  </div>
</template>

<style scoped>
.upload-zone {
  border: 1px dashed var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}

.upload-zone:hover,
.upload-zone.dragging {
  border-color: var(--accent);
  background: rgba(5, 150, 105, 0.04);
}

.upload-text {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.upload-text strong {
  font-weight: 500;
  color: var(--text);
}

.upload-hint {
  margin-top: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  color: var(--text-tertiary);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
</style>
