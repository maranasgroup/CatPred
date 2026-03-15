<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import type { PredictionMode } from './api/types'
import { useReadiness } from './composables/useReadiness'
import { usePrediction } from './composables/usePrediction'
import { useInputRows } from './composables/useInputRows'
import SkipLink from './components/SkipLink.vue'
import AppFooter from './components/AppFooter.vue'
import ParameterSelector from './components/ParameterSelector.vue'
import InputPanel from './components/InputPanel.vue'
import StatusBar from './components/StatusBar.vue'
import ResultPanel from './components/ResultPanel.vue'

const readiness = useReadiness()
const prediction = usePrediction()
const inputs = useInputRows()

const mode = ref<PredictionMode>('substrate')
const predictKcat = ref(true)
const predictKm = ref(true)
const importError = ref('')

onMounted(() => {
  readiness.check()
})

// Reset form when mode changes
watch(mode, () => {
  inputs.clear()
  importError.value = ''
})

function buildPayload(parameter: string, rows: ReturnType<typeof inputs.collectRowsForParameter>) {
  const backend = readiness.chooseBackend()
  return {
    parameter,
    checkpoint_dir: parameter,
    input_rows: rows,
    use_gpu: false,
    results_dir: 'web-app',
    backend,
    fallback_to_local: readiness.shouldFallback(backend, parameter as 'kcat' | 'km' | 'ki'),
  }
}

function onSubmit() {
  importError.value = ''
  const validationError = inputs.validateAll(mode.value)
  if (validationError) {
    importError.value = validationError
    return
  }

  const jobs: { parameter: 'kcat' | 'km' | 'ki'; payload: ReturnType<typeof buildPayload> }[] = []

  if (mode.value === 'substrate') {
    if (predictKcat.value && readiness.isParameterAvailable('kcat')) {
      const rows = inputs.collectRowsForParameter('substrate', 'kcat')
      jobs.push({ parameter: 'kcat', payload: buildPayload('kcat', rows) })
    }
    if (predictKm.value && readiness.isParameterAvailable('km')) {
      const rows = inputs.collectRowsForParameter('substrate', 'km')
      jobs.push({ parameter: 'km', payload: buildPayload('km', rows) })
    }
  } else {
    if (!readiness.isParameterAvailable('ki')) {
      importError.value = 'No checkpoint available for Ki.'
      return
    }
    const rows = inputs.collectRowsForParameter('inhibition', 'ki')
    jobs.push({ parameter: 'ki', payload: buildPayload('ki', rows) })
  }

  if (jobs.length === 0) {
    importError.value = 'No available parameters to predict. Check server status.'
    return
  }

  prediction.runAll(jobs)
}

function onImportCsv(text: string) {
  const err = inputs.importCsv(text, mode.value)
  importError.value = err
}
</script>

<template>
  <SkipLink />

  <main id="main-content" class="main">
    <div class="container">
      <div class="page-header">
        <div class="page-header-left">
          <h1 class="page-title">CatPred</h1>
          <nav class="page-links" aria-label="Project links">
            <a
              href="https://www.nature.com/articles/s41467-025-57215-9"
              target="_blank"
              rel="noreferrer"
              class="page-link"
            >
              Paper
              <span class="link-arrow" aria-hidden="true">&nearr;</span>
            </a>
            <span class="link-sep">&middot;</span>
            <a
              href="https://github.com/maranasgroup/catpred/"
              target="_blank"
              rel="noreferrer"
              class="page-link"
            >
              GitHub
              <span class="link-arrow" aria-hidden="true">&nearr;</span>
            </a>
          </nav>
        </div>
        <StatusBar
          :service-status="readiness.status.value"
          :service-hint="readiness.statusHint.value"
          :prediction-status="prediction.status.value"
          :prediction-error="prediction.error.value"
          :elapsed="prediction.elapsedSeconds.value"
          :can-retry="prediction.lastJobs.value !== null && prediction.status.value === 'error'"
          @retry="prediction.retry()"
        />
      </div>

      <div class="param-row">
        <ParameterSelector
          :mode="mode"
          :predict-kcat="predictKcat"
          :predict-km="predictKm"
          :available="readiness.availableCheckpoints.value"
          @update:mode="mode = $event"
          @update:predict-kcat="predictKcat = $event"
          @update:predict-km="predictKm = $event"
        />
      </div>

      <p v-if="importError" class="form-error" role="alert">{{ importError }}</p>

      <div class="grid">
        <InputPanel
          :rows="inputs.rows.value"
          :mode="mode"
          :disabled="prediction.isRunning.value"
          @add-row="inputs.addRow()"
          @remove-row="inputs.removeRow($event)"
          @update-field="(id, field, value) => inputs.updateField(id, field, value)"
          @add-substrate="inputs.addSubstrate($event)"
          @remove-substrate="(rowId, subId) => inputs.removeSubstrate(rowId, subId)"
          @update-substrate-smiles="(rowId, subId, smiles) => inputs.updateSubstrateSmiles(rowId, subId, smiles)"
          @set-primary="(rowId, subId) => inputs.setPrimary(rowId, subId)"
          @load-sample="inputs.loadSample(mode)"
          @import-csv="onImportCsv"
          @submit="onSubmit"
        />
        <ResultPanel
          :results="prediction.results.value"
        />
      </div>
    </div>
  </main>

  <AppFooter />
</template>

<style scoped>
.main {
  padding: 2rem 0 3rem;
}

.page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.page-title {
  font-family: var(--font-serif);
  font-size: clamp(1.5rem, 3vw, 2rem);
  font-weight: 400;
  letter-spacing: -0.02em;
  color: var(--text);
}

.page-links {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-shrink: 0;
}

.page-link {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.875rem;
  font-weight: 450;
  color: var(--text-secondary);
  text-decoration: none;
  padding-bottom: 1px;
  border-bottom: 1px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.page-link:hover {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

.link-arrow {
  font-size: 0.8em;
  opacity: 0.6;
}

.page-link:hover .link-arrow {
  opacity: 1;
}

.link-sep {
  color: var(--text-tertiary);
  font-size: 0.75rem;
}

@media (max-width: 640px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
}

.param-row {
  margin: 0.75rem 0;
}

.form-error {
  margin-bottom: 0.75rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid rgba(220, 38, 38, 0.3);
  border-radius: var(--radius-sm);
  background: rgba(220, 38, 38, 0.04);
  color: var(--danger);
  font-size: 0.8rem;
}

.grid {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 1rem;
  align-items: start;
}

@media (max-width: 1024px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
