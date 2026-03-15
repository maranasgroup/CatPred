<script setup lang="ts">
import { computed } from 'vue'
import type { Parameter, PredictionResultEntry } from '../api/types'
import { PARAMETER_LABELS } from '../api/types'
import { parsePrediction, formatNumber, formatUnit, confidenceRange } from '../composables/usePrediction'

const props = defineProps<{
  results: PredictionResultEntry[]
}>()

const hasResults = computed(() =>
  props.results.some((r) => r.response.preview_rows.length > 0),
)

function truncate(value: unknown, max: number): string {
  const text = String(value ?? '')
  return text.length <= max ? text : text.slice(0, max - 1) + '\u2026'
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number') return formatNumber(value)
  return String(value)
}

function paramColor(param: Parameter): string {
  const colors: Record<Parameter, string> = {
    kcat: 'var(--color-kcat)',
    km: 'var(--color-km)',
    ki: 'var(--color-ki)',
  }
  return colors[param]
}

function exportCsv(entry: PredictionResultEntry) {
  const rows = entry.response.preview_rows
  if (!rows.length) return
  const keys = Object.keys(rows[0])
  const header = keys.join(',')
  const body = rows.map((row) =>
    keys
      .map((k) => {
        const v = row[k]
        if (v === null || v === undefined) return ''
        return String(v)
      })
      .join(','),
  )
  const csv = [header, ...body].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `catpred-${entry.parameter}-results.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function tableKeys(entry: PredictionResultEntry): string[] {
  const rows = entry.response.preview_rows
  if (!rows.length) return []
  return Object.keys(rows[0])
}
</script>

<template>
  <section class="result-panel" aria-live="polite">
    <div class="panel-head">
      <h3 class="panel-title">Results</h3>
    </div>

    <!-- Empty state -->
    <div v-if="!hasResults" class="empty-state">
      <svg class="empty-icon" viewBox="0 0 48 48" width="48" height="48" fill="none" aria-hidden="true">
        <circle cx="24" cy="24" r="18" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" opacity="0.4"/>
        <path d="M17 30c2-4.5 4-8 7-10 3 2 5 5.5 7 10" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" opacity="0.35"/>
        <circle cx="24" cy="18" r="2.5" stroke="currentColor" stroke-width="1" opacity="0.3"/>
      </svg>
      <p>Run a prediction to see results here.</p>
    </div>

    <!-- Result groups by parameter -->
    <div v-else class="result-groups">
      <div
        v-for="entry in results"
        :key="entry.parameter"
        class="result-group"
      >
        <div class="group-head">
          <span
            class="param-chip"
            :style="{ color: paramColor(entry.parameter), borderColor: paramColor(entry.parameter) }"
          >{{ PARAMETER_LABELS[entry.parameter] }}</span>
          <button
            type="button"
            class="export-btn"
            @click="exportCsv(entry)"
          >Export CSV</button>
        </div>

        <div class="result-cards">
          <article
            v-for="(row, idx) in entry.response.preview_rows"
            :key="idx"
            class="result-card"
          >
            <div class="card-head">
              <h4>Result {{ idx + 1 }}</h4>
            </div>

            <div class="card-main">
              <strong :style="{ color: paramColor(entry.parameter) }">
                {{ formatNumber(parsePrediction(row).linear) }}
              </strong>
              <span class="card-unit">{{ formatUnit(parsePrediction(row).unit) || 'predicted unit' }}</span>
            </div>

            <p
              v-if="confidenceRange(parsePrediction(row).log10, parsePrediction(row).sdTotal)"
              class="card-range"
            >
              &plusmn;1 SD range:
              {{ confidenceRange(parsePrediction(row).log10, parsePrediction(row).sdTotal)![0] }}
              &ndash;
              {{ confidenceRange(parsePrediction(row).log10, parsePrediction(row).sdTotal)![1] }}
              {{ formatUnit(parsePrediction(row).unit) }}
            </p>

            <dl class="metrics">
              <div class="metric">
                <dt>log&#x2081;&#x2080;</dt>
                <dd>{{ formatNumber(parsePrediction(row).log10) }}</dd>
              </div>
              <div class="metric">
                <dt>SD total (log&#x2081;&#x2080;)</dt>
                <dd>{{ formatNumber(parsePrediction(row).sdTotal) }}</dd>
              </div>
              <div class="metric">
                <dt>SD epistemic (log&#x2081;&#x2080;)</dt>
                <dd>{{ formatNumber(parsePrediction(row).sdEpistemic) }}</dd>
              </div>
            </dl>

            <div class="card-meta">
              <span>SMILES: {{ truncate(row.SMILES, 24) }}</span>
              <span>ID: {{ row.pdbpath || '\u2014' }}</span>
            </div>
          </article>
        </div>

        <!-- Detail table per parameter -->
        <details class="details-table">
          <summary>View detailed {{ PARAMETER_LABELS[entry.parameter] }} output table</summary>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th v-for="key in tableKeys(entry)" :key="key">{{ key }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, idx) in entry.response.preview_rows" :key="idx">
                  <td v-for="key in tableKeys(entry)" :key="key">{{ formatCell(row[key]) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </div>
  </section>
</template>

<style scoped>
.result-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem;
  box-shadow: var(--shadow-sm);
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.panel-title {
  font-family: var(--font-serif);
  font-size: 1.25rem;
  font-weight: 400;
}

/* Empty state */
.empty-state {
  border: 1px dashed var(--border);
  border-radius: var(--radius-md);
  padding: 2rem 1rem;
  text-align: center;
  color: var(--text-tertiary);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
}

.empty-icon {
  opacity: 0.5;
}

.empty-state p {
  font-size: 0.85rem;
}

/* Result groups */
.result-groups {
  display: grid;
  gap: 1rem;
}

.result-group {
  /* No extra styling needed, gap handles separation */
}

.group-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.param-chip {
  height: 22px;
  padding: 0 0.5rem;
  border: 1px solid;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
}

.export-btn {
  height: 28px;
  padding: 0 0.6rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  color: var(--text-secondary);
  transition: all 0.15s;
}

.export-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(5, 150, 105, 0.04);
}

/* Cards */
.result-cards {
  display: grid;
  gap: 0.5rem;
}

.result-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg);
  padding: 0.75rem;
  animation: card-in 0.3s ease backwards;
}

.result-card:nth-child(2) { animation-delay: 0.06s; }
.result-card:nth-child(3) { animation-delay: 0.12s; }
.result-card:nth-child(4) { animation-delay: 0.18s; }
.result-card:nth-child(5) { animation-delay: 0.24s; }

@keyframes card-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-head h4 {
  font-family: var(--font-serif);
  font-size: 1rem;
  font-weight: 400;
}

.card-main {
  margin-top: 0.375rem;
  display: flex;
  align-items: baseline;
  gap: 0.375rem;
}

.card-main strong {
  font-family: var(--font-serif);
  font-size: 1.75rem;
  font-weight: 400;
  line-height: 1;
}

.card-unit {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.card-range {
  margin-top: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-tertiary);
}

.metrics {
  margin-top: 0.5rem;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.375rem;
}

.metric {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  padding: 0.375rem;
}

.metric dt {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  font-weight: 500;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.metric dd {
  margin-top: 0.125rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.card-meta {
  margin-top: 0.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.375rem;
}

.card-meta span {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-secondary);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.125rem 0.4rem;
  background: var(--bg-surface);
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Detail table */
.details-table {
  margin-top: 0.5rem;
  border-top: 1px solid var(--border);
  padding-top: 0.5rem;
}

.details-table summary {
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}

.table-wrap {
  margin-top: 0.5rem;
  overflow: auto;
}

table {
  width: 100%;
  min-width: 600px;
}

th, td {
  text-align: left;
  border-bottom: 1px solid var(--border);
  padding: 0.375rem 0.5rem;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

th {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}

@media (max-width: 640px) {
  .metrics {
    grid-template-columns: 1fr;
  }
}
</style>
