import { ref, computed } from 'vue'
import { fetchPredict } from '../api/client'
import type {
  Parameter,
  PredictPayload,
  PredictionResultEntry,
  ParsedPrediction,
} from '../api/types'

export type PredictionStatus = 'idle' | 'running' | 'success' | 'error'

export interface PredictionJob {
  parameter: Parameter
  payload: PredictPayload
}

const STORAGE_KEY = 'catpred:lastResults'

function loadStoredResults(): PredictionResultEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed as PredictionResultEntry[]
  } catch {
    return []
  }
}

function storeResults(data: PredictionResultEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
  } catch {
    // storage full or unavailable
  }
}

export function parsePrediction(row: Record<string, unknown>): ParsedPrediction {
  const keys = Object.keys(row)
  const linearKey = keys.find((k) => k.startsWith('Prediction_('))
  const unitMatch = linearKey?.match(/^Prediction_\((.*)\)$/)
  const unit = unitMatch ? unitMatch[1] : ''

  return {
    linear: linearKey ? (row[linearKey] as number | null) : null,
    linearKey: linearKey || 'Prediction',
    unit,
    log10: (row.Prediction_log10 as number | null) ?? null,
    sdTotal: (row.SD_total as number | null) ?? null,
    sdAleatoric: (row.SD_aleatoric as number | null) ?? null,
    sdEpistemic: (row.SD_epistemic as number | null) ?? null,
  }
}

const SUPERSCRIPT: Record<string, string> = {
  '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3',
  '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077',
  '8': '\u2078', '9': '\u2079', '-': '\u207B', '+': '\u207A',
}

export function formatUnit(unit: string): string {
  if (!unit) return ''
  // Replace ^(...) and ^N patterns with Unicode superscripts
  return unit.replace(/\^(?:\(([^)]+)\)|([0-9+-]+))/g, (_, group, bare) => {
    const content = group ?? bare
    return [...content].map((ch) => SUPERSCRIPT[ch] ?? ch).join('')
  })
}

export function formatNumber(value: unknown): string {
  const n = Number(value)
  if (!Number.isFinite(n)) return '\u2014'
  return n.toFixed(1)
}

export function confidenceRange(
  log10: number | null,
  sd: number | null,
): [string, string] | null {
  if (log10 === null || sd === null || !Number.isFinite(log10) || !Number.isFinite(sd)) {
    return null
  }
  const lo = Math.pow(10, log10 - sd)
  const hi = Math.pow(10, log10 + sd)
  return [formatNumber(lo), formatNumber(hi)]
}

export function formatElapsed(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds))
  const mins = Math.floor(safe / 60)
  const secs = safe % 60
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

export function usePrediction() {
  const status = ref<PredictionStatus>('idle')
  const error = ref('')
  const results = ref<PredictionResultEntry[]>(loadStoredResults())
  const elapsedSeconds = ref(0)
  const lastJobs = ref<PredictionJob[] | null>(null)

  let startTime = 0
  let timerInterval: ReturnType<typeof setInterval> | null = null
  let abortController: AbortController | null = null

  const isRunning = computed(() => status.value === 'running')
  const hasResults = computed(() => results.value.length > 0)

  function startTimer() {
    stopTimer()
    startTime = Date.now()
    elapsedSeconds.value = 0
    timerInterval = setInterval(() => {
      elapsedSeconds.value = Math.floor((Date.now() - startTime) / 1000)
    }, 1000)
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval)
      timerInterval = null
    }
  }

  async function runAll(jobs: PredictionJob[]) {
    abortController?.abort()
    abortController = new AbortController()
    lastJobs.value = jobs
    status.value = 'running'
    error.value = ''
    results.value = []
    startTimer()

    try {
      const settled = await Promise.all(
        jobs.map(async (job) => {
          const data = await fetchPredict(job.payload, abortController!.signal)
          return { parameter: job.parameter, response: data } as PredictionResultEntry
        }),
      )
      results.value = settled
      storeResults(settled)
      status.value = 'success'
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Prediction failed.'
      status.value = 'error'
    } finally {
      stopTimer()
      abortController = null
    }
  }

  function retry() {
    if (lastJobs.value) {
      runAll(lastJobs.value)
    }
  }

  function cancel() {
    abortController?.abort()
  }

  return {
    status,
    error,
    results,
    elapsedSeconds,
    lastJobs,
    isRunning,
    hasResults,
    runAll,
    retry,
    cancel,
  }
}
