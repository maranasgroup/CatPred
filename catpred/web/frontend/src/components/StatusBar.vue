<script setup lang="ts">
import type { ServiceStatus } from '../composables/useReadiness'
import type { PredictionStatus } from '../composables/usePrediction'
import { formatElapsed } from '../composables/usePrediction'

defineProps<{
  serviceStatus: ServiceStatus
  serviceHint: string
  predictionStatus: PredictionStatus
  predictionError: string
  elapsed: number
  canRetry: boolean
}>()

const emit = defineEmits<{
  retry: []
}>()
</script>

<template>
  <div class="status-bar" role="status" aria-live="polite">
    <!-- Service status -->
    <div class="status-row">
      <span class="status-label">Service</span>
      <span :class="['badge', serviceStatus]">
        {{ serviceStatus === 'checking' ? 'Checking...' :
           serviceStatus === 'online' ? 'Online' :
           serviceStatus === 'limited' ? 'Limited' : 'Offline' }}
      </span>
      <span v-if="serviceHint" class="status-hint">{{ serviceHint }}</span>
    </div>

    <!-- Prediction status -->
    <div v-if="predictionStatus !== 'idle'" class="status-row prediction-row">
      <span v-if="predictionStatus === 'running'" class="status-running">
        <span class="spinner" aria-hidden="true"></span>
        Running {{ formatElapsed(elapsed) }}
      </span>
      <span v-else-if="predictionStatus === 'success'" class="status-success">
        Prediction complete &middot; {{ formatElapsed(elapsed) }}
      </span>
      <span v-else-if="predictionStatus === 'error'" class="status-error">
        {{ predictionError || 'Prediction failed' }}
        <button
          v-if="canRetry"
          type="button"
          class="retry-btn"
          @click="emit('retry')"
        >Retry</button>
      </span>
    </div>
  </div>
</template>

<style scoped>
.status-bar {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  padding: 0.5rem 0.75rem;
  display: grid;
  gap: 0.375rem;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  min-height: 28px;
}

.status-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}

.badge {
  display: inline-flex;
  align-items: center;
  height: 22px;
  padding: 0 0.5rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
}

.badge.online {
  color: var(--ok);
  border-color: rgba(5, 150, 105, 0.3);
  background: rgba(5, 150, 105, 0.06);
}

.badge.offline,
.badge.limited {
  color: var(--danger);
  border-color: rgba(220, 38, 38, 0.3);
  background: rgba(220, 38, 38, 0.06);
}

.badge.checking {
  color: var(--text-tertiary);
}

.status-hint {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-tertiary);
}

.prediction-row {
  border-top: 1px solid var(--border);
  padding-top: 0.375rem;
}

.status-running {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--accent);
  font-weight: 500;
}

.spinner {
  width: 12px;
  height: 12px;
  border: 2px solid rgba(5, 150, 105, 0.2);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.status-success {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--ok);
  font-weight: 500;
}

.status-error {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--danger);
}

.retry-btn {
  height: 24px;
  padding: 0 0.5rem;
  border: 1px solid rgba(220, 38, 38, 0.3);
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  color: var(--danger);
  transition: background 0.15s;
}

.retry-btn:hover {
  background: rgba(220, 38, 38, 0.06);
}
</style>
