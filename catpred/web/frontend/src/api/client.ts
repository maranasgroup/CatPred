import type { ReadyResponse, PredictPayload, PredictResponse } from './types'

const PREDICT_TIMEOUT_MS = 120_000

export async function fetchReady(signal?: AbortSignal): Promise<ReadyResponse> {
  const res = await fetch('/ready', {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal,
  })
  if (!res.ok) {
    throw new Error(`Service check failed (${res.status})`)
  }
  return res.json()
}

export async function fetchPredict(
  payload: PredictPayload,
  signal?: AbortSignal,
): Promise<PredictResponse> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), PREDICT_TIMEOUT_MS)

  // Combine external signal with timeout
  if (signal) {
    signal.addEventListener('abort', () => controller.abort())
  }

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })

    if (!res.ok) {
      let detail = 'Prediction could not be completed.'
      try {
        const data = await res.json()
        if (data?.detail) detail = String(data.detail)
      } catch {
        // ignore parse error
      }
      throw new Error(detail)
    }

    return res.json()
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(
        `Prediction timed out after ${Math.floor(PREDICT_TIMEOUT_MS / 1000)}s. ` +
          'This may be a cold-start delay — try again.',
      )
    }
    throw err
  } finally {
    clearTimeout(timeout)
  }
}
