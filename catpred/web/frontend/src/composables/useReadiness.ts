import { ref, computed } from 'vue'
import { fetchReady } from '../api/client'
import type { Parameter } from '../api/types'
import { SUPPORTED_PARAMETERS } from '../api/types'

export type ServiceStatus = 'checking' | 'online' | 'limited' | 'offline'

export function useReadiness() {
  const status = ref<ServiceStatus>('checking')
  const statusHint = ref('')
  const defaultBackend = ref('local')
  const modalReady = ref(false)
  const localReady = ref(false)
  const fallbackToLocalEnabled = ref(false)
  const availableCheckpoints = ref<Set<Parameter>>(new Set(SUPPORTED_PARAMETERS))
  const localCheckpoints = ref<Set<Parameter>>(new Set(SUPPORTED_PARAMETERS))

  function isParameterAvailable(param: Parameter): boolean {
    return availableCheckpoints.value.has(param)
  }

  const firstAvailable = computed<Parameter | null>(() => {
    for (const p of SUPPORTED_PARAMETERS) {
      if (availableCheckpoints.value.has(p)) return p
    }
    return null
  })

  function chooseBackend(): string {
    if (defaultBackend.value === 'modal' && modalReady.value) return 'modal'
    if (defaultBackend.value === 'local' && localCheckpoints.value.size > 0) return 'local'
    if (modalReady.value) return 'modal'
    if (localReady.value) return 'local'
    return defaultBackend.value || 'local'
  }

  function shouldFallback(requestBackend: string, param: Parameter): boolean {
    if (requestBackend !== 'modal') return false
    if (!fallbackToLocalEnabled.value) return false
    if (!localReady.value) return false
    return localCheckpoints.value.has(param)
  }

  function parseCheckpoints(available: Record<string, string> | undefined): Set<Parameter> {
    if (!available) return new Set(SUPPORTED_PARAMETERS)
    return new Set(
      Object.keys(available)
        .map((k) => k.toLowerCase() as Parameter)
        .filter((k) => SUPPORTED_PARAMETERS.includes(k)),
    )
  }

  async function check() {
    status.value = 'checking'
    statusHint.value = ''

    try {
      const data = await fetchReady()

      defaultBackend.value = data.default_backend || 'local'
      modalReady.value = Boolean(data.backends?.modal?.ready)
      localReady.value = Boolean(data.backends?.local?.ready)
      fallbackToLocalEnabled.value = Boolean(data.fallback_to_local_enabled)

      localCheckpoints.value = parseCheckpoints(data.api?.available_checkpoints)

      if (localCheckpoints.value.size > 0) {
        availableCheckpoints.value = new Set(localCheckpoints.value)
      } else if (modalReady.value) {
        availableCheckpoints.value = new Set(SUPPORTED_PARAMETERS)
      } else {
        availableCheckpoints.value = new Set()
      }

      if (data.ready) {
        status.value = 'online'
        const params = Array.from(localCheckpoints.value).join(', ')
        statusHint.value = params
          ? `${data.default_backend} · ${params}`
          : `${data.default_backend} · no local checkpoints`
      } else {
        status.value = 'limited'
        statusHint.value = modalReady.value
          ? 'Backend available in limited mode'
          : 'No local checkpoints found'
      }
    } catch {
      status.value = 'offline'
      statusHint.value = 'Could not contact service'
    }
  }

  return {
    status,
    statusHint,
    availableCheckpoints,
    firstAvailable,
    isParameterAvailable,
    chooseBackend,
    shouldFallback,
    check,
  }
}
