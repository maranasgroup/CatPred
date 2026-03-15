export interface ReadyResponse {
  ready: boolean
  default_backend: string
  fallback_to_local_enabled: boolean
  backends: {
    local?: { ready: boolean }
    modal?: { ready: boolean }
  }
  api?: {
    allow_input_file: boolean
    allow_unsafe_request_overrides: boolean
    max_input_rows: number
    max_input_file_bytes: number
    available_checkpoints: Record<string, string>
    missing_checkpoints: string[]
  }
}

export interface InputRow {
  SMILES: string
  sequence: string
  pdbpath: string
}

export interface PredictPayload {
  parameter: string
  checkpoint_dir: string
  input_rows: InputRow[]
  use_gpu: boolean
  results_dir: string
  backend: string
  fallback_to_local: boolean
}

export interface PredictResponse {
  backend: string
  output_file: string
  row_count: number
  preview_rows: Record<string, unknown>[]
  metadata: Record<string, unknown>
}

export type Parameter = 'kcat' | 'km' | 'ki'

export const SUPPORTED_PARAMETERS: Parameter[] = ['kcat', 'km', 'ki']

export const PARAMETER_LABELS: Record<Parameter, string> = {
  kcat: 'kcat',
  km: 'Km',
  ki: 'Ki',
}

export interface ParsedPrediction {
  linear: number | null
  linearKey: string
  unit: string
  log10: number | null
  sdTotal: number | null
  sdAleatoric: number | null
  sdEpistemic: number | null
}

export type PredictionMode = 'substrate' | 'inhibition'

export interface SubstrateEntry {
  id: number
  smiles: string
  isPrimary: boolean
}

export interface PredictionResultEntry {
  parameter: Parameter
  response: PredictResponse
}
