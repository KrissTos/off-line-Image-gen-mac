import type { AppStatus, GenerateParams, OutputItem, SSEEvent } from './types'
import type { WorkflowData } from './workflow'

const BASE = ''   // same-origin; Vite proxies /api in dev

// ── Generic helpers ───────────────────────────────────────────────────────────

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`)
  return r.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail ?? `POST ${path} → ${r.status}`)
  }
  return r.json()
}

async function del<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path, { method: 'DELETE' })
  if (!r.ok) throw new Error(`DELETE ${path} → ${r.status}`)
  return r.json()
}

// ── Status / devices / models ─────────────────────────────────────────────────

export const fetchStatus  = () => get<AppStatus>('/api/status')
export const pingServer      = () => fetch('/api/ping', { method: 'POST' }).catch(() => {})
export const stopGeneration  = () => fetch('/api/stop', { method: 'POST' }).catch(() => {})
export const fetchDevices = () => get<{ devices: string[] }>('/api/devices')
export const fetchModels  = () =>
  get<{ choices: string[]; available: string[]; current: string | null }>('/api/models')

export const loadModel = (model_choice: string, device: string) =>
  post<{ status: string }>('/api/models/load', { model_choice, device })

export const deleteModel = (name: string) =>
  del<{ status: string }>(`/api/models/${encodeURIComponent(name)}`)

export interface ModelSource {
  id:           string
  name:         string
  url:          string
  type:         'base' | 'lora' | 'upscaler'
  description:  string
  model_choice?: string   // exact MODEL_CHOICES string for local-cache detection (base type only)
  vram_gb?:     number    // approx GPU memory the model needs (base type only) — drives Recommended tag
  family?:      string    // LoRA only: klein-9B | klein-4B | klein | Z-Image | LTX-Video (set by the server)
  function?:    string    // LoRA only: what it does (style, camera, detail, ...), filled from the model card
  custom?:      boolean   // added by hand: never pruned or auto-described
  provider?:          'civitai'
  nsfw?:              boolean
  installed?:         boolean   // CivitAI: file downloaded into lora_uploads/
  update?:            boolean   // CivitAI: a newer version than the installed one is listed
  installed_version?: number | null
  civitai?:           { modelId: number; versionId: number; file: string; sizeKB: number; trained: string[] }
}

export interface ModelUpdateResult {
  choice:       string
  repo_id:      string
  local_hash:   string | null
  online_hash:  string | null
  status:       'up_to_date' | 'update_available' | 'not_downloaded' | 'error'
}

export const checkModelUpdates = () =>
  get<{ results: ModelUpdateResult[] }>('/api/models/check-updates')

export interface DownloadProgressEvent {
  type:       'file_progress' | 'done' | 'error'
  filename?:  string
  downloaded?: number
  total?:     number
  pct?:       number
  message?:   string
}

export async function streamUpdateModel(
  model_choice: string,
  onEvent: (e: DownloadProgressEvent) => void,
  signal?: AbortSignal,
): Promise<string> {
  const r = await fetch('/api/models/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_choice, device: '' }),
    signal,
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail ?? `POST /api/models/update → ${r.status}`)
  }
  const reader = r.body!.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  let finalMsg = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop()!
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const event: DownloadProgressEvent = JSON.parse(line.slice(6))
      onEvent(event)
      if (event.type === 'done')  finalMsg = event.message ?? ''
      if (event.type === 'error') throw new Error(event.message)
    }
  }
  return finalMsg
}

export const openFolderDialog  = () =>
  get<{ path: string | null; cancelled: boolean }>('/api/open-folder-dialog')
export const openWorkflowFolderDialog = () =>
  get<{ path: string | null; cancelled: boolean }>('/api/open-workflow-folder-dialog')
export const openFileDialog    = () =>
  get<{ path: string | null; cancelled: boolean }>('/api/open-file-dialog')
export const openOutputFolder  = () =>
  get<{ ok: boolean }>('/api/open-output-folder')

// ── Depth Map ─────────────────────────────────────────────────────────────────

export interface DepthMapResult {
  url:      string
  filename: string
}

export const generateDepthMap = (params: {
  file_path?:  string
  filename?:   string
  model_repo?: string
}) => post<DepthMapResult>('/api/depth-map', params)

// ── Watermark Remover ─────────────────────────────────────────────────────────

export interface EraseDetectResult {
  image_id:  string
  image_url: string
  mask_id:   string
  mask_url:  string
}

export const eraseDetect = (filePath: string) =>
  post<EraseDetectResult>('/api/erase/detect', { file_path: filePath })

export interface EraseResult {
  url:      string
  filename: string
}

export const eraseRemove = (filePath: string, maskId: string) =>
  post<EraseResult>('/api/erase', { file_path: filePath, mask_id: maskId })

export interface SingleUpscaleResult {
  saved_path: string
  filename:   string
  url:        string | null
  width:      number
  height:     number
}

export const upscaleSingleImage = (params: {
  source:       'gallery' | 'path'
  filename?:    string
  file_path?:   string
  model_path:   string
  scale_choice: string
}) => post<SingleUpscaleResult>('/api/upscale/single', params)

// ── Upload ────────────────────────────────────────────────────────────────────

export async function uploadImage(file: File): Promise<{ id: string; url: string }> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/api/upload', { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`Upload failed: ${r.status}`)
  return r.json()
}

/**
 * Fetch an already-served output image and re-upload it as a new temp file.
 * Used by the iterate-masks loop to chain passes: output of pass N → input of pass N+1.
 */
export async function uploadFromUrl(url: string): Promise<{ id: string; url: string }> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`Failed to fetch image for re-upload: ${r.status}`)
  const blob = await r.blob()
  const file = new File([blob], 'iteration_output.png', { type: blob.type || 'image/png' })
  return uploadImage(file)
}

export async function uploadLora(file: File): Promise<{ path: string; name: string }> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/api/lora/upload', { method: 'POST', body: fd })
  if (!r.ok) {
    let detail = `LoRA upload failed: ${r.status}`
    try {
      const body = await r.json()
      if (body?.detail) detail = body.detail
    } catch { /* ignore parse errors */ }
    throw new Error(detail)
  }
  return r.json()
}

export async function uploadUpscaleModel(file: File): Promise<{ path: string; name: string }> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/api/upscale/upload', { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`Upscale model upload failed: ${r.status}`)
  return r.json()
}

export async function streamBatchUpscale(
  params: { input_folder: string; output_folder: string; scale_choice: string; model_path: string },
  onEvent: (e: { type: string; message?: string }) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch('/api/upscale/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail ?? `Batch upscale failed: ${r.status}`)
  }
  const reader  = r.body!.getReader()
  const decoder = new TextDecoder()
  let   buf     = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const ev = JSON.parse(line.slice(6))
          onEvent(ev)
          if (ev.type === 'done' || ev.type === 'error') return
        } catch { /* ignore malformed */ }
      }
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

export const fetchOutputs = (limit = 20) =>
  get<{ files: OutputItem[] }>(`/api/outputs?limit=${limit}`)

export const deleteOutput = (filename: string) =>
  del<{ deleted: string }>(`/api/output/${filename}`)

// ── Workflows ─────────────────────────────────────────────────────────────────

export const fetchWorkflows   = () => get<{ workflows: string[] }>('/api/workflows')
export const loadRun          = (run: string) => get<WorkflowData>(`/api/runs/${encodeURIComponent(run)}`)
export const loadWorkflow     = (name: string) => get<WorkflowData>(`/api/workflows/${encodeURIComponent(name)}`)
export const saveWorkflow     = (data: Record<string, unknown>) =>
  post<{ status: string; name: string }>('/api/workflows/save', data)

export async function importComfyUI(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/api/workflows/import', { method: 'POST', body: fd })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail ?? `Import failed: ${r.status}`)
  }
  return r.json()
}

// ── LoRA ──────────────────────────────────────────────────────────────────────

export const loadLora  = (lora_path: string, strength: number, device: string) =>
  post<{ status: string }>('/api/lora/load', { lora_path, strength, device })
export const clearLora = () => del<{ status: string }>('/api/lora')
export interface LoraLibraryEntry {
  name: string; path: string; model_type: string; variant?: string | null
  trigger?: string | null; trigger_note?: string; trigger_source?: string; trigger_origin?: string
}
export const listLoras = () => get<{ files: LoraLibraryEntry[] }>('/api/lora/list')
export async function setLoraTrigger(name: string, trigger: string): Promise<{ trigger: string | null }> {
  const r = await fetch('/api/lora/trigger', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, trigger }),
  })
  if (!r.ok) throw new Error(`Saving trigger failed: ${r.status}`)
  return r.json()
}

// ── Settings ──────────────────────────────────────────────────────────────────

export const fetchSettings  = () => get<Record<string, unknown>>('/api/settings')
export const updateSettings = (settings: Record<string, unknown>) =>
  post<{ status: string }>('/api/settings', { settings })

// ── Storage ───────────────────────────────────────────────────────────────────

export const fetchStorage = () =>
  get<{ models: { name: string; size: string; choice: string }[]; summary: string }>('/api/storage')

export type ModelExtras = {
  upscale_models: { name: string; size: string }[]
}
export const fetchModelExtras  = () => get<ModelExtras>('/api/models/extras')
export const deleteUpscaleModel = (filename: string) =>
  del<{ status: string }>(`/api/upscale/${encodeURIComponent(filename)}`)

// ── Generation SSE ────────────────────────────────────────────────────────────

/**
 * POST /api/generate and consume the SSE stream.
 * Calls onEvent for each event; returns when the stream ends or errors.
 */
export async function streamGenerate(
  params: GenerateParams,
  onEvent: (e: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let r: Response
  try {
    r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal,
    })
  } catch (err) {
    const msg = (err as Error).message ?? 'network error'
    throw new Error(`Cannot reach the generation server — is it still running? (${msg})`)
  }

  if (!r.ok) {
    let detail = r.statusText
    try {
      const body = await r.json()
      detail = body.detail ?? body.message ?? detail
    } catch { /* non-JSON body */ }
    throw new Error(`Generation request failed (HTTP ${r.status}): ${detail}`)
  }

  const reader  = r.body!.getReader()
  const decoder = new TextDecoder()
  let   buf     = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event: SSEEvent = JSON.parse(line.slice(6))
          onEvent(event)
          if (event.type === 'done' || event.type === 'error') return
        } catch {
          // ignore malformed lines
        }
      }
    }
  }
}

/**
 * POST /api/batch/generate and consume the SSE stream.
 * Processes all images in a folder and streams progress events including batch_progress.
 * Calls onEvent for each event; returns when the stream ends or errors.
 */
export async function streamBatchGenerate(
  params: GenerateParams,
  folder: string,
  onEvent: (e: SSEEvent | { type: 'batch_progress'; current: number; total: number; filename: string }) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch('/api/batch/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...params, input_folder: folder }),
    signal,
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail ?? `Batch generate failed: ${r.status}`)
  }
  const reader  = r.body!.getReader()
  const decoder = new TextDecoder()
  let   buf     = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const ev = JSON.parse(line.slice(6))
          onEvent(ev)
          // Only stop on the batch-level done (has 'processed' field).
          // Per-image 'done' and 'error' events must not abort the batch stream.
          if (ev.type === 'done' && 'processed' in ev) return
        } catch { /* ignore malformed */ }
      }
    }
  }
}

// ── Model sources ─────────────────────────────────────────────────────────────

export async function fetchModelSources(): Promise<ModelSource[]> {
  const r = await fetch('/api/model-sources')
  if (!r.ok) throw new Error(`Failed to fetch model sources: ${r.status}`)
  const data = await r.json()
  return data.sources as ModelSource[]
}

export async function saveModelSources(sources: ModelSource[]): Promise<void> {
  const r = await fetch('/api/model-sources', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sources }),
  })
  if (!r.ok) {
    let detail = `Failed to save model sources: ${r.status}`
    try { const b = await r.json(); if (b?.detail) detail = b.detail } catch {}
    throw new Error(detail)
  }
}

export interface DiscoverResult {
  added: number; skipped: number; described: number; failed: number; sources: ModelSource[]
  civitai?: { added: number; failed: string[]; updates: number }
}

export async function discoverModelSources(): Promise<DiscoverResult> {
  const r = await fetch('/api/model-sources/discover')
  if (!r.ok) throw new Error(`Discovery failed: ${r.status}`)
  return r.json()
}

export interface CivitaiStatus { has_key: boolean; show_nsfw: boolean }
export interface CivitaiJob {
  state: 'idle' | 'queued' | 'downloading' | 'verifying' | 'done' | 'error'
  bytes?: number; total?: number; error?: string | null; file?: string
}

export const fetchCivitaiStatus = () => get<CivitaiStatus>('/api/civitai/status')
export const setCivitaiKey = (key: string) => post<{ has_key: boolean }>('/api/civitai/key', { key })
export const clearCivitaiKey = () => del<{ has_key: boolean }>('/api/civitai/key')
export const startCivitaiDownload = (versionId: number) =>
  post<CivitaiJob>('/api/civitai/download', { version_id: versionId })
export const fetchCivitaiJob = (versionId: number) => get<CivitaiJob>(`/api/civitai/download/${versionId}`)

export async function deleteCivitai(versionId: number): Promise<void> {
  const r = await fetch(`/api/civitai/${versionId}`, { method: 'DELETE' })
  if (!r.ok) {
    const b = await r.json().catch(() => ({}))
    throw new Error(b.detail ?? `Delete failed: ${r.status}`)
  }
}

// ── SAM click-to-mask ─────────────────────────────────────────────────────────

export async function segmentStatus(): Promise<{ loaded: boolean }> {
  return get('/api/segment/status')
}

export async function segmentPrepare(imageId: string): Promise<{ ready: boolean; ms: number }> {
  return post('/api/segment/prepare', { image_id: imageId })
}

export async function segmentMask(
  imageId: string,
  points: { x: number; y: number; label: 0 | 1 }[],
  box: { x0: number; y0: number; x1: number; y1: number } | null,
): Promise<Blob> {
  const r = await fetch('/api/segment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, points, box }),
  })
  if (!r.ok) {
    let detail = `${r.status}`
    try { detail = (await r.json()).detail ?? detail } catch { /* non-JSON error */ }
    throw new Error(`Segment failed: ${detail}`)
  }
  return r.blob()
}
