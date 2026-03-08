import type { AppStatus, GenerateParams, OutputItem, SSEEvent } from './types'

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
export const pingServer   = () => fetch('/api/ping', { method: 'POST' }).catch(() => {})
export const fetchDevices = () => get<{ devices: string[] }>('/api/devices')
export const fetchModels  = () =>
  get<{ choices: string[]; available: string[]; current: string | null }>('/api/models')

export const loadModel = (model_choice: string, device: string) =>
  post<{ status: string }>('/api/models/load', { model_choice, device })

export const deleteModel = (name: string) =>
  del<{ status: string }>(`/api/models/${encodeURIComponent(name)}`)

export interface ModelUpdateResult {
  choice:       string
  repo_id:      string
  local_hash:   string | null
  online_hash:  string | null
  status:       'up_to_date' | 'update_available' | 'not_downloaded' | 'error'
}

export const checkModelUpdates = () =>
  get<{ results: ModelUpdateResult[] }>('/api/models/check-updates')

export const updateModel = (model_choice: string) =>
  post<{ status: string }>('/api/models/update', { model_choice, device: '' })

export const openFolderDialog = () =>
  get<{ path: string | null; cancelled: boolean }>('/api/open-folder-dialog')

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
  if (!r.ok) throw new Error(`LoRA upload failed: ${r.status}`)
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
export const loadWorkflow     = (name: string) => get<Record<string, unknown>>(`/api/workflows/${encodeURIComponent(name)}`)
export const saveWorkflow     = (data: Record<string, unknown>) =>
  post<{ status: string }>('/api/workflows/save', data)

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

// ── Settings ──────────────────────────────────────────────────────────────────

export const fetchSettings  = () => get<Record<string, unknown>>('/api/settings')
export const updateSettings = (settings: Record<string, unknown>) =>
  post<{ status: string }>('/api/settings', { settings })

// ── Storage ───────────────────────────────────────────────────────────────────

export const fetchStorage = () =>
  get<{ models: { name: string; size: string; choice: string }[]; summary: string }>('/api/storage')

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
  const r = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  })

  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail ?? `Generate failed: ${r.status}`)
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
