// Gallery preview modal: pure helpers, tested with node --test.
import type { OutputItem } from './types'
import type { WorkflowData } from './workflow'

export interface Row { label: string; value: string }

export function loraName(path: string): string {
  const base = path.split('/').pop() ?? path
  return base.replace(/\.safetensors$/i, '')
}

/** Checks use `!= null` so seed 0 / steps 0 survive. */
export function paramRows(item: OutputItem, hasMask: boolean): Row[] {
  const rows: Row[] = []
  if (item.model_choice) rows.push({ label: 'Model', value: item.model_choice })
  if (item.width && item.height) rows.push({ label: 'Size', value: `${item.width} × ${item.height}` })
  if (item.steps != null) rows.push({ label: 'Steps', value: String(item.steps) })
  if (item.seed != null) rows.push({ label: 'Seed', value: String(item.seed) })
  if (item.kind === 'video' && item.num_frames) {
    rows.push({ label: 'Frames', value: item.fps ? `${item.num_frames} @ ${item.fps} fps` : String(item.num_frames) })
  }
  if (hasMask && item.mask_mode) rows.push({ label: 'Mask mode', value: item.mask_mode })
  return rows
}

export function loraRows(item: OutputItem): Row[] {
  return (item.lora_files ?? []).map(l => ({ label: loraName(l.path), value: Number(l.strength ?? 1).toFixed(2) }))
}

export function neighbor(outputs: OutputItem[], url: string, dir: -1 | 1): OutputItem | null {
  const i = outputs.findIndex(o => o.url === url)
  if (i < 0) return null
  return outputs[i + dir] ?? null
}

/** The output this one was upscaled from, when it still exists in the gallery. */
export function upscaleSource(outputs: OutputItem[], wf: WorkflowData | null, item: OutputItem): OutputItem | null {
  const entries = (wf?.outputs as { file?: string; upscaled_from?: string }[] | undefined) ?? []
  const from = entries.find(o => o.file === item.file)?.upscaled_from
  if (!from) return null
  return outputs.find(o => o.run === item.run && o.file === from) ?? null
}

export interface KeyInfo {
  key: string; metaKey: boolean; ctrlKey: boolean; altKey: boolean
  targetTag: string; targetEditable: boolean
}

/** Modal keyboard map. Arrows are left alone when a modifier is held (browser back/forward) or the
 *  focus is a control that owns them (video seek, text fields). */
export function keyAction(e: KeyInfo): 'close' | 'prev' | 'next' | null {
  if (e.key === 'Escape') return 'close'
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return null
  if (e.metaKey || e.ctrlKey || e.altKey) return null
  if (e.targetEditable || ['VIDEO', 'INPUT', 'TEXTAREA', 'SELECT'].includes(e.targetTag)) return null
  return e.key === 'ArrowLeft' ? 'prev' : 'next'
}
