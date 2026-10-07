// Model Sources list: categories, LoRA family folders, function filter. Pure: tested with node --test.
import type { ModelSource } from './api'

const FAMILY_FOLDERS = [
  { key: 'klein-9B',  label: 'FLUX.2-klein 9B' },
  { key: 'klein-4B',  label: 'FLUX.2-klein 4B' },
  { key: 'klein',     label: 'FLUX.2-klein (any size)' },
  { key: 'Z-Image',   label: 'Z-Image' },
  { key: 'LTX-Video', label: 'LTX-Video 0.9' },
] as const
const OTHER = { key: 'other', label: 'Other' }

const CATEGORIES: { type: ModelSource['type']; label: string }[] = [
  { type: 'base',     label: 'Models' },
  { type: 'lora',     label: 'LoRAs' },
  { type: 'upscaler', label: 'Upscalers' },
]

export interface Folder   { key: string; label: string; items: ModelSource[] }
export interface Category { type: ModelSource['type']; label: string; items: ModelSource[]; folders?: Folder[] }

export function familyLabel(family: string | undefined): string {
  return FAMILY_FOLDERS.find(f => f.key === family)?.label ?? OTHER.label
}

const byName = (a: ModelSource, b: ModelSource) =>
  (Number(!!b.installed) - Number(!!a.installed)) || a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })

export function groupSources(sources: ModelSource[]): Category[] {
  const out: Category[] = []
  for (const c of CATEGORIES) {
    const items = sources.filter(s => s.type === c.type)
    if (!items.length) continue
    if (c.type !== 'lora') { out.push({ ...c, items }); continue }
    const known = new Set<string>(FAMILY_FOLDERS.map(f => f.key))
    const folders: Folder[] = [...FAMILY_FOLDERS, OTHER]
      .map(f => ({
        key: f.key, label: f.label,
        items: items.filter(s => (f.key === 'other' ? !known.has(s.family ?? '') : s.family === f.key)).sort(byName),
      }))
      .filter(f => f.items.length > 0)
    out.push({ ...c, items, folders })
  }
  return out
}

export function functionCounts(items: ModelSource[]): { fn: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const s of items) if (s.function) counts.set(s.function, (counts.get(s.function) ?? 0) + 1)
  return [...counts].map(([fn, count]) => ({ fn, count }))
    .sort((a, b) => b.count - a.count || a.fn.localeCompare(b.fn))
}

export function filterByFunction(items: ModelSource[], fn: string | null): ModelSource[] {
  return fn ? items.filter(s => s.function === fn) : items
}

/** Open/closed state of the folders as stored in localStorage; anything unreadable → {}. */
export function parseOpenState(raw: string | null): Record<string, boolean> {
  try {
    const v = raw ? JSON.parse(raw) : null
    if (!v || typeof v !== 'object' || Array.isArray(v)) return {}
    return Object.fromEntries(Object.entries(v).filter(([, b]) => typeof b === 'boolean')) as Record<string, boolean>
  } catch { return {} }
}

export type CivitaiRowState = 'download' | 'installed' | 'update'
export const civitaiRowState = (s: ModelSource): CivitaiRowState =>
  s.update ? 'update' : s.installed ? 'installed' : 'download'

/** File size for a row: '' unknown, MB under 1 GB, else GB with one decimal. */
export function formatSize(kb: number | undefined): string {
  if (!kb || kb <= 0) return ''
  const mb = kb / 1024
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`
}

export const progressPct = (bytes?: number, total?: number): number =>
  !total || total <= 0 ? 0 : Math.min(100, Math.round(((bytes ?? 0) / total) * 100))

/** CivitAI part of the Update message ('' when there is nothing to say). */
export function civitaiNote(c: { added: number; failed: string[]; updates: number } | undefined): string {
  if (!c) return ''
  const parts: string[] = []
  if (c.added > 0) parts.push(`${c.added} CivitAI new`)
  if (c.updates > 0) parts.push(`${c.updates} update${c.updates > 1 ? 's' : ''}`)
  if (c.failed.length) parts.push(`could not reach CivitAI for ${c.failed.join(', ')}`)
  return parts.join(' · ')
}
