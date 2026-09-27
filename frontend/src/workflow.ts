// Workflow (run folder or saved workflow) → generate params. Pure: tested with node --test.
import type { GenerateParams, LoraSlot } from './types'

/** One ref slot as returned by GET /api/runs/{run} and GET /api/workflows/{name}. */
export interface WorkflowSlot { imageUrl: string; maskUrl: string | null; strength: number }

export interface WorkflowData {
  [key: string]: unknown
  ref_slots?: WorkflowSlot[]
  warnings?:  string[]
}

const NUMBER_KEYS = ['height', 'width', 'steps', 'seed', 'guidance', 'img_strength',
                     'repeat_count', 'num_frames', 'fps'] as const
const STRING_KEYS = ['model_choice', 'model_source', 'device', 'upscale_model_path',
                     'mask_mode', 'outpaint_align'] as const
const BOOL_KEYS   = ['upscale_enabled', 'fast_preview'] as const

/** Every param the workflow carries (`!= null` checks, so seed 0 survives); empty strings are
 *  skipped except for the prompt. `opts.seed` (the clicked output's seed) overrides the stored one. */
export function workflowToParams(wf: WorkflowData, opts: { seed?: number | null } = {}): Partial<GenerateParams> {
  const p: Partial<GenerateParams> = {}
  if (typeof wf.prompt === 'string') p.prompt = wf.prompt
  for (const k of NUMBER_KEYS) {
    const v = wf[k]
    if (typeof v === 'number' && Number.isFinite(v)) p[k] = v
  }
  for (const k of STRING_KEYS) {
    const v = wf[k]
    if (typeof v === 'string' && v !== '') p[k] = v
  }
  for (const k of BOOL_KEYS) {
    const v = wf[k]
    if (typeof v === 'boolean') p[k] = v
  }
  if (Array.isArray(wf.lora_files)) {
    p.lora_files = (wf.lora_files as unknown[])
      .filter((l): l is LoraSlot => !!l && typeof (l as LoraSlot).path === 'string')
      .map(l => ({ ...l, strength: Number(l.strength ?? 1) }))
  }
  if (opts.seed != null) p.seed = opts.seed
  return p
}
