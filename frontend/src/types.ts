// ── Domain types ──────────────────────────────────────────────────────────────

export interface LoraSlot {
  path:     string   // absolute path on server (inside lora_uploads/)
  strength: number   // 0–2
}

export interface AppStatus {
  model:   string | null
  device:  string | null
  loaded:  boolean
  busy:    boolean
  vram_gb: number
}

export interface GenerateParams {
  prompt:             string
  height:             number
  width:              number
  steps:              number
  seed:               number
  guidance:           number
  device:             string
  model_choice:       string
  model_source:       string
  input_image_ids:    string[]
  mask_image_id:      string | null
  lora_files:         LoraSlot[]
  img_strength:       number
  repeat_count:       number
  auto_save:          boolean
  output_dir:         string
  upscale_enabled:    boolean
  upscale_model_path: string
  num_frames:         number
  fps:                number
  mask_mode:          string
  outpaint_align:     string
}

/** One reference-image slot — image + optional per-slot mask, labeled #1/#2/… */
export interface RefImageSlot {
  slotId:   number        // 1-based display label
  imageId:  string        // temp upload id
  imageUrl: string        // preview URL (/api/temp/…)
  maskId:   string | null // temp upload id for mask
  maskUrl:  string | null // preview URL for mask
  strength: number        // per-slot inpaint strength (0–1); also drives img_strength for slot #1
  w?:       number        // natural image width (populated on thumbnail load)
  h?:       number        // natural image height
}

export type SSEEvent =
  | { type: 'progress'; message: string; step?: number; total?: number }
  | { type: 'image';    url: string; info?: string; path?: string }
  | { type: 'video';    url: string; path?: string }
  | { type: 'done' }
  | { type: 'error';    message: string }

export interface OutputItem {
  name:          string
  url:           string
  mtime:         number
  kind:          'image' | 'video'
  /** Filled from sidecar JSON written at generation time */
  prompt?:       string
  model_choice?: string
}

export interface Workflow {
  name: string
}
