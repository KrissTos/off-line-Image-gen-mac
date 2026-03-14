// ── Domain types ──────────────────────────────────────────────────────────────

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
  lora_file:          string | null
  lora_strength:      number
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
  ip_adapter_image_ids: string[]
  ip_adapter_scales:    number[]
  ip_adapter_enabled:   boolean
}

/** One reference-image slot — image + optional per-slot mask, labeled #1/#2/… */
export interface RefImageSlot {
  slotId:   number        // 1-based display label
  imageId:  string        // temp upload id
  imageUrl: string        // preview URL (/api/temp/…)
  maskId:   string | null // temp upload id for mask
  maskUrl:  string | null // preview URL for mask
  strength: number        // per-slot inpaint strength (0–1); also drives img_strength for slot #1
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

/** One IP-Adapter reference image slot */
export interface IpAdapterSlot {
  slotId:   number       // 1-based
  imageId:  string       // temp upload id
  imageUrl: string       // preview URL
  scale:    number       // per-image strength (0–1)
}

/** Status of IP-Adapter weights on disk / in pipeline */
export interface IpAdapterStatus {
  downloaded:  boolean
  loaded:      boolean
}
