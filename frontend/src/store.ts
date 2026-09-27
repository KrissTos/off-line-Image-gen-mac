import { useReducer } from 'react'
import type { GenerateParams, OutputItem, AppStatus, RefImageSlot } from './types'
import { canvasForRef, sizeFamily } from './canvasSize'

// ── State ─────────────────────────────────────────────────────────────────────

export interface State {
  // Generation params
  params: GenerateParams
  // App / pipeline status
  status: AppStatus
  // Available options
  models:          string[]
  availableModels: string[]
  devices:         string[]
  workflows:       string[]
  // Recent outputs
  outputs: OutputItem[]
  // Generation state
  isGenerating:  boolean
  progressMsg:   string
  progressStep:  number | undefined
  progressTotal: number | undefined
  // Latest result
  resultUrl:  string | null
  resultInfo: string | undefined
  // Reference image slots (#1, #2, …)
  refSlots: RefImageSlot[]
  // UI state
  settingsOpen: boolean
  error: string | null
}

const DEFAULT_PARAMS: GenerateParams = {
  prompt:             '',
  height:             512,
  width:              512,
  steps:              20,
  seed:               -1,
  guidance:           3.5,
  device:             'mps',
  model_choice:       '',
  model_source:       'Local',
  input_image_ids:    [],
  mask_image_id:      null,
  lora_files:         [],
  img_strength:       0.75,
  repeat_count:       1,
  auto_save:          true,
  output_dir:         '',
  upscale_enabled:    false,
  upscale_model_path: '',
  num_frames:         25,
  fps:                24,
  fast_preview:       false,
  mask_mode:          'Crop & Composite (Fast)',
  outpaint_align:     'center',
  depth_model_repo:   'depth-anything/DA3MONO-LARGE',
}

export const initialState: State = {
  params:          DEFAULT_PARAMS,
  status:          { model: null, device: null, loaded: false, busy: false, vram_gb: 0, total_vram_gb: 0 },
  models:          [],
  availableModels: [],
  devices:         [],
  workflows:       [],
  outputs:         [],
  isGenerating:    false,
  progressMsg:     '',
  progressStep:    undefined,
  progressTotal:   undefined,
  resultUrl:       null,
  resultInfo:      undefined,
  refSlots:        [],
  settingsOpen:    false,
  error:           null,
}

// ── Actions ───────────────────────────────────────────────────────────────────

export type Action =
  | { type: 'SET_PARAM';       key: keyof GenerateParams; value: unknown }
  | { type: 'SET_PARAMS';      params: Partial<GenerateParams> }
  | { type: 'SET_STATUS';      status: AppStatus }
  | { type: 'SET_MODELS';      choices: string[]; available: string[]; current: string | null }
  | { type: 'SET_DEVICES';     devices: string[] }
  | { type: 'SET_WORKFLOWS';   workflows: string[] }
  | { type: 'SET_OUTPUTS';     outputs: OutputItem[] }
  | { type: 'START_GENERATE' }
  | { type: 'STOP_GENERATE' }
  | { type: 'SET_PROGRESS';    message: string; step?: number; total?: number }
  | { type: 'ADD_RESULT';      url: string; info?: string }
  | { type: 'SET_RESULT_URL';  url: string }
  | { type: 'CLEAR_RESULT' }
  | { type: 'SET_ERROR';       message: string }
  | { type: 'TOGGLE_SETTINGS' }
  // Reference image slots
  | { type: 'ADD_REF_SLOT';         imageId: string; imageUrl: string; keepSize?: boolean }
  | { type: 'REMOVE_REF_SLOT';      slotId: number }
  | { type: 'SET_SLOT_MASK';        slotId: number; maskId: string; maskUrl: string }
  | { type: 'CLEAR_SLOT_MASK';      slotId: number }
  | { type: 'CLEAR_ALL_SLOTS' }
  | { type: 'UPDATE_SLOT_STRENGTH'; slotId: number; strength: number }
  | { type: 'SET_SLOT_DIMS';        slotId: number; w: number; h: number }

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Derive params.input_image_ids and mask_image_id from slots */
function slotsToParams(slots: RefImageSlot[]): Pick<GenerateParams, 'input_image_ids' | 'mask_image_id'> {
  return {
    input_image_ids: slots.map(s => s.imageId),
    mask_image_id:   slots[0]?.maskId ?? null,   // first slot's mask as primary mask
  }
}

/**
 * Auto-size: output width/height = slot #1's aspect at the model family's best
 * resolution (see canvasSize.ts). Runs only on real events — slot #1 gets its
 * first dims, slot #1 is removed, model family changes — so a preset picked
 * afterwards (e.g. to outpaint) is kept. Skipped for slots restored with their
 * saved size (Load Params / workflow), which carry keepSize.
 */
function autoSizeParams(params: GenerateParams, slot1: RefImageSlot | undefined): GenerateParams {
  if (!slot1?.w || !slot1?.h || slot1.keepSize) return params
  const c = canvasForRef(slot1.w, slot1.h, sizeFamily(params.model_choice))
  return { ...params, width: c.w, height: c.h }
}

// ── Reducer ───────────────────────────────────────────────────────────────────

function reducer(state: State, action: Action): State {
  switch (action.type) {

    case 'SET_PARAM': {
      const params = { ...state.params, [action.key]: action.value }
      // Budgets differ per family (LTX 768×512 vs FLUX 1 MP) → re-fit on family switch
      if (action.key === 'model_choice'
          && sizeFamily(String(action.value)) !== sizeFamily(state.params.model_choice)) {
        return { ...state, params: autoSizeParams(params, state.refSlots[0]) }
      }
      return { ...state, params }
    }

    case 'SET_PARAMS':
      return { ...state, params: { ...state.params, ...action.params } }

    case 'SET_STATUS':
      return { ...state, status: action.status }

    case 'SET_MODELS':
      return { ...state, models: action.choices, availableModels: action.available }

    case 'SET_DEVICES':
      return { ...state, devices: action.devices }

    case 'SET_WORKFLOWS':
      return { ...state, workflows: action.workflows }

    case 'SET_OUTPUTS':
      return { ...state, outputs: action.outputs }

    case 'START_GENERATE':
      return {
        ...state,
        isGenerating:  true,
        error:         null,
        progressMsg:   '',
        progressStep:  undefined,
        progressTotal: undefined,
      }

    case 'STOP_GENERATE':
      return { ...state, isGenerating: false, progressMsg: '', progressStep: undefined, progressTotal: undefined }

    case 'SET_PROGRESS':
      return { ...state, progressMsg: action.message, progressStep: action.step, progressTotal: action.total }

    case 'ADD_RESULT':
      return { ...state, resultUrl: action.url, resultInfo: action.info }

    case 'SET_RESULT_URL':
      return { ...state, resultUrl: action.url, resultInfo: undefined }

    case 'CLEAR_RESULT':
      return { ...state, resultUrl: null, resultInfo: undefined, error: null }

    case 'SET_ERROR':
      return { ...state, error: action.message }

    case 'TOGGLE_SETTINGS':
      return { ...state, settingsOpen: !state.settingsOpen }

    case 'ADD_REF_SLOT': {
      const newSlot: RefImageSlot = {
        slotId:   state.refSlots.length + 1,
        imageId:  action.imageId,
        imageUrl: action.imageUrl,
        maskId:   null,
        maskUrl:  null,
        strength: state.params.img_strength,  // inherit current global strength
        keepSize: action.keepSize,
      }
      const slots = [...state.refSlots, newSlot]
      return { ...state, refSlots: slots, params: { ...state.params, ...slotsToParams(slots) } }
    }

    case 'REMOVE_REF_SLOT': {
      const slots = state.refSlots
        .filter(s => s.slotId !== action.slotId)
        .map((s, i) => ({ ...s, slotId: i + 1 }))   // re-number
      let params = { ...state.params, ...slotsToParams(slots) }
      if (slots[0] && slots[0].imageId !== state.refSlots[0]?.imageId) {
        params = autoSizeParams(params, slots[0])   // a different image became slot #1
      }
      return { ...state, refSlots: slots, params }
    }

    case 'SET_SLOT_MASK': {
      const slots = state.refSlots.map(s =>
        s.slotId === action.slotId
          ? { ...s, maskId: action.maskId, maskUrl: action.maskUrl }
          : s
      )
      return { ...state, refSlots: slots, params: { ...state.params, ...slotsToParams(slots) } }
    }

    case 'CLEAR_SLOT_MASK': {
      const slots = state.refSlots.map(s =>
        s.slotId === action.slotId ? { ...s, maskId: null, maskUrl: null } : s
      )
      return { ...state, refSlots: slots, params: { ...state.params, ...slotsToParams(slots) } }
    }

    case 'CLEAR_ALL_SLOTS':
      return {
        ...state,
        refSlots: [],
        params: { ...state.params, input_image_ids: [], mask_image_id: null },
      }

    case 'UPDATE_SLOT_STRENGTH': {
      const slots = state.refSlots.map(s =>
        s.slotId === action.slotId ? { ...s, strength: action.strength } : s
      )
      // Slot #1's strength drives the global img_strength for single-pass generate
      const extra = action.slotId === 1 ? { img_strength: action.strength } : {}
      return { ...state, refSlots: slots, params: { ...state.params, ...extra } }
    }

    case 'SET_SLOT_DIMS': {
      const prev  = state.refSlots.find(s => s.slotId === action.slotId)
      const slots = state.refSlots.map(s =>
        s.slotId === action.slotId ? { ...s, w: action.w, h: action.h } : s
      )
      // First dims for slot #1 = a new image just loaded (thumbnail re-renders re-send
      // the same dims and must not undo a preset picked since)
      const isNewSlot1 = action.slotId === 1 && !prev?.w
      return {
        ...state,
        refSlots: slots,
        params: isNewSlot1 ? autoSizeParams(state.params, slots[0]) : state.params,
      }
    }

    default:
      return state
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useAppState() {
  const [state, dispatch] = useReducer(reducer, initialState)
  return { state, dispatch }
}
