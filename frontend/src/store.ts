import { useReducer } from 'react'
import type { GenerateParams, OutputItem, AppStatus, RefImageSlot, IpAdapterSlot, IpAdapterStatus } from './types'

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
  // IP-Adapter
  ipAdapterSlots:   IpAdapterSlot[]
  ipAdapterEnabled: boolean
  ipAdapterStatus:  IpAdapterStatus | null
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
  lora_file:          null,
  lora_strength:      1.0,
  img_strength:       0.75,
  repeat_count:       1,
  auto_save:          true,
  output_dir:         '',
  upscale_enabled:    false,
  upscale_model_path: '',
  num_frames:         25,
  fps:                24,
  mask_mode:          'Crop & Composite (Fast)',
  outpaint_align:     'center',
  ip_adapter_image_ids: [],
  ip_adapter_scales:    [],
  ip_adapter_enabled:   false,
}

export const initialState: State = {
  params:          DEFAULT_PARAMS,
  status:          { model: null, device: null, loaded: false, busy: false, vram_gb: 0 },
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
  ipAdapterSlots:   [],
  ipAdapterEnabled: false,
  ipAdapterStatus:  null,
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
  | { type: 'ADD_REF_SLOT';         imageId: string; imageUrl: string }
  | { type: 'REMOVE_REF_SLOT';      slotId: number }
  | { type: 'SET_SLOT_MASK';        slotId: number; maskId: string; maskUrl: string }
  | { type: 'CLEAR_SLOT_MASK';      slotId: number }
  | { type: 'CLEAR_ALL_SLOTS' }
  | { type: 'UPDATE_SLOT_STRENGTH'; slotId: number; strength: number }
  | { type: 'ADD_IPA_SLOT';     imageId: string; imageUrl: string }
  | { type: 'REMOVE_IPA_SLOT';  slotId: number }
  | { type: 'UPDATE_IPA_SCALE'; slotId: number; scale: number }
  | { type: 'CLEAR_IPA_SLOTS' }
  | { type: 'TOGGLE_IPA' }
  | { type: 'SET_IPA_STATUS';   status: IpAdapterStatus }

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Derive params.input_image_ids and mask_image_id from slots */
function slotsToParams(slots: RefImageSlot[]): Pick<GenerateParams, 'input_image_ids' | 'mask_image_id'> {
  return {
    input_image_ids: slots.map(s => s.imageId),
    mask_image_id:   slots[0]?.maskId ?? null,   // first slot's mask as primary mask
  }
}

function ipaSlotsToParams(slots: IpAdapterSlot[], enabled: boolean):
  Pick<GenerateParams, 'ip_adapter_image_ids' | 'ip_adapter_scales' | 'ip_adapter_enabled'> {
  return {
    ip_adapter_image_ids: slots.map(s => s.imageId),
    ip_adapter_scales:    slots.map(s => s.scale),
    ip_adapter_enabled:   enabled && slots.length > 0,
  }
}

// ── Reducer ───────────────────────────────────────────────────────────────────

function reducer(state: State, action: Action): State {
  switch (action.type) {

    case 'SET_PARAM':
      return { ...state, params: { ...state.params, [action.key]: action.value } }

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
      }
      const slots = [...state.refSlots, newSlot]
      return { ...state, refSlots: slots, params: { ...state.params, ...slotsToParams(slots) } }
    }

    case 'REMOVE_REF_SLOT': {
      const slots = state.refSlots
        .filter(s => s.slotId !== action.slotId)
        .map((s, i) => ({ ...s, slotId: i + 1 }))   // re-number
      return { ...state, refSlots: slots, params: { ...state.params, ...slotsToParams(slots) } }
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

    case 'ADD_IPA_SLOT': {
      if (state.ipAdapterSlots.length >= 3) return state
      const slot: IpAdapterSlot = {
        slotId:   state.ipAdapterSlots.length + 1,
        imageId:  action.imageId,
        imageUrl: action.imageUrl,
        scale:    0.6,
      }
      const slots = [...state.ipAdapterSlots, slot]
      return { ...state, ipAdapterSlots: slots,
        params: { ...state.params, ...ipaSlotsToParams(slots, state.ipAdapterEnabled) } }
    }

    case 'REMOVE_IPA_SLOT': {
      const slots = state.ipAdapterSlots
        .filter(s => s.slotId !== action.slotId)
        .map((s, i) => ({ ...s, slotId: i + 1 }))
      return { ...state, ipAdapterSlots: slots,
        params: { ...state.params, ...ipaSlotsToParams(slots, state.ipAdapterEnabled) } }
    }

    case 'UPDATE_IPA_SCALE': {
      const slots = state.ipAdapterSlots.map(s =>
        s.slotId === action.slotId ? { ...s, scale: action.scale } : s)
      return { ...state, ipAdapterSlots: slots,
        params: { ...state.params, ...ipaSlotsToParams(slots, state.ipAdapterEnabled) } }
    }

    case 'CLEAR_IPA_SLOTS':
      return { ...state, ipAdapterSlots: [],
        params: { ...state.params, ...ipaSlotsToParams([], state.ipAdapterEnabled) } }

    case 'TOGGLE_IPA': {
      const enabled = !state.ipAdapterEnabled
      return { ...state, ipAdapterEnabled: enabled,
        params: { ...state.params, ...ipaSlotsToParams(state.ipAdapterSlots, enabled) } }
    }

    case 'SET_IPA_STATUS':
      return { ...state, ipAdapterStatus: action.status }

    default:
      return state
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useAppState() {
  const [state, dispatch] = useReducer(reducer, initialState)
  return { state, dispatch }
}
