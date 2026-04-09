import { useState, useRef, useEffect } from 'react'
import {
  ChevronDown, ChevronRight, Play, Square,
  Layers, Sliders, Video, UploadCloud, X, Workflow, Cpu,
  Wand2, ArrowUpCircle, FolderInput, ListOrdered, FolderOpen, ImagePlus, Plus,
} from 'lucide-react'
import type { GenerateParams, RefImageSlot, LoraSlot } from '../types'
import { importComfyUI, loadWorkflow, saveWorkflow, uploadLora, uploadUpscaleModel, streamBatchUpscale, streamBatchGenerate, openFolderDialog, openFileDialog, upscaleSingleImage, updateSettings, openWorkflowFolderDialog, listLoras, stopGeneration, generateDepthMap } from '../api'
import HelpTip from './HelpTip'

// ── Helpers ───────────────────────────────────────────────────────────────────

interface AccordionProps {
  label:    string
  icon?:    React.ReactNode
  children: React.ReactNode
  defaultOpen?: boolean
}
function Accordion({ label, icon, children, defaultOpen = false }: AccordionProps) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b border-border last:border-0">
      <button
        className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-label hover:text-white transition-colors"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">{icon}{label}</span>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>
      {open && <div className="px-4 pb-4 space-y-3">{children}</div>}
    </div>
  )
}

function Slider({
  label, value, min, max, step = 1, onChange, unit = '', helpTip,
}: {
  label: string; value: number; min: number; max: number
  step?: number; onChange: (v: number) => void; unit?: string; helpTip?: React.ReactNode
}) {
  const id = `slider-${label.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')}`
  return (
    <div>
      <div className="flex justify-between mb-1">
        <label htmlFor={id} className="text-xs text-muted flex items-center gap-1">
          {label}{helpTip}
        </label>
        <span className="text-xs text-white" aria-hidden="true">{value}{unit}</span>
      </div>
      <input
        id={id}
        type="range" min={min} max={max} step={step} value={value}
        aria-label={`${label}: ${value}${unit}`}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1 appearance-none bg-border rounded-full accent-accent"
      />
    </div>
  )
}

function NumberInput({
  label, value, onChange, placeholder = '', helpTip,
}: {
  label: string; value: number; onChange: (v: number) => void; placeholder?: string; helpTip?: React.ReactNode
}) {
  return (
    <div>
      <label className="text-xs text-muted flex items-center gap-1 mb-1">{label}{helpTip}</label>
      <input
        type="number" value={value === -1 ? '' : value} placeholder={placeholder}
        onChange={e => onChange(e.target.value === '' ? -1 : Number(e.target.value))}
        className="w-full bg-card border border-border rounded-md px-3 py-1.5 text-sm text-white
                   focus:outline-none focus:border-accent transition-colors"
      />
    </div>
  )
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-muted">{label}</span>
      <button
        role="switch"
        aria-checked={value}
        aria-label={label}
        onClick={() => onChange(!value)}
        className={`relative w-9 h-5 rounded-full transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent/50 ${value ? 'bg-accent' : 'bg-border'}`}
      >
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform shadow
          ${value ? 'translate-x-4' : 'translate-x-0.5'}`} />
      </button>
    </div>
  )
}

// ── Model selector ────────────────────────────────────────────────────────────

interface ModelSelectorProps {
  choices:   string[]
  available: string[]
  value:     string
  device:    string
  devices:   string[]
  onChange:  (model: string, device: string) => void
}
function ModelSelector({ choices, available, value, device, devices, onChange }: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      <div>
        <label className="text-xs text-muted block mb-1">Model</label>
        <select
          value={value}
          onChange={e => onChange(e.target.value, device)}
          className="w-full bg-card border border-border rounded-md px-3 py-1.5 text-sm text-white
                     focus:outline-none focus:border-accent transition-colors"
        >
          {choices.map(c => (
            <option key={c} value={c}>
              {available.includes(c) ? '✓ ' : '↓ '}{c}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label className="text-xs text-muted block mb-1">Device</label>
        <div className="flex gap-2">
          {devices.map(d => (
            <button
              key={d}
              onClick={() => onChange(value, d)}
              className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-colors
                ${device === d
                  ? 'bg-accent text-white'
                  : 'bg-card border border-border text-muted hover:text-white'}`}
            >
              {d.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Size panel ────────────────────────────────────────────────────────────────

type SizePreset = { label: string; w: number; h: number }

const PRESETS_FLUX: SizePreset[] = [
  { label: 'Square',    w: 512,  h: 512  },
  { label: 'Landscape', w: 768,  h: 512  },
  { label: 'Portrait',  w: 512,  h: 768  },
  { label: 'Wide',      w: 896,  h: 512  },
  { label: 'Tall',      w: 512,  h: 896  },
  { label: 'HD Square', w: 1024, h: 1024 },
]

const PRESETS_ZIMAGE: SizePreset[] = [
  { label: 'Square',    w: 512, h: 512 },
  { label: 'Landscape', w: 768, h: 512 },
  { label: 'Portrait',  w: 512, h: 768 },
]

const PRESETS_LTX: SizePreset[] = [
  { label: 'Square',    w: 512, h: 512 },
  { label: 'Landscape', w: 704, h: 480 },
  { label: 'Portrait',  w: 480, h: 704 },
  { label: 'Wide',      w: 768, h: 512 },
]

function presetsForModel(model: string): SizePreset[] {
  if (model.includes('LTX'))     return PRESETS_LTX
  if (model.includes('Z-Image')) return PRESETS_ZIMAGE
  return PRESETS_FLUX
}

function snapTo64(n: number): number {
  return Math.max(64, Math.round(n / 64) * 64)
}

const ALIGN_GRID = [
  'top-left',    'top',    'top-right',
  'left',        'center', 'right',
  'bottom-left', 'bottom', 'bottom-right',
]

function AlignPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="space-y-1">
      <p className="text-[10px] text-label">Outpaint anchor</p>
      <div className="grid grid-cols-3 gap-1 w-24">
        {ALIGN_GRID.map(pos => {
          const active = value === pos
          return (
            <button
              key={pos}
              title={pos.replace('-', ' ')}
              onClick={() => onChange(pos)}
              aria-label={`Anchor ${pos}`}
              aria-pressed={active}
              className={`h-7 rounded flex items-center justify-center transition-colors
                ${active ? 'bg-accent' : 'bg-card border border-border hover:border-accent/60'}`}
            >
              <span className={`w-2 h-2 rounded-sm ${active ? 'bg-white' : 'bg-muted'}`} />
            </button>
          )
        })}
      </div>
    </div>
  )
}

function SizePanel({
  params, onChange, hasRefImage, refImageSize,
}: {
  params: GenerateParams
  onChange: (k: keyof GenerateParams, v: unknown) => void
  hasRefImage: boolean
  refImageSize?: { w: number; h: number }
}) {
  const presets = presetsForModel(params.model_choice)
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2">
        {presets.map(p => {
          const active = params.width === p.w && params.height === p.h
          return (
            <button
              key={`${p.w}×${p.h}`}
              onClick={() => { onChange('width', p.w); onChange('height', p.h) }}
              className={`py-1.5 px-1 rounded-md text-center transition-colors flex flex-col items-center gap-0.5
                ${active
                  ? 'bg-accent text-white'
                  : 'bg-card border border-border text-muted hover:text-white'}`}
            >
              <span className="text-[11px] font-medium leading-none">{p.label}</span>
              <span className={`text-[9px] leading-none ${active ? 'text-white/70' : 'text-muted/60'}`}>{p.w}×{p.h}</span>
            </button>
          )
        })}
      </div>
      <div className="flex gap-2 items-end">
        <NumberInput label="Width"  value={params.width}  onChange={v => onChange('width', v)} />
        <NumberInput label="Height" value={params.height} onChange={v => onChange('height', v)} />
        {refImageSize && (
          <button
            title={`Set to ref image size (${refImageSize.w}×${refImageSize.h} → snapped to 64)`}
            onClick={() => {
              onChange('width',  snapTo64(refImageSize.w))
              onChange('height', snapTo64(refImageSize.h))
            }}
            className="shrink-0 mb-[1px] px-2 py-1 rounded bg-card border border-border text-[10px] text-muted
                       hover:text-white hover:border-accent transition-colors whitespace-nowrap"
          >
            ↕ ref size
          </button>
        )}
      </div>
      {hasRefImage && (
        <AlignPicker
          value={params.outpaint_align}
          onChange={v => onChange('outpaint_align', v)}
        />
      )}
    </div>
  )
}

// ── LoRA panel ────────────────────────────────────────────────────────────────

const MAX_LORAS = 5

interface LoraPanelProps {
  loraFiles:   LoraSlot[]
  onChange:    (files: LoraSlot[]) => void
  onStatus:    (msg: string) => void
  modelChoice: string
}
function LoraPanel({ loraFiles, onChange, onStatus, modelChoice }: LoraPanelProps) {
  const [library, setLibrary] = useState<Array<{ name: string; path: string; model_type: string }>>([])

  // Which LoRA type is compatible with the current model?
  const compatibleType = modelChoice.startsWith('FLUX') ? 'flux'
    : (modelChoice.includes('Z-Image') && modelChoice.includes('Full')) ? 'zimage'
    : null
  // Show compatible LoRAs + unknowns (undetectable type); hide only positively incompatible ones
  const filteredLibrary = compatibleType
    ? library.filter(l => l.model_type === compatibleType || l.model_type === 'unknown')
    : library
  const [uploading, setUploading] = useState<number | null>(null)  // slot index being uploaded
  const fileRefs = useRef<(HTMLInputElement | null)[]>([])
  // Stable keys for React reconciliation — index is unreliable when removing middle slots
  const slotKeys = useRef<string[]>([])
  const keyCounter = useRef(0)

  function getKey(idx: number): string {
    while (slotKeys.current.length <= idx) {
      slotKeys.current.push(`lora-${keyCounter.current++}`)
    }
    return slotKeys.current[idx]
  }

  // Load library on mount and whenever a new file is uploaded
  function refreshLibrary() {
    listLoras().then(r => setLibrary(r.files)).catch(() => {})
  }
  useEffect(() => { refreshLibrary() }, [])

  async function handleUpload(idx: number, file: File) {
    setUploading(idx)
    try {
      const { path, name } = await uploadLora(file)
      refreshLibrary()
      const updated = loraFiles.map((s, i) => i === idx ? { ...s, path } : s)
      onChange(updated)
      onStatus(`✓ LoRA uploaded: ${name}`)
    } catch (e: unknown) {
      onStatus(`LoRA upload error: ${(e as Error).message}`)
    } finally {
      setUploading(null)
    }
  }

  function addSlot() {
    if (loraFiles.length >= MAX_LORAS) return
    onChange([...loraFiles, { path: '', strength: 1.0 }])
  }

  function removeSlot(idx: number) {
    slotKeys.current.splice(idx, 1)
    onChange(loraFiles.filter((_, i) => i !== idx))
    onStatus(loraFiles.length === 1 ? 'LoRA cleared' : `LoRA slot ${idx + 1} removed`)
  }

  function setPath(idx: number, path: string) {
    onChange(loraFiles.map((s, i) => i === idx ? { ...s, path } : s))
  }

  function setStrength(idx: number, strength: number) {
    onChange(loraFiles.map((s, i) => i === idx ? { ...s, strength } : s))
  }

  return (
    <div className="space-y-3">
      {loraFiles.map((slot, idx) => (
        <div key={getKey(idx)} className="space-y-2 bg-card border border-border rounded-md p-2.5">
          {/* Row: dropdown + upload button + remove */}
          <div className="flex items-center gap-1.5">
            <select
              value={slot.path}
              onChange={e => setPath(idx, e.target.value)}
              className="flex-1 min-w-0 bg-bg border border-border rounded px-2 py-1 text-xs text-white truncate
                         focus:outline-none focus:border-accent"
            >
              <option value="">— select a LoRA —</option>
              {/* If current path isn't in the filtered library (manually picked), show it */}
              {slot.path && !filteredLibrary.find(l => l.path === slot.path) && (
                <option value={slot.path}>{slot.path.split('/').pop()}</option>
              )}
              {filteredLibrary.map(l => (
                <option key={l.path} value={l.path}>{l.name}</option>
              ))}
            </select>

            {/* Upload new file → auto-adds to library */}
            <button
              title="Upload .safetensors from disk"
              onClick={() => fileRefs.current[idx]?.click()}
              disabled={uploading === idx}
              className="shrink-0 p-1.5 rounded border border-border text-muted hover:text-white hover:border-accent transition-colors disabled:opacity-40"
            >
              {uploading === idx ? <span className="text-[10px]">…</span> : <UploadCloud size={12} />}
            </button>

            <button
              title="Remove this LoRA slot"
              onClick={() => removeSlot(idx)}
              className="shrink-0 p-1.5 rounded border border-border text-muted hover:text-white hover:border-red-500 transition-colors"
            >
              <X size={12} />
            </button>

            <input
              ref={el => { fileRefs.current[idx] = el }}
              type="file"
              accept=".safetensors,.pt,.bin"
              className="hidden"
              onChange={e => { if (e.target.files?.[0]) handleUpload(idx, e.target.files[0]); if (e.target) e.target.value = '' }}
            />
          </div>

          {/* Strength slider */}
          <Slider
            label={`Strength ${idx + 1}`}
            value={slot.strength}
            min={0} max={2} step={0.05}
            onChange={v => setStrength(idx, v)}
            helpTip={<HelpTip text="How strongly this LoRA is applied. 0 = no effect, 1 = full strength." />}
          />
        </div>
      ))}

      {loraFiles.length < MAX_LORAS && (
        <button
          onClick={addSlot}
          className="w-full border border-dashed border-border rounded-lg py-2 flex items-center justify-center gap-2
                     text-muted hover:border-accent hover:text-white transition-colors text-xs"
        >
          <Plus size={12} />
          Add LoRA {loraFiles.length > 0 ? `(${loraFiles.length}/${MAX_LORAS})` : ''}
        </button>
      )}
    </div>
  )
}

// ── Upscale panel ─────────────────────────────────────────────────────────────

interface UpscalePanelProps {
  enabled:   boolean
  modelPath: string
  onChange:  (k: keyof GenerateParams, v: unknown) => void
  onStatus:  (msg: string) => void
}
function UpscalePanel({ enabled, modelPath, onChange, onStatus }: UpscalePanelProps) {
  const [loading, setLoading] = useState(false)
  const [modelName, setModelName] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleUpload(file: File) {
    setLoading(true)
    try {
      const { path, name } = await uploadUpscaleModel(file)
      onChange('upscale_model_path', path)
      setModelName(name)
      updateSettings({ upscale_model_path: path }).catch(() => {})
      onStatus(`✓ Upscale model loaded: ${name}`)
    } catch (e: unknown) {
      onStatus(`Upscale error: ${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    onChange('upscale_model_path', '')
    setModelName(null)
    updateSettings({ upscale_model_path: '' }).catch(() => {})
    onStatus('Upscale model cleared')
  }

  return (
    <div className="space-y-3">
      <Toggle
        label="Enable upscaling"
        value={enabled}
        onChange={v => onChange('upscale_enabled', v)}
      />
      {enabled && (
        <div className="space-y-3">
          {modelPath ? (
            <div className="flex items-center justify-between bg-card border border-border rounded-md px-3 py-2">
              <span className="text-xs text-white truncate">{modelName ?? modelPath.split('/').pop()}</span>
              <button onClick={handleClear} className="text-muted hover:text-white ml-2 shrink-0">
                <X size={13} />
              </button>
            </div>
          ) : (
            <button
              onClick={() => fileRef.current?.click()}
              disabled={loading}
              className="w-full border border-dashed border-border rounded-lg py-3 flex items-center justify-center gap-2
                         text-muted hover:border-accent hover:text-white transition-colors text-xs disabled:opacity-50"
            >
              <UploadCloud size={14} />
              {loading ? 'Uploading…' : 'Upload .pth / .safetensors'}
            </button>
          )}
          <input
            ref={fileRef}
            type="file"
            accept=".pth,.pt,.onnx,.safetensors,.bin"
            className="hidden"
            onChange={e => {
              if (e.target.files?.[0]) handleUpload(e.target.files[0])
              e.target.value = ''
            }}
          />
        </div>
      )}
    </div>
  )
}

// ── Batch Upscale panel ───────────────────────────────────────────────────────

interface BatchUpscalePanelProps {
  modelPath: string
  onStatus:  (msg: string) => void
}
function BatchUpscalePanel({ modelPath, onStatus }: BatchUpscalePanelProps) {
  const [inputFolder,    setInputFolder]    = useState('')
  const [outputFolder,   setOutputFolder]   = useState('')
  const [scale,          setScale]          = useState('×4')
  const [running,        setRunning]        = useState(false)
  const [log,            setLog]            = useState<string[]>([])
  const [pickingInput,   setPickingInput]   = useState(false)
  const [pickingOutput,  setPickingOutput]  = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const logRef   = useRef<HTMLDivElement>(null)

  async function pickFolder(setter: (p: string) => void, setPickingFlag: (v: boolean) => void) {
    setPickingFlag(true)
    try {
      const data = await openFolderDialog()
      if (!data.cancelled && data.path) setter(data.path)
    } catch (e: unknown) {
      onStatus(`Folder picker error: ${(e as Error).message}`)
    } finally {
      setPickingFlag(false)
    }
  }

  async function handleRun() {
    if (!modelPath) { onStatus('⚠ Load an upscale model first'); return }
    if (!inputFolder.trim()) { onStatus('⚠ Enter an input folder path'); return }
    setRunning(true)
    setLog([])
    abortRef.current = new AbortController()
    try {
      await streamBatchUpscale(
        { input_folder: inputFolder.trim(), output_folder: outputFolder.trim(), scale_choice: scale, model_path: modelPath },
        ev => {
          if (ev.type === 'log' && ev.message) {
            setLog(prev => {
              const next = [...prev, ev.message!]
              // auto-scroll
              setTimeout(() => { logRef.current?.scrollTo(0, logRef.current.scrollHeight) }, 0)
              return next
            })
          }
          if (ev.type === 'error') onStatus(`Batch error: ${ev.message}`)
          if (ev.type === 'done')  onStatus('✓ Batch upscale complete')
        },
        abortRef.current.signal,
      )
    } catch (e: unknown) {
      if ((e as Error).name !== 'AbortError') onStatus(`Batch error: ${(e as Error).message}`)
    } finally {
      setRunning(false)
    }
  }

  function handleStop() {
    abortRef.current?.abort()
    setRunning(false)
    onStatus('Batch upscale stopped')
  }

  return (
    <div className="space-y-3">
      {/* Input folder */}
      <div>
        <label className="text-xs text-muted block mb-1">Input folder</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={inputFolder}
            onChange={e => setInputFolder(e.target.value)}
            placeholder="/Users/you/Pictures/originals"
            className="flex-1 bg-card border border-border rounded-md px-3 py-1.5 text-xs text-white
                       placeholder-muted focus:outline-none focus:border-accent transition-colors"
          />
          <button
            onClick={() => pickFolder(setInputFolder, setPickingInput)}
            disabled={pickingInput}
            title="Browse for input folder"
            aria-label="Browse for input folder"
            className="px-2 py-1.5 rounded-md bg-card border border-border text-muted hover:text-white
                       hover:border-accent transition-colors disabled:opacity-50 shrink-0"
          >
            <FolderOpen size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Output folder */}
      <div>
        <label className="text-xs text-muted block mb-1">Output folder <span className="opacity-50">(blank = same as input)</span></label>
        <div className="flex gap-2">
          <input
            type="text"
            value={outputFolder}
            onChange={e => setOutputFolder(e.target.value)}
            placeholder="/Users/you/Pictures/upscaled"
            className="flex-1 bg-card border border-border rounded-md px-3 py-1.5 text-xs text-white
                       placeholder-muted focus:outline-none focus:border-accent transition-colors"
          />
          <button
            onClick={() => pickFolder(setOutputFolder, setPickingOutput)}
            disabled={pickingOutput}
            title="Browse for output folder"
            aria-label="Browse for output folder"
            className="px-2 py-1.5 rounded-md bg-card border border-border text-muted hover:text-white
                       hover:border-accent transition-colors disabled:opacity-50 shrink-0"
          >
            <FolderOpen size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Scale selector */}
      <div>
        <label className="text-xs text-muted block mb-1">Scale</label>
        <div className="flex gap-2">
          {['×2', '×3', '×4'].map(s => (
            <button
              key={s}
              onClick={() => setScale(s)}
              className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-colors
                ${scale === s ? 'bg-accent text-white' : 'bg-card border border-border text-muted hover:text-white'}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Model hint when none loaded */}
      {!modelPath && (
        <p className="text-[10px] text-amber-400/80">⚠ Enable upscaling and upload a model above first.</p>
      )}

      {/* Run / Stop */}
      {running ? (
        <button
          onClick={handleStop}
          className="w-full py-2 rounded-lg bg-red-600/80 hover:bg-red-600 text-white text-xs font-medium
                     flex items-center justify-center gap-2 transition-colors"
        >
          <Square size={12} /> Stop
        </button>
      ) : (
        <button
          onClick={handleRun}
          disabled={!modelPath || !inputFolder.trim()}
          className="w-full py-2 rounded-lg bg-accent hover:bg-accent/80 text-white text-xs font-medium
                     flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <FolderInput size={13} /> Run Batch Upscale
        </button>
      )}

      {/* Log */}
      {log.length > 0 && (
        <div
          ref={logRef}
          className="bg-bg border border-border rounded-md p-2 text-[10px] text-muted font-mono
                     max-h-36 overflow-y-auto space-y-0.5"
        >
          {log.map((line, i) => (
            <div key={i} className={line.includes('✓') ? 'text-green-400' : line.includes('✗') ? 'text-red-400' : ''}>
              {line}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Single Upscale panel ──────────────────────────────────────────────────────

interface SingleUpscalePanelProps {
  modelPath: string
  onStatus:  (msg: string) => void
}
function SingleUpscalePanel({ modelPath, onStatus }: SingleUpscalePanelProps) {
  const [filePath,   setFilePath]   = useState('')
  const [scale,      setScale]      = useState('×4')
  const [running,    setRunning]    = useState(false)
  const [resultPath, setResultPath] = useState<string | null>(null)
  const [picking,    setPicking]    = useState(false)

  async function handlePick() {
    setPicking(true)
    setResultPath(null)
    try {
      const data = await openFileDialog()
      if (!data.cancelled && data.path) setFilePath(data.path)
    } catch (e: unknown) {
      onStatus(`File picker error: ${(e as Error).message}`)
    } finally {
      setPicking(false)
    }
  }

  async function handleRun() {
    if (!modelPath) { onStatus('⚠ Load an upscale model first'); return }
    if (!filePath.trim()) { onStatus('⚠ Pick an image file first'); return }
    setRunning(true)
    setResultPath(null)
    try {
      const result = await upscaleSingleImage({
        source:       'path',
        file_path:    filePath.trim(),
        model_path:   modelPath,
        scale_choice: scale,
      })
      setResultPath(result.saved_path)
      onStatus(`✓ Saved: ${result.filename} (${result.width}×${result.height})`)
    } catch (e: unknown) {
      onStatus(`Upscale error: ${(e as Error).message}`)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* File picker */}
      <div>
        <label className="text-xs text-muted block mb-1">Image file</label>
        <div className="flex gap-2">
          <div className="flex-1 bg-card border border-border rounded-md px-3 py-1.5 text-xs text-white
                          placeholder-muted truncate min-w-0">
            {filePath ? filePath.split('/').pop() : <span className="text-muted">No file selected</span>}
          </div>
          <button
            onClick={handlePick}
            disabled={picking}
            title="Browse for image"
            aria-label="Browse for image"
            className="px-2 py-1.5 rounded-md bg-card border border-border text-muted hover:text-white
                       hover:border-accent transition-colors disabled:opacity-50 shrink-0"
          >
            {picking ? <span className="text-[10px]">…</span> : <ImagePlus size={13} aria-hidden="true" />}
          </button>
        </div>
        {filePath && (
          <p className="text-[10px] text-muted/60 mt-0.5 truncate font-mono" title={filePath}>{filePath}</p>
        )}
      </div>

      {/* Scale selector */}
      <div>
        <label className="text-xs text-muted block mb-1">Scale</label>
        <div className="flex gap-2">
          {['×2', '×3', '×4'].map(s => (
            <button
              key={s}
              onClick={() => setScale(s)}
              className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-colors
                ${scale === s ? 'bg-accent text-white' : 'bg-card border border-border text-muted hover:text-white'}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {!modelPath && (
        <p className="text-[10px] text-amber-400/80">⚠ Enable upscaling and upload a model above first.</p>
      )}

      <button
        onClick={handleRun}
        disabled={running || !filePath.trim() || !modelPath}
        className="w-full py-2 rounded-lg bg-accent hover:bg-accent/80 text-white text-xs font-medium
                   flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <ArrowUpCircle size={13} />
        {running ? 'Upscaling…' : 'Upscale Image'}
      </button>

      {resultPath && (
        <div className="bg-bg border border-green-800/40 rounded-md px-2 py-1.5 text-[10px] text-green-400 font-mono
                        break-all leading-snug">
          Saved: {resultPath}
        </div>
      )}
    </div>
  )
}

// ── Workflow panel ────────────────────────────────────────────────────────────

interface WorkflowPanelProps {
  workflows:    string[]
  params:       GenerateParams
  refSlots:     RefImageSlot[]
  onLoad:       (wf: Record<string, unknown>) => Promise<void>
  onRefresh:    () => void
  onImportComfyUI: (wf: Record<string, unknown>, notes: string) => void
  onStatus:     (msg: string) => void
}
function WorkflowPanel({ workflows, params, refSlots, onLoad, onRefresh, onImportComfyUI, onStatus }: WorkflowPanelProps) {
  const [selected, setSelected] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const [saveName, setSaveName] = useState('')

  async function handleLoad() {
    if (!selected) return
    try {
      const wf = await loadWorkflow(selected)
      await onLoad(wf)
    } catch (e: unknown) {
      onStatus(`Error loading workflow: ${(e as Error).message}`)
    }
  }

  async function handleOpenFolder() {
    try {
      const { path, cancelled } = await openWorkflowFolderDialog()
      if (cancelled || !path) return
      const name = path.split('/').filter(Boolean).pop() ?? ''
      if (!name) return
      const wf = await loadWorkflow(name)
      await onLoad(wf)
      setSelected(name)
    } catch (e: unknown) {
      onStatus(`Error loading workflow: ${(e as Error).message}`)
    }
  }

  async function handleSave() {
    if (!saveName.trim()) return
    try {
      await saveWorkflow({
        ...params,
        lora_files: params.lora_files.filter(s => s.path !== ''),
        name: saveName.trim(),
        ref_slots: refSlots.map(s => ({
          imageId:  s.imageId,
          maskId:   s.maskId ?? null,
          strength: s.strength,
        })),
      })
      onStatus(`✓ Saved workflow: ${saveName}`)
      setSaveName('')
      onRefresh()
    } catch (e: unknown) {
      onStatus(`Error: ${(e as Error).message}`)
    }
  }

  async function handleImportComfyUI(file: File) {
    try {
      const result = await importComfyUI(file)
      const notes = [
        result.unknown_nodes?.length > 0
          ? `⚠ ${result.unknown_nodes.length} custom node(s) skipped`
          : null,
        result.checkpoint_name !== '(unknown)'
          ? `Model: ${result.checkpoint_name} → ${result.model_choice}`
          : null,
      ].filter(Boolean).join('\n')
      onImportComfyUI(result, notes || '✓ Workflow imported')
      onStatus(notes || `✓ Imported ${file.name}`)
    } catch (e: unknown) {
      onStatus(`Import error: ${(e as Error).message}`)
    }
  }

  return (
    <div className="space-y-3">
      {/* Load */}
      <div>
        <label className="text-xs text-muted block mb-1">Load saved workflow</label>
        <div className="flex gap-2">
          <select
            value={selected}
            onChange={e => setSelected(e.target.value)}
            className="w-0 flex-1 min-w-0 bg-card border border-border rounded-md px-2 py-1.5 text-xs text-white
                       focus:outline-none focus:border-accent"
          >
            <option value="">Select…</option>
            {workflows.slice(0, 15).map(w => <option key={w} value={w}>{w}</option>)}
          </select>
          <button onClick={handleLoad} disabled={!selected}
            className="px-3 py-1.5 rounded-md bg-accent text-white text-xs font-medium disabled:opacity-40 shrink-0">
            Load
          </button>
          <button onClick={handleOpenFolder}
            title="Browse for workflow folder in Finder"
            className="px-2.5 py-1.5 rounded-md bg-card border border-border text-muted hover:text-white hover:border-accent transition-colors shrink-0">
            <FolderOpen size={14} />
          </button>
        </div>
      </div>

      {/* Save */}
      <div>
        <label className="text-xs text-muted block mb-1">Save current params</label>
        <div className="flex gap-2">
          <input value={saveName} onChange={e => setSaveName(e.target.value)}
            placeholder="Workflow name…"
            className="flex-1 bg-card border border-border rounded-md px-2 py-1.5 text-sm text-white
                       placeholder-muted focus:outline-none focus:border-accent" />
          <button onClick={handleSave} disabled={!saveName.trim()}
            className="px-3 py-1.5 rounded-md bg-card border border-border text-white text-xs font-medium
                       disabled:opacity-40 hover:border-accent transition-colors">
            Save
          </button>
        </div>
      </div>

      {/* ComfyUI import */}
      <div>
        <label className="text-xs text-muted block mb-1">Import ComfyUI workflow</label>
        <button
          onClick={() => fileRef.current?.click()}
          className="w-full border border-dashed border-border rounded-lg py-3 flex items-center justify-center gap-2
                     text-muted hover:border-accent hover:text-white transition-colors text-xs"
        >
          <UploadCloud size={14} /> Select workflow.json
        </button>
        <input ref={fileRef} type="file" accept=".json" className="hidden"
          onChange={e => { if (e.target.files?.[0]) handleImportComfyUI(e.target.files[0]) }} />
      </div>
    </div>
  )
}

// ── Depth Map panel ───────────────────────────────────────────────────────────

interface DepthMapPanelProps {
  modelRepo:     string
  onModelChange: (repo: string) => void
  onRefresh:     () => void
  onStatus:      (msg: string) => void
}
function DepthMapPanel({ modelRepo, onModelChange, onRefresh, onStatus }: DepthMapPanelProps) {
  const [filePath,   setFilePath]   = useState('')
  const [running,    setRunning]    = useState(false)
  const [resultPath, setResultPath] = useState<string | null>(null)
  const [picking,    setPicking]    = useState(false)

  async function handlePick() {
    setPicking(true)
    setResultPath(null)
    try {
      const data = await openFileDialog()
      if (!data.cancelled && data.path) setFilePath(data.path)
    } catch (e: unknown) {
      onStatus(`File picker error: ${(e as Error).message}`)
    } finally {
      setPicking(false)
    }
  }

  async function handleRun() {
    if (!filePath.trim()) { onStatus('⚠ Pick an image file first'); return }
    setRunning(true)
    setResultPath(null)
    try {
      const result = await generateDepthMap({ file_path: filePath.trim(), model_repo: modelRepo })
      onStatus(`✓ Depth map saved: ${result.filename}`)
      setResultPath(result.filename)
      onRefresh()
    } catch (e: unknown) {
      onStatus(`Depth map failed: ${(e as Error).message}`)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* Model selector */}
      <div>
        <label className="text-xs text-muted block mb-1">Model</label>
        <select
          value={modelRepo}
          onChange={e => onModelChange(e.target.value)}
          className="w-full bg-card border border-border rounded-md px-3 py-1.5 text-sm text-white
                     focus:outline-none focus:border-accent transition-colors"
        >
          <option value="depth-anything/DA3MONO-LARGE">DA3MONO-LARGE — best quality (~1.3 GB)</option>
          <option value="depth-anything/Depth-Anything-V2-Large-hf">DA2-Large — fallback (~1.3 GB)</option>
        </select>
      </div>

      {/* File picker */}
      <div>
        <label className="text-xs text-muted block mb-1">Image</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={filePath}
            onChange={e => setFilePath(e.target.value)}
            placeholder="/Users/you/Pictures/photo.png"
            className="flex-1 bg-card border border-border rounded-md px-3 py-1.5 text-xs text-white
                       placeholder-muted focus:outline-none focus:border-accent transition-colors"
          />
          <button
            onClick={handlePick}
            disabled={picking || running}
            title="Browse for image"
            aria-label="Browse for image"
            className="px-2 py-1.5 rounded-md bg-card border border-border text-muted hover:text-white
                       hover:border-accent transition-colors disabled:opacity-50 shrink-0"
          >
            <FolderOpen size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      <button
        onClick={handleRun}
        disabled={running || !filePath.trim()}
        className="w-full py-2 rounded-lg bg-accent hover:bg-accent/80 text-white text-xs font-medium
                   flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <Layers size={13} />
        {running ? 'Generating…' : 'Create Depth Map'}
      </button>

      {resultPath && (
        <div className="bg-bg border border-green-800/40 rounded-md px-2 py-1.5 text-[10px] text-green-400 font-mono
                        break-all leading-snug">
          Saved: {resultPath}
        </div>
      )}
    </div>
  )
}

// ── Batch Img2Img panel ───────────────────────────────────────────────────────

interface BatchImgImgPanelProps {
  params:        GenerateParams
  isGenerating:  boolean
  onRefresh:     () => void
  onStatus:      (msg: string) => void
}
function BatchImgImgPanel({ params, isGenerating, onRefresh, onStatus }: BatchImgImgPanelProps) {
  const [inputFolder, setInputFolder] = useState('')
  const [running,     setRunning]     = useState(false)
  const [log,         setLog]         = useState<string[]>([])
  const [progress,    setProgress]    = useState<{ current: number; total: number; filename: string } | null>(null)
  const [picking,     setPicking]     = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const logRef   = useRef<HTMLDivElement>(null)

  async function pickFolder() {
    setPicking(true)
    try {
      const data = await openFolderDialog()
      if (!data.cancelled && data.path) setInputFolder(data.path)
    } catch (e: unknown) {
      onStatus(`Folder picker error: ${(e as Error).message}`)
    } finally {
      setPicking(false)
    }
  }

  async function handleRun() {
    if (!inputFolder.trim()) { onStatus('⚠ Enter an input folder path'); return }
    setRunning(true)
    setLog([])
    setProgress(null)
    abortRef.current = new AbortController()
    try {
      await streamBatchGenerate(
        params,
        inputFolder.trim(),
        ev => {
          if (ev.type === 'batch_progress') {
            setProgress({ current: ev.current, total: ev.total, filename: ev.filename })
            setLog(prev => {
              const next = [...prev, `[${ev.current}/${ev.total}] ${ev.filename}`]
              setTimeout(() => { logRef.current?.scrollTo(0, logRef.current.scrollHeight) }, 0)
              return next
            })
          } else if (ev.type === 'image' || ev.type === 'video') {
            onRefresh()
          } else if (ev.type === 'error') {
            setLog(prev => [...prev, `✗ ${ev.message ?? 'Unknown error'}`])
          } else if (ev.type === 'done') {
            const processed = (ev as { type: string; processed?: number }).processed
            onStatus(`✓ Batch complete — ${processed ?? '?'} images generated`)
            setProgress(null)
          }
        },
        abortRef.current.signal,
      )
    } catch (e: unknown) {
      if ((e as Error).name !== 'AbortError') onStatus(`Batch error: ${(e as Error).message}`)
    } finally {
      setRunning(false)
    }
  }

  function handleStop() {
    stopGeneration()
    abortRef.current?.abort()
    setRunning(false)
    setProgress(null)
    onStatus('Batch stopped')
  }

  const disabled = isGenerating || running

  return (
    <div className="space-y-3">
      {/* Input folder */}
      <div>
        <label className="text-xs text-muted block mb-1">Input folder</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={inputFolder}
            onChange={e => setInputFolder(e.target.value)}
            placeholder="/Users/you/Pictures/batch"
            disabled={disabled}
            className="flex-1 bg-card border border-border rounded-md px-3 py-1.5 text-xs text-white
                       placeholder-muted focus:outline-none focus:border-accent transition-colors disabled:opacity-50"
          />
          <button
            onClick={pickFolder}
            disabled={disabled || picking}
            title="Browse for input folder"
            aria-label="Browse for input folder"
            className="px-2 py-1.5 rounded-md bg-card border border-border text-muted hover:text-white
                       hover:border-accent transition-colors disabled:opacity-50 shrink-0"
          >
            <FolderOpen size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Progress */}
      {progress && (
        <p className="text-[11px] text-muted font-mono truncate">
          Image {progress.current}/{progress.total} — {progress.filename}
        </p>
      )}

      {/* Run / Stop */}
      {running ? (
        <button
          onClick={handleStop}
          className="w-full py-2 rounded-lg bg-red-600/80 hover:bg-red-600 text-white text-xs font-medium
                     flex items-center justify-center gap-2 transition-colors"
        >
          <Square size={12} /> Stop
        </button>
      ) : (
        <button
          onClick={handleRun}
          disabled={isGenerating || !inputFolder.trim()}
          className="w-full py-2 rounded-lg bg-accent hover:bg-accent/80 text-white text-xs font-medium
                     flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <FolderInput size={13} /> Run Batch Img2Img
        </button>
      )}

      {/* Log */}
      {log.length > 0 && (
        <div
          ref={logRef}
          className="bg-bg border border-border rounded-md p-2 text-[10px] text-muted font-mono
                     max-h-36 overflow-y-auto space-y-0.5"
        >
          {log.map((line, i) => (
            <div key={i} className={line.startsWith('✗') ? 'text-red-400' : line.includes('✓') ? 'text-green-400' : ''}>
              {line}
            </div>
          ))}
        </div>
      )}

      {isGenerating && !running && (
        <p className="text-[10px] text-amber-400/80">⚠ Wait for current generation to finish before running batch.</p>
      )}
    </div>
  )
}

// ── Main Sidebar ──────────────────────────────────────────────────────────────

interface SidebarProps {
  params:               GenerateParams
  models:               string[]
  availableModels:      string[]
  devices:              string[]
  workflows:            string[]
  isGenerating:         boolean
  hasIteratableMasks:   boolean                    // true when ≥1 ref slot has a mask → show Iterate button
  hasRefImage:          boolean                    // true when slot #1 has an image → show outpaint anchor
  refImageSize?:        { w: number; h: number }   // slot #1 natural dims for "Use ref size" button
  refSlots:             RefImageSlot[]
  onParamChange:        (k: keyof GenerateParams, v: unknown) => void
  onParamsChange:       (p: Partial<GenerateParams>) => void
  onGenerate:           () => void
  onStop:               () => void
  onIterate:            () => void
  onWorkflowLoad:       (wf: Record<string, unknown>) => Promise<void>
  onWorkflowRefresh:    () => void
  onRefresh:            () => void
  onStatus:             (msg: string) => void
}

export default function Sidebar({
  params, models, availableModels, devices, workflows, isGenerating,
  hasIteratableMasks, hasRefImage, refImageSize, refSlots,
  onParamChange, onParamsChange, onGenerate, onStop, onIterate,
  onWorkflowLoad, onWorkflowRefresh, onRefresh, onStatus,
}: SidebarProps) {
  const isVideo    = params.model_choice.includes('LTX-Video')
  const isFlux     = params.model_choice.startsWith('FLUX')
  const isZImageFull = params.model_choice.includes('Z-Image') && params.model_choice.includes('Full')

  function handleImportComfyUI(wf: Record<string, unknown>) {
    const p: Partial<GenerateParams> = {}
    if (wf.prompt)       p.prompt       = String(wf.prompt)
    if (wf.height)       p.height       = Number(wf.height)
    if (wf.width)        p.width        = Number(wf.width)
    if (wf.steps)        p.steps        = Number(wf.steps)
    if (wf.seed)         p.seed         = Number(wf.seed)
    if (wf.guidance)     p.guidance     = Number(wf.guidance)
    if (wf.model_choice) p.model_choice = String(wf.model_choice)
    onParamsChange(p)
  }

  return (
    <aside aria-label="Generation controls" className="w-[576px] flex flex-col bg-surface border-r border-border overflow-y-auto shrink-0">

      {/* Prompt */}
      <div className="p-4 border-b border-border">
        <label className="text-xs font-semibold text-label uppercase tracking-wider block mb-2">Prompt</label>
        <textarea
          value={params.prompt}
          onChange={e => onParamChange('prompt', e.target.value)}
          rows={8}
          placeholder="Describe the image you want to generate…"
          className="w-full bg-card border border-border rounded-lg px-3 py-2.5 text-sm text-white
                     placeholder-muted resize-y min-h-[5rem] focus:outline-none focus:border-accent transition-colors"
        />
      </div>

      {/* Model */}
      <Accordion label="Model" icon={<Cpu size={13} />} defaultOpen>
        <ModelSelector
          choices={models} available={availableModels}
          value={params.model_choice} device={params.device} devices={devices}
          onChange={(model, device) => { onParamChange('model_choice', model); onParamChange('device', device) }}
        />
      </Accordion>

      {/* Params */}
      <Accordion label="Parameters" icon={<Sliders size={13} />} defaultOpen>
        <Slider label="Steps"    value={params.steps}    min={1}   max={50}  onChange={v => onParamChange('steps', v)}
          helpTip={<HelpTip text="Denoising iterations. More steps = more detail but slower. FLUX.2: 20, Z-Image Turbo: 4, LTX-Video: 25. Going beyond the recommended value rarely helps." />} />
        {/* Guidance slider hidden for quantized models (all current models). Value is still
            sent to the backend and auto-set by guidanceForModel(). Unhide here when adding
            full-precision non-distilled models where guidance meaningfully affects output. */}
        <Slider label="Repeat"   value={params.repeat_count} min={1} max={8} onChange={v => onParamChange('repeat_count', v)}
          helpTip={<HelpTip text="Generate N images in sequence with the same settings (different seeds if seed is -1). Each result is saved individually." />} />
        <NumberInput label="Seed (-1 = random)" value={params.seed} onChange={v => onParamChange('seed', v)} placeholder="-1"
          helpTip={<HelpTip text="Random seed. -1 = new random seed each time. Set a fixed number to reproduce the same image with different settings." />} />
      </Accordion>

      {/* Output Size */}
      <Accordion label="Output Size" icon={<Layers size={13} />}>
        <SizePanel params={params} onChange={onParamChange} hasRefImage={hasRefImage} refImageSize={refImageSize} />
      </Accordion>

      {/* LoRA */}
      {(isZImageFull || isFlux) && (
        <Accordion label="LoRA" icon={<Wand2 size={13} />}>
          <LoraPanel
            loraFiles={params.lora_files}
            onChange={files => onParamChange('lora_files', files)}
            onStatus={onStatus}
            modelChoice={params.model_choice}
          />
          {isFlux && !isZImageFull && (
            <p className="text-[11px] text-[var(--color-muted)] flex items-start gap-1 mt-1">
              <HelpTip text="LoRAs trained for standard FLUX.1 may not be compatible with FLUX.2-klein. If loading fails, try a LoRA trained specifically for FLUX.2-klein." position="right" />
              <span>FLUX.2-klein LoRA compatibility varies by trainer.</span>
            </p>
          )}
        </Accordion>
      )}

      {/* Upscale */}
      <Accordion label="Upscale" icon={<ArrowUpCircle size={13} />}>
        <UpscalePanel
          enabled={params.upscale_enabled}
          modelPath={params.upscale_model_path}
          onChange={onParamChange}
          onStatus={onStatus}
        />
        <div className="border-t border-border pt-3 mt-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-2">Single image</p>
          <SingleUpscalePanel
            modelPath={params.upscale_model_path}
            onStatus={onStatus}
          />
        </div>
        <div className="border-t border-border pt-3 mt-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-2">Batch folder upscale</p>
          <BatchUpscalePanel
            modelPath={params.upscale_model_path}
            onStatus={onStatus}
          />
        </div>
      </Accordion>

      {/* Batch Img2Img */}
      <Accordion label="Batch Img2Img" icon={<FolderInput size={13} />}>
        <BatchImgImgPanel
          params={params}
          isGenerating={isGenerating}
          onRefresh={onRefresh}
          onStatus={onStatus}
        />
      </Accordion>

      {/* Depth Map */}
      <Accordion label="Depth Map" icon={<Layers size={13} />}>
        <DepthMapPanel
          modelRepo={params.depth_model_repo}
          onModelChange={async (repo) => {
            onParamChange('depth_model_repo', repo)
            try { await updateSettings({ depth_model_repo: repo }) } catch { /* non-fatal */ }
          }}
          onRefresh={onRefresh}
          onStatus={onStatus}
        />
      </Accordion>

      {/* Video */}
      {isVideo && (
        <Accordion label="Video" icon={<Video size={13} />}>
          <Slider label="Frames" value={params.num_frames} min={9} max={121} step={8}
            onChange={v => onParamChange('num_frames', v)} />
          <Slider label="FPS"    value={params.fps}        min={4} max={60}
            onChange={v => onParamChange('fps', v)} />
        </Accordion>
      )}

      {/* Workflow */}
      <Accordion label="Workflows" icon={<Workflow size={13} />}>
        <WorkflowPanel
          workflows={workflows}
          params={params}
          refSlots={refSlots}
          onLoad={onWorkflowLoad}
          onRefresh={onWorkflowRefresh}
          onImportComfyUI={handleImportComfyUI}
          onStatus={onStatus}
        />
      </Accordion>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Generate / Iterate / Stop — single button that adapts to context */}
      <div className="p-4 border-t border-border space-y-2">
        {params.model_choice.startsWith('FLUX') &&
         params.mask_mode === 'Inpainting Pipeline (Quality)' && (
          <p className="text-[10px] text-amber-400/70 bg-amber-900/20 border border-amber-800/30
                        rounded px-2 py-1 leading-tight">
            ⚠ Inpainting Pipeline unavailable for FLUX.2-klein — will use img2img
          </p>
        )}
        {isGenerating ? (
          <button onClick={onStop}
            className="w-full py-3 rounded-xl bg-red-600/80 hover:bg-red-600 text-white font-semibold text-sm transition-colors">
            Stop
          </button>
        ) : (() => {
          // Rename to "Iterate Masks" only when Inpainting Pipeline mode is active AND masks exist
          const useIterate = hasIteratableMasks && params.mask_mode === 'Inpainting Pipeline (Quality)'
          return (
            <button
              onClick={useIterate ? onIterate : onGenerate}
              disabled={!params.prompt.trim() || !params.model_choice}
              className="w-full py-3 rounded-xl bg-accent hover:bg-accent/80 text-white font-semibold text-sm
                         transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {useIterate ? <ListOrdered size={15} /> : <Play size={15} />}
              {useIterate ? 'Iterate Masks' : 'Generate'}
            </button>
          )
        })()}
      </div>
    </aside>
  )
}
