import { useState, useRef } from 'react'
import {
  ChevronDown, ChevronRight, Play,
  Layers, Sliders, Video, UploadCloud, X, Workflow, Cpu,
  Wand2, ArrowUpCircle, FolderOpen,
} from 'lucide-react'
import type { GenerateParams } from '../types'
import { importComfyUI, loadWorkflow, saveWorkflow, uploadLora } from '../api'

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
        className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-muted hover:text-white transition-colors"
        onClick={() => setOpen(!open)}
      >
        <span className="flex items-center gap-2">{icon}{label}</span>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>
      {open && <div className="px-4 pb-4 space-y-3">{children}</div>}
    </div>
  )
}

function Slider({
  label, value, min, max, step = 1, onChange, unit = '',
}: {
  label: string; value: number; min: number; max: number
  step?: number; onChange: (v: number) => void; unit?: string
}) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <label className="text-xs text-muted">{label}</label>
        <span className="text-xs text-white">{value}{unit}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1 appearance-none bg-border rounded-full accent-accent"
      />
    </div>
  )
}

function NumberInput({
  label, value, onChange, placeholder = '',
}: {
  label: string; value: number; onChange: (v: number) => void; placeholder?: string
}) {
  return (
    <div>
      <label className="text-xs text-muted block mb-1">{label}</label>
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
    <label className="flex items-center justify-between cursor-pointer">
      <span className="text-xs text-muted">{label}</span>
      <div
        onClick={() => onChange(!value)}
        className={`relative w-9 h-5 rounded-full transition-colors ${value ? 'bg-accent' : 'bg-border'}`}
      >
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform shadow
          ${value ? 'translate-x-4' : 'translate-x-0.5'}`} />
      </div>
    </label>
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

const PRESETS = ['512×512', '768×512', '512×768', '1024×1024', '1024×576', '576×1024']

function SizePanel({ params, onChange }: { params: GenerateParams; onChange: (k: keyof GenerateParams, v: number) => void }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        {PRESETS.map(p => {
          const [w, h] = p.split('×').map(Number)
          const active = params.width === w && params.height === h
          return (
            <button
              key={p}
              onClick={() => { onChange('width', w); onChange('height', h) }}
              className={`py-1.5 rounded-md text-xs transition-colors
                ${active
                  ? 'bg-accent text-white'
                  : 'bg-card border border-border text-muted hover:text-white'}`}
            >
              {p}
            </button>
          )
        })}
      </div>
      <div className="flex gap-2">
        <NumberInput label="Width"  value={params.width}  onChange={v => onChange('width', v)} />
        <NumberInput label="Height" value={params.height} onChange={v => onChange('height', v)} />
      </div>
    </div>
  )
}

// ── LoRA panel ────────────────────────────────────────────────────────────────

interface LoraPanelProps {
  loraFile:   string | null
  strength:   number
  onChange:   (k: keyof GenerateParams, v: unknown) => void
  onStatus:   (msg: string) => void
}
function LoraPanel({ loraFile, strength, onChange, onStatus }: LoraPanelProps) {
  const [loading, setLoading] = useState(false)
  const [loraName, setLoraName] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleUpload(file: File) {
    setLoading(true)
    try {
      const { path, name } = await uploadLora(file)
      onChange('lora_file', path)
      setLoraName(name)
      onStatus(`✓ LoRA loaded: ${name}`)
    } catch (e: unknown) {
      onStatus(`LoRA error: ${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    onChange('lora_file', null)
    setLoraName(null)
    onStatus('LoRA cleared')
  }

  return (
    <div className="space-y-3">
      {loraFile ? (
        <div className="flex items-center justify-between bg-card border border-border rounded-md px-3 py-2">
          <span className="text-xs text-white truncate">{loraName ?? loraFile.split('/').pop()}</span>
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
          {loading ? 'Uploading…' : 'Upload .safetensors'}
        </button>
      )}
      <input ref={fileRef} type="file" accept=".safetensors,.pt,.bin" className="hidden"
        onChange={e => { if (e.target.files?.[0]) handleUpload(e.target.files[0]) }} />

      <Slider label="LoRA strength" value={strength} min={0} max={2} step={0.05}
        onChange={v => onChange('lora_strength', v)} />
    </div>
  )
}

// ── Upscale panel ─────────────────────────────────────────────────────────────

interface UpscalePanelProps {
  enabled:   boolean
  modelPath: string
  onChange:  (k: keyof GenerateParams, v: unknown) => void
}
function UpscalePanel({ enabled, modelPath, onChange }: UpscalePanelProps) {
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <div className="space-y-3">
      <Toggle
        label="Enable upscaling"
        value={enabled}
        onChange={v => onChange('upscale_enabled', v)}
      />
      {enabled && (
        <div>
          <label className="text-xs text-muted block mb-1">Model file</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={modelPath}
              onChange={e => onChange('upscale_model_path', e.target.value)}
              placeholder="path/to/4x_ESRGAN.pth"
              className="flex-1 min-w-0 bg-card border border-border rounded-md px-3 py-1.5 text-sm text-white
                         placeholder-muted focus:outline-none focus:border-accent transition-colors"
            />
            <button
              onClick={() => fileRef.current?.click()}
              title="Browse for model file"
              className="shrink-0 px-2.5 py-1.5 rounded-md bg-card border border-border
                         text-muted hover:text-white hover:border-accent transition-colors"
            >
              <FolderOpen size={14} />
            </button>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".pth,.pt,.onnx,.safetensors,.bin"
            className="hidden"
            onChange={e => {
              const f = e.target.files?.[0]
              if (f) onChange('upscale_model_path', f.name)
              e.target.value = ''
            }}
          />
          <p className="text-[10px] text-muted mt-1 opacity-70">
            Browse sets filename; type full path if needed.
          </p>
        </div>
      )}
    </div>
  )
}

// ── Workflow panel ────────────────────────────────────────────────────────────

interface WorkflowPanelProps {
  workflows:    string[]
  params:       GenerateParams
  onLoad:       (wf: Record<string, unknown>) => void
  onRefresh:    () => void
  onImportComfyUI: (wf: Record<string, unknown>, notes: string) => void
  onStatus:     (msg: string) => void
}
function WorkflowPanel({ workflows, params, onLoad, onRefresh, onImportComfyUI, onStatus }: WorkflowPanelProps) {
  const [selected, setSelected] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const [saveName, setSaveName] = useState('')

  async function handleLoad() {
    if (!selected) return
    try {
      const wf = await loadWorkflow(selected)
      onLoad(wf)
      onStatus(`✓ Loaded workflow: ${selected}`)
    } catch (e: unknown) {
      onStatus(`Error: ${(e as Error).message}`)
    }
  }

  async function handleSave() {
    if (!saveName.trim()) return
    try {
      await saveWorkflow({ ...params, name: saveName.trim() })
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
            className="flex-1 bg-card border border-border rounded-md px-2 py-1.5 text-sm text-white
                       focus:outline-none focus:border-accent"
          >
            <option value="">Select…</option>
            {workflows.map(w => <option key={w} value={w}>{w}</option>)}
          </select>
          <button onClick={handleLoad} disabled={!selected}
            className="px-3 py-1.5 rounded-md bg-accent text-white text-xs font-medium disabled:opacity-40">
            Load
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

// ── Main Sidebar ──────────────────────────────────────────────────────────────

interface SidebarProps {
  params:            GenerateParams
  models:            string[]
  availableModels:   string[]
  devices:           string[]
  workflows:         string[]
  isGenerating:      boolean
  onParamChange:     (k: keyof GenerateParams, v: unknown) => void
  onParamsChange:    (p: Partial<GenerateParams>) => void
  onGenerate:        () => void
  onStop:            () => void
  onWorkflowLoad:    (wf: Record<string, unknown>) => void
  onWorkflowRefresh: () => void
  onStatus:          (msg: string) => void
}

export default function Sidebar({
  params, models, availableModels, devices, workflows, isGenerating,
  onParamChange, onParamsChange, onGenerate, onStop,
  onWorkflowLoad, onWorkflowRefresh, onStatus,
}: SidebarProps) {
  const isVideo = params.model_choice.includes('LTX-Video')

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
    <aside className="w-[576px] flex flex-col bg-surface border-r border-border overflow-y-auto shrink-0">

      {/* Prompt */}
      <div className="p-4 border-b border-border">
        <label className="text-xs font-semibold text-muted uppercase tracking-wider block mb-2">Prompt</label>
        <textarea
          value={params.prompt}
          onChange={e => onParamChange('prompt', e.target.value)}
          rows={8}
          placeholder="Describe the image you want to generate…"
          className="w-full bg-card border border-border rounded-lg px-3 py-2.5 text-sm text-white
                     placeholder-muted resize-none focus:outline-none focus:border-accent transition-colors"
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
        <Slider label="Steps"    value={params.steps}    min={1}   max={50}  onChange={v => onParamChange('steps', v)} />
        <Slider label="Guidance" value={params.guidance} min={0}   max={20}  step={0.5} onChange={v => onParamChange('guidance', v)} />
        <Slider label="Repeat"   value={params.repeat_count} min={1} max={8} onChange={v => onParamChange('repeat_count', v)} />
        <NumberInput label="Seed (-1 = random)" value={params.seed} onChange={v => onParamChange('seed', v)} placeholder="-1" />
      </Accordion>

      {/* Size */}
      <Accordion label="Size" icon={<Layers size={13} />}>
        <SizePanel params={params} onChange={(k, v) => onParamChange(k, v)} />
      </Accordion>

      {/* LoRA */}
      <Accordion label="LoRA" icon={<Wand2 size={13} />}>
        <LoraPanel
          loraFile={params.lora_file}
          strength={params.lora_strength}
          onChange={onParamChange}
          onStatus={onStatus}
        />
      </Accordion>

      {/* Upscale */}
      <Accordion label="Upscale" icon={<ArrowUpCircle size={13} />}>
        <UpscalePanel
          enabled={params.upscale_enabled}
          modelPath={params.upscale_model_path}
          onChange={onParamChange}
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
          onLoad={onWorkflowLoad}
          onRefresh={onWorkflowRefresh}
          onImportComfyUI={handleImportComfyUI}
          onStatus={onStatus}
        />
      </Accordion>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Generate button */}
      <div className="p-4 border-t border-border">
        {isGenerating ? (
          <button onClick={onStop}
            className="w-full py-3 rounded-xl bg-red-600/80 hover:bg-red-600 text-white font-semibold text-sm transition-colors">
            Stop
          </button>
        ) : (
          <button onClick={onGenerate} disabled={!params.prompt.trim() || !params.model_choice}
            className="w-full py-3 rounded-xl bg-accent hover:bg-accent/80 text-white font-semibold text-sm
                       transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
            <Play size={15} />
            Generate
          </button>
        )}
      </div>
    </aside>
  )
}
