import { useEffect, useRef, useCallback, useState } from 'react'
import { useAppState } from './store'
import type { GenerateParams, SSEEvent, OutputItem } from './types'
import {
  fetchStatus, fetchModels, fetchDevices, fetchWorkflows,
  fetchOutputs, uploadImage, uploadFromUrl, streamGenerate, pingServer,
  fetchSettings, deleteOutput, upscaleSingleImage,
} from './api'

function hexToRgbVar(hex: string): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `${r} ${g} ${b}`
}

export function applyThemeColors(colors: Record<string, string>) {
  const el = document.documentElement
  for (const [k, v] of Object.entries(colors)) {
    if (v.startsWith('#')) el.style.setProperty(`--color-${k}`, hexToRgbVar(v))
  }
}
import TopBar        from './components/TopBar'
import Sidebar       from './components/Sidebar'
import Canvas        from './components/Canvas'
import RefImagesRow  from './components/RefImagesRow'
import Gallery       from './components/Gallery'
import SettingsDrawer from './components/SettingsDrawer'

// ── Row resize drag handle ─────────────────────────────────────────────────────

function RowDivider({
  onDragStart,
  onKeyResize,
  label,
}: {
  onDragStart: (e: React.MouseEvent) => void
  onKeyResize: (delta: number) => void
  label: string
}) {
  return (
    <div
      role="separator"
      tabIndex={0}
      aria-label={label}
      onMouseDown={onDragStart}
      onKeyDown={e => {
        if (e.key === 'ArrowUp')   { e.preventDefault(); onKeyResize(-2) }
        if (e.key === 'ArrowDown') { e.preventDefault(); onKeyResize(2) }
      }}
      className="shrink-0 h-2 cursor-row-resize flex items-center justify-center
                 bg-border/30 hover:bg-accent/40 focus:bg-accent/30 focus:outline-none transition-colors group"
    >
      <div className="w-10 h-0.5 rounded-full bg-muted/30 group-hover:bg-accent/70 transition-colors" />
    </div>
  )
}

function guidanceForModel(model: string): number {
  if (model.includes('Z-Image')) return 0
  if (model.includes('LTX-Video')) return 3.0
  return 3.5
}

function stepsForModel(model: string): number {
  if (model.includes('Z-Image')) return 4   // distilled — more steps hurt quality
  if (model.includes('LTX-Video')) return 25
  return 20  // FLUX.2 variants
}

export default function App() {
  const { state, dispatch } = useAppState()
  const abortRef    = useRef<AbortController | null>(null)
  const [statusMsg, setStatusMsg] = useState('')
  const [upscalingGalleryUrl, setUpscalingGalleryUrl] = useState<string | null>(null)
  const centerRef   = useRef<HTMLDivElement>(null)
  const [rowPcts, setRowPcts] = useState<[number, number, number]>([50, 36, 14])

  function keyResize(divider: 0 | 1, delta: number) {
    setRowPcts(([s0, s1, s2]) => {
      if (divider === 0) {
        const a = Math.max(10, Math.min(80, s0 + delta))
        const b = Math.max(5,  Math.min(80, s1 - delta))
        const c = Math.max(4, 100 - a - b)
        return [a, b, c]
      } else {
        const b = Math.max(5,  Math.min(80, s1 + delta))
        const c = Math.max(4,  Math.min(55, s2 - delta))
        const a = Math.max(10, 100 - b - c)
        return [a, b, c]
      }
    })
  }

  function startRowDrag(divider: 0 | 1) {
    return (e: React.MouseEvent) => {
      e.preventDefault()
      const startY = e.clientY
      const totalH = centerRef.current?.offsetHeight ?? 600
      const [s0, s1, s2] = rowPcts

      const onMove = (ev: MouseEvent) => {
        const dp = ((ev.clientY - startY) / totalH) * 100
        if (divider === 0) {
          const a = Math.max(10, Math.min(80, s0 + dp))
          const b = Math.max(5,  Math.min(80, s1 - dp))
          const c = Math.max(4, 100 - a - b)
          setRowPcts([a, b, c])
        } else {
          const b = Math.max(5,  Math.min(80, s1 + dp))
          const c = Math.max(4,  Math.min(55, s2 - dp))
          const a = Math.max(10, 100 - b - c)
          setRowPcts([a, b, c])
        }
      }
      const onUp = () => {
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        window.removeEventListener('mousemove', onMove)
        window.removeEventListener('mouseup', onUp)
      }
      document.body.style.cursor = 'row-resize'
      document.body.style.userSelect = 'none'
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseup', onUp)
    }
  }

  // ── Bootstrap ─────────────────────────────────────────────────────────────

  const refreshOutputs = useCallback(async () => {
    try {
      const { files } = await fetchOutputs(30)
      dispatch({ type: 'SET_OUTPUTS', outputs: files })
    } catch { /* ignore */ }
  }, [dispatch])

  const refreshWorkflows = useCallback(async () => {
    try {
      const { workflows } = await fetchWorkflows()
      dispatch({ type: 'SET_WORKFLOWS', workflows })
    } catch { /* ignore */ }
  }, [dispatch])

  useEffect(() => {
    fetchModels().then(data => {
      dispatch({ type: 'SET_MODELS', choices: data.choices, available: data.available, current: data.current })
      const m = data.current || data.choices[0] || ''
      if (m) {
        dispatch({ type: 'SET_PARAM', key: 'model_choice', value: m })
        dispatch({ type: 'SET_PARAM', key: 'guidance', value: guidanceForModel(m) })
        dispatch({ type: 'SET_PARAM', key: 'steps',    value: stepsForModel(m) })
      }
    }).catch(() => {})

    fetchDevices().then(({ devices }) => {
      dispatch({ type: 'SET_DEVICES', devices })
      if (devices.length > 0) {
        dispatch({ type: 'SET_PARAM', key: 'device', value: devices[0] })
      }
    }).catch(() => {})

    fetchSettings().then(s => {
      if (s.theme_colors && typeof s.theme_colors === 'object') {
        applyThemeColors(s.theme_colors as Record<string, string>)
      }
      // support both old key ('upscaler_model_path') and new key ('upscale_model_path')
      const savedModel = (s.upscale_model_path || s.upscaler_model_path) as string | undefined
      if (savedModel) {
        dispatch({ type: 'SET_PARAM', key: 'upscale_model_path', value: savedModel })
      }
    }).catch(() => {})

    refreshWorkflows()
    refreshOutputs()

    const pollStatus = async () => {
      try {
        const s = await fetchStatus()
        dispatch({ type: 'SET_STATUS', status: s })
      } catch { /* ignore */ }
    }
    pollStatus()
    const interval = setInterval(pollStatus, 4000)
    return () => clearInterval(interval)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Browser heartbeat — keeps server alive while the tab is open ───────────

  useEffect(() => {
    pingServer()                                    // immediate ping on mount
    const id = setInterval(pingServer, 5000)        // then every 5 s
    return () => clearInterval(id)
  }, [])

  // ── Generation ────────────────────────────────────────────────────────────

  const handleGenerate = useCallback(async () => {
    if (state.isGenerating) return
    dispatch({ type: 'START_GENERATE' })

    abortRef.current = new AbortController()
    const params: GenerateParams = { ...state.params }

    try {
      const onEvent = (e: SSEEvent) => {
        switch (e.type) {
          case 'progress':
            dispatch({ type: 'SET_PROGRESS', message: e.message, step: e.step, total: e.total })
            break
          case 'image':
            dispatch({ type: 'ADD_RESULT', url: e.url, info: e.info })
            refreshOutputs()
            break
          case 'video':
            dispatch({ type: 'ADD_RESULT', url: e.url })
            refreshOutputs()
            break
          case 'error':
            dispatch({ type: 'SET_ERROR', message: e.message })
            break
          case 'done':
            break
        }
      }
      await streamGenerate(params, onEvent, abortRef.current.signal)
    } catch (err: unknown) {
      const msg = (err as Error).message ?? 'Unknown error'
      if (msg !== 'The user aborted a request.') {
        dispatch({ type: 'SET_ERROR', message: msg })
      }
    } finally {
      dispatch({ type: 'STOP_GENERATE' })
      abortRef.current = null
    }
  }, [state.isGenerating, state.params, dispatch, refreshOutputs])

  const handleStop = useCallback(() => {
    abortRef.current?.abort()
    dispatch({ type: 'STOP_GENERATE' })
  }, [dispatch])

  // ── Reference image slots ──────────────────────────────────────────────────

  const handleAddRefSlots = useCallback(async (files: File[]) => {
    for (const file of files) {
      try {
        const { id, url } = await uploadImage(file)
        dispatch({ type: 'ADD_REF_SLOT', imageId: id, imageUrl: url })
      } catch (err: unknown) {
        dispatch({ type: 'SET_ERROR', message: (err as Error).message })
      }
    }
  }, [dispatch])

  const handleRemoveRefSlot = useCallback((slotId: number) => {
    dispatch({ type: 'REMOVE_REF_SLOT', slotId })
  }, [dispatch])

  const handleUploadSlotMask = useCallback(async (slotId: number, file: File) => {
    try {
      const { id, url } = await uploadImage(file)
      dispatch({ type: 'SET_SLOT_MASK', slotId, maskId: id, maskUrl: url })
    } catch (err: unknown) {
      dispatch({ type: 'SET_ERROR', message: (err as Error).message })
    }
  }, [dispatch])

  const handleClearSlotMask = useCallback((slotId: number) => {
    dispatch({ type: 'CLEAR_SLOT_MASK', slotId })
  }, [dispatch])

  const handleSlotStrengthChange = useCallback((slotId: number, strength: number) => {
    dispatch({ type: 'UPDATE_SLOT_STRENGTH', slotId, strength })
  }, [dispatch])

  // ── Iterative multi-mask generation ───────────────────────────────────────
  //
  // For each slot that has a mask (in slotId order):
  //   Pass 1  → use slot #1's image as base input + this slot as reference
  //   Pass N  → re-upload previous output as base + this slot as reference
  //
  const handleIterateGenerate = useCallback(async () => {
    if (state.isGenerating) return

    const slotsWithMasks = state.refSlots.filter(s => s.maskId)
    if (slotsWithMasks.length === 0) return
    if (state.refSlots.length === 0) return

    dispatch({ type: 'START_GENERATE' })
    abortRef.current = new AbortController()

    try {
      let currentOutputUrl: string | null = null

      for (let i = 0; i < slotsWithMasks.length; i++) {
        const slot = slotsWithMasks[i]
        const passNum    = i + 1
        const totalPasses = slotsWithMasks.length

        dispatch({ type: 'SET_PROGRESS', message: `Pass ${passNum}/${totalPasses} — applying mask on slot #${slot.slotId}…` })

        // Build input image list for this pass
        let inputImageIds: string[]
        if (i === 0) {
          // First pass: always use slot #1's image as the base being edited
          const baseId = state.refSlots[0].imageId
          inputImageIds = slot.slotId === 1 ? [baseId] : [baseId, slot.imageId]
        } else {
          // Subsequent passes: re-upload previous output as the new base
          const { id: prevId } = await uploadFromUrl(currentOutputUrl!)
          inputImageIds = slot.slotId === 1 ? [prevId] : [prevId, slot.imageId]
        }

        const passParams: GenerateParams = {
          ...state.params,
          input_image_ids: inputImageIds,
          mask_image_id:   slot.maskId,
          img_strength:    slot.strength,
        }

        let passOutputUrl: string | null = null

        const onEvent = (e: SSEEvent) => {
          switch (e.type) {
            case 'progress':
              dispatch({ type: 'SET_PROGRESS', message: `Pass ${passNum}/${totalPasses}: ${e.message}`, step: e.step, total: e.total })
              break
            case 'image':
              passOutputUrl = e.url
              dispatch({ type: 'ADD_RESULT', url: e.url, info: e.info })
              refreshOutputs()
              break
            case 'video':
              passOutputUrl = e.url
              dispatch({ type: 'ADD_RESULT', url: e.url })
              refreshOutputs()
              break
            case 'error':
              dispatch({ type: 'SET_ERROR', message: e.message })
              break
          }
        }

        await streamGenerate(passParams, onEvent, abortRef.current.signal)

        if (!passOutputUrl) break          // stopped or failed — abort loop
        currentOutputUrl = passOutputUrl
      }
    } catch (err: unknown) {
      const msg = (err as Error).message ?? 'Unknown error'
      if (msg !== 'The user aborted a request.') {
        dispatch({ type: 'SET_ERROR', message: msg })
      }
    } finally {
      dispatch({ type: 'STOP_GENERATE' })
      abortRef.current = null
    }
  }, [state.isGenerating, state.params, state.refSlots, dispatch, refreshOutputs])

  // ── Delete output ─────────────────────────────────────────────────────────

  const handleDeleteOutput = useCallback(async (filename: string) => {
    try {
      await deleteOutput(filename)
      if (state.resultUrl === `/api/output/${filename}`) {
        dispatch({ type: 'CLEAR_RESULT' })
      }
      await refreshOutputs()
    } catch (err: unknown) {
      setStatusMsg(`Delete failed: ${(err as Error).message}`)
    }
  }, [state.resultUrl, dispatch, refreshOutputs])

  // ── Gallery upscale ────────────────────────────────────────────────────────

  const handleUpscaleGalleryItem = useCallback(async (item: OutputItem) => {
    setUpscalingGalleryUrl(item.url)
    try {
      await upscaleSingleImage({
        source:       'gallery',
        filename:     item.name,
        model_path:   state.params.upscale_model_path,
        scale_choice: '×4',
      })
      await refreshOutputs()
    } catch (err: unknown) {
      dispatch({ type: 'SET_ERROR', message: (err as Error).message })
    } finally {
      setUpscalingGalleryUrl(null)
    }
  }, [state.params.upscale_model_path, dispatch, refreshOutputs])

  // ── Gallery select: load result + inject prompt/model ─────────────────────

  const handleSelectGallery = useCallback((item: OutputItem) => {
    dispatch({ type: 'SET_RESULT_URL', url: item.url })
    if (item.prompt) {
      dispatch({ type: 'SET_PARAM', key: 'prompt', value: item.prompt })
    }
    if (item.model_choice) {
      dispatch({ type: 'SET_PARAM', key: 'model_choice', value: item.model_choice })
    }
  }, [dispatch])

  // ── Workflow ──────────────────────────────────────────────────────────────

  const handleWorkflowLoad = useCallback((wf: Record<string, unknown>) => {
    const p: Partial<GenerateParams> = {}
    if (wf.prompt)       p.prompt       = String(wf.prompt)
    if (wf.height)       p.height       = Number(wf.height)
    if (wf.width)        p.width        = Number(wf.width)
    if (wf.steps)        p.steps        = Number(wf.steps)
    if (wf.seed)         p.seed         = Number(wf.seed)
    if (wf.guidance)     p.guidance     = Number(wf.guidance)
    if (wf.model_choice) p.model_choice = String(wf.model_choice)
    if (wf.device)       p.device       = String(wf.device)
    dispatch({ type: 'SET_PARAMS', params: p })
  }, [dispatch])

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-bg text-white overflow-hidden">
      {/* Top bar */}
      <TopBar
        status={state.status}
        onOpenSettings={() => dispatch({ type: 'TOGGLE_SETTINGS' })}
      />

      {/* Main area */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left sidebar (doubled width) */}
        <Sidebar
          params={state.params}
          models={state.models}
          availableModels={state.availableModels}
          devices={state.devices}
          workflows={state.workflows}
          isGenerating={state.isGenerating}
          hasIteratableMasks={state.refSlots.some(s => !!s.maskId)}
          hasRefImage={state.refSlots.length > 0 && !!state.refSlots[0]?.imageId}
          refImageSize={
            state.refSlots[0]?.w && state.refSlots[0]?.h
              ? { w: state.refSlots[0].w, h: state.refSlots[0].h }
              : undefined
          }
          onParamChange={(key, value) => {
            dispatch({ type: 'SET_PARAM', key, value })
            if (key === 'model_choice') {
              dispatch({ type: 'SET_PARAM', key: 'guidance', value: guidanceForModel(String(value)) })
              dispatch({ type: 'SET_PARAM', key: 'steps',    value: stepsForModel(String(value)) })
            }
          }}
          onParamsChange={(p) => dispatch({ type: 'SET_PARAMS', params: p })}
          onGenerate={handleGenerate}
          onStop={handleStop}
          onIterate={handleIterateGenerate}
          onWorkflowLoad={handleWorkflowLoad}
          onWorkflowRefresh={refreshWorkflows}
          onStatus={setStatusMsg}
        />

        {/* Center: 3 rows — drag-resizable */}
        <div ref={centerRef} className="flex flex-col flex-1 overflow-hidden">

          {/* Info / status bar (thin, shrink-0) */}
          {(state.resultInfo || statusMsg) && (
            <div className="shrink-0 px-4 py-1.5 bg-surface border-t border-border text-xs text-muted truncate">
              {state.resultInfo || statusMsg}
            </div>
          )}

          {/* Row 1: Result image */}
          <div className="overflow-hidden min-h-0" style={{ flex: `0 0 ${rowPcts[0]}%` }}>
            <Canvas
              resultUrl={state.resultUrl}
              isGenerating={state.isGenerating}
              progressMsg={state.progressMsg}
              progressStep={state.progressStep}
              progressTotal={state.progressTotal}
              error={state.error}
              onDropRef={file => handleAddRefSlots([file])}
            />
          </div>

          <RowDivider onDragStart={startRowDrag(0)} onKeyResize={d => keyResize(0, d)} label="Resize canvas and reference images rows" />

          {/* Row 2: Reference images + masks */}
          <div className="overflow-hidden min-h-0" style={{ flex: `0 0 ${rowPcts[1]}%` }}>
            <RefImagesRow
              slots={state.refSlots}
              maskMode={state.params.mask_mode}
              modelChoice={state.params.model_choice}
              onAddSlots={handleAddRefSlots}
              onAddSlotDirect={(imageId, imageUrl) => dispatch({ type: 'ADD_REF_SLOT', imageId, imageUrl })}
              onRemoveSlot={handleRemoveRefSlot}
              onUploadMask={handleUploadSlotMask}
              onClearMask={handleClearSlotMask}
              onSlotStrengthChange={handleSlotStrengthChange}
              onSlotDimsLoaded={(slotId, w, h) => dispatch({ type: 'SET_SLOT_DIMS', slotId, w, h })}
              onParamChange={(k, v) => dispatch({ type: 'SET_PARAM', key: k, value: v })}
            />
          </div>

          <RowDivider onDragStart={startRowDrag(1)} onKeyResize={d => keyResize(1, d)} label="Resize reference images and gallery rows" />

          {/* Row 3: Saved outputs gallery */}
          <div className="overflow-hidden min-h-0" style={{ flex: `0 0 ${rowPcts[2]}%` }}>
            <Gallery
              outputs={state.outputs}
              onSelect={handleSelectGallery}
              onDelete={handleDeleteOutput}
              upscaleModelPath={state.params.upscale_model_path}
              onUpscale={handleUpscaleGalleryItem}
              upscalingItem={upscalingGalleryUrl}
            />
          </div>

        </div>
      </div>

      {/* Settings drawer */}
      <SettingsDrawer
        open={state.settingsOpen}
        onClose={() => dispatch({ type: 'TOGGLE_SETTINGS' })}
      />
    </div>
  )
}
