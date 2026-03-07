import { useEffect, useRef, useCallback, useState } from 'react'
import { useAppState } from './store'
import type { GenerateParams, SSEEvent, OutputItem } from './types'
import {
  fetchStatus, fetchModels, fetchDevices, fetchWorkflows,
  fetchOutputs, uploadImage, streamGenerate,
} from './api'
import TopBar        from './components/TopBar'
import Sidebar       from './components/Sidebar'
import Canvas        from './components/Canvas'
import RefImagesRow  from './components/RefImagesRow'
import Gallery       from './components/Gallery'
import SettingsDrawer from './components/SettingsDrawer'

export default function App() {
  const { state, dispatch } = useAppState()
  const abortRef = useRef<AbortController | null>(null)
  const [statusMsg, setStatusMsg] = useState('')

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
      if (data.current) {
        dispatch({ type: 'SET_PARAM', key: 'model_choice', value: data.current })
      } else if (data.choices.length > 0) {
        dispatch({ type: 'SET_PARAM', key: 'model_choice', value: data.choices[0] })
      }
    }).catch(() => {})

    fetchDevices().then(({ devices }) => {
      dispatch({ type: 'SET_DEVICES', devices })
      if (devices.length > 0) {
        dispatch({ type: 'SET_PARAM', key: 'device', value: devices[0] })
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
          onParamChange={(key, value) => dispatch({ type: 'SET_PARAM', key, value })}
          onParamsChange={(p) => dispatch({ type: 'SET_PARAMS', params: p })}
          onGenerate={handleGenerate}
          onStop={handleStop}
          onWorkflowLoad={handleWorkflowLoad}
          onWorkflowRefresh={refreshWorkflows}
          onStatus={setStatusMsg}
        />

        {/* Center: 3 rows */}
        <div className="flex flex-col flex-1 overflow-hidden">

          {/* Row 1: Result image (takes all free space) */}
          <div className="flex-1 overflow-hidden min-h-0">
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

          {/* Info / status bar */}
          {(state.resultInfo || statusMsg) && (
            <div className="shrink-0 px-4 py-1.5 bg-surface border-t border-border text-xs text-muted truncate">
              {state.resultInfo || statusMsg}
            </div>
          )}

          {/* Row 2: Reference images + masks (#1, #2, …) */}
          <RefImagesRow
            slots={state.refSlots}
            imgStrength={state.params.img_strength}
            maskMode={state.params.mask_mode}
            onAddSlots={handleAddRefSlots}
            onRemoveSlot={handleRemoveRefSlot}
            onUploadMask={handleUploadSlotMask}
            onClearMask={handleClearSlotMask}
            onParamChange={(k, v) => dispatch({ type: 'SET_PARAM', key: k, value: v })}
          />

          {/* Row 3: Saved outputs gallery */}
          <Gallery
            outputs={state.outputs}
            onSelect={handleSelectGallery}
          />

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
