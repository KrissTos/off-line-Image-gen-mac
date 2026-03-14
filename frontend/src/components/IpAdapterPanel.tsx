// src/components/IpAdapterPanel.tsx
import { useRef, useState, useCallback } from 'react'
import type { IpAdapterSlot, IpAdapterStatus } from '../types'
import type { Action } from '../store'
import { uploadImage, streamIpAdapterDownload, fetchIpAdapterStatus } from '../api'
import HelpTip from './HelpTip'

interface Props {
  slots:    IpAdapterSlot[]
  enabled:  boolean
  status:   IpAdapterStatus | null
  dispatch: (a: Action) => void
}

export default function IpAdapterPanel({ slots, enabled, status, dispatch }: Props) {
  const fileRef                           = useRef<HTMLInputElement>(null)
  const [downloading, setDownloading]     = useState(false)
  const [dlPct,       setDlPct]           = useState(0)
  const [dlMsg,       setDlMsg]           = useState('')
  const [dragOver,    setDragOver]        = useState<number | null>(null)

  const downloaded = status?.downloaded ?? false

  async function handleDownload() {
    setDownloading(true)
    setDlPct(0)
    setDlMsg('Starting download…')
    try {
      await streamIpAdapterDownload(ev => {
        if (ev.type === 'progress') { setDlPct(ev.pct ?? 0); setDlMsg(ev.message ?? '') }
        if (ev.type === 'done')     { setDlMsg('Download complete!') }
        if (ev.type === 'error')    { setDlMsg(`Error: ${ev.message}`) }
      })
      const s = await fetchIpAdapterStatus()
      dispatch({ type: 'SET_IPA_STATUS', status: s })
    } catch (e: unknown) {
      setDlMsg(`Error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setDownloading(false)
    }
  }

  const handleFile = useCallback(async (file: File) => {
    if (slots.length >= 3) return
    try {
      const { id, url } = await uploadImage(file)
      dispatch({ type: 'ADD_IPA_SLOT', imageId: id, imageUrl: url })
    } catch (e) {
      console.error('IPA upload failed', e)
    }
  }, [slots.length, dispatch])

  function onDragOver(e: React.DragEvent, slotId: number) {
    e.preventDefault()
    setDragOver(slotId)
  }
  function onDragLeave() { setDragOver(null) }

  async function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(null)
    const file = e.dataTransfer.files[0]
    if (file) { await handleFile(file); return }
    const url = e.dataTransfer.getData('text/plain')
    if (url) {
      try {
        const r    = await fetch(url)
        const blob = await r.blob()
        await handleFile(new File([blob], 'ref.png', { type: blob.type || 'image/png' }))
      } catch (e) { console.error('Drop failed', e) }
    }
  }

  return (
    <div className="space-y-3">

      {/* Not downloaded */}
      {!downloaded && !downloading && (
        <div className="space-y-2">
          <p className="text-xs text-[var(--color-muted)]">
            Weights not installed (~7 GB total including image encoder)
          </p>
          <button
            onClick={handleDownload}
            className="w-full py-2 rounded bg-[var(--color-accent)] text-white text-sm font-medium hover:opacity-90 transition-opacity"
          >
            Download IP-Adapter weights
          </button>
          <div className="flex items-center gap-1">
            <HelpTip text="IP-Adapter lets you guide generation using a reference photo — useful for matching a face, artistic style, color mood, or object appearance. Downloaded once, stored in ./models/ip_adapter/." />
            <span className="text-[11px] text-[var(--color-muted)]">What is IP-Adapter?</span>
          </div>
        </div>
      )}

      {/* Downloading */}
      {downloading && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-[var(--color-muted)]">
            <span className="truncate pr-2">{dlMsg}</span>
            <span className="shrink-0">{dlPct}%</span>
          </div>
          <div className="h-1 bg-[var(--color-border)] rounded-full overflow-hidden">
            <div className="h-full bg-[var(--color-accent)] transition-all duration-300" style={{ width: `${dlPct}%` }} />
          </div>
        </div>
      )}

      {/* Ready */}
      {downloaded && !downloading && (
        <div className="space-y-3">
          {/* Enable toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-[var(--color-label)]">Enable</span>
              <HelpTip text="When enabled, IP-Adapter injects visual features from your reference images into every generation. Disable to generate without it." />
            </div>
            <button
              onClick={() => dispatch({ type: 'TOGGLE_IPA' })}
              className={`relative w-10 h-5 rounded-full transition-colors ${enabled ? 'bg-[var(--color-accent)]' : 'bg-[var(--color-border)]'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${enabled ? 'translate-x-5' : 'translate-x-0'}`} />
            </button>
          </div>

          {/* Slots */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1">
              <span className="text-xs text-[var(--color-label)]">Reference images</span>
              <HelpTip text="Up to 3 reference images. Their visual features are blended together. Higher scale = stronger influence. 0.4–0.7 recommended." />
            </div>
            <div className="flex gap-2 flex-wrap">
              {slots.map(slot => (
                <div key={slot.slotId} className="flex flex-col items-center gap-1">
                  <div
                    className={`relative w-16 h-16 rounded border-2 overflow-hidden bg-[var(--color-surface)] ${dragOver === slot.slotId ? 'border-[var(--color-accent)]' : 'border-[var(--color-border)]'}`}
                    onDragOver={e => onDragOver(e, slot.slotId)}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                  >
                    <img src={slot.imageUrl} className="w-full h-full object-cover" alt="" />
                    <button
                      onClick={() => dispatch({ type: 'REMOVE_IPA_SLOT', slotId: slot.slotId })}
                      className="absolute top-0.5 right-0.5 w-4 h-4 rounded-full bg-black/60 text-white text-[10px] flex items-center justify-center hover:bg-black/90"
                    >×</button>
                  </div>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={slot.scale}
                    onChange={e => dispatch({ type: 'UPDATE_IPA_SCALE', slotId: slot.slotId, scale: parseFloat(e.target.value) })}
                    className="w-16 accent-[var(--color-accent)]"
                  />
                  <span className="text-[10px] text-[var(--color-muted)]">{slot.scale.toFixed(2)}</span>
                </div>
              ))}

              {slots.length < 3 && (
                <div
                  className={`w-16 h-16 rounded border-2 border-dashed flex items-center justify-center cursor-pointer transition-colors ${dragOver === -1 ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border)] hover:border-[var(--color-accent)]'}`}
                  onClick={() => fileRef.current?.click()}
                  onDragOver={e => onDragOver(e, -1)}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                >
                  <span className="text-[var(--color-muted)] text-2xl leading-none">+</span>
                </div>
              )}
            </div>
          </div>

          {slots.length > 0 && (
            <button
              onClick={() => dispatch({ type: 'CLEAR_IPA_SLOTS' })}
              className="text-xs text-[var(--color-muted)] hover:text-white transition-colors"
            >
              Clear all
            </button>
          )}
        </div>
      )}

      <input
        ref={fileRef} type="file" accept="image/*" className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = '' }}
      />
    </div>
  )
}
