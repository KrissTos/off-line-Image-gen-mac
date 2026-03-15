import { useRef, useState, useEffect, useCallback } from 'react'
import { UploadCloud, X, Plus, Pencil } from 'lucide-react'
import type { RefImageSlot, GenerateParams } from '../types'
import { uploadFromUrl } from '../api'
import HelpTip from './HelpTip'

// ── MaskEditorModal ────────────────────────────────────────────────────────────

interface MaskEditorProps {
  slot:         RefImageSlot
  baseImageUrl: string        // ALWAYS slot #1's URL — the image being edited
  onClose:      () => void
  onApply:      (maskFile: File) => void
}

function MaskEditorModal({ slot, baseImageUrl, onClose, onApply }: MaskEditorProps) {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const imgRef     = useRef<HTMLImageElement>(null)

  const [naturalSize, setNaturalSize]   = useState({ w: 0, h: 0 })
  const [displaySize, setDisplaySize]   = useState({ w: 0, h: 0 })
  const [rect, setRect]                 = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const draggingRef                     = useRef(false)
  const startPtRef                      = useRef({ x: 0, y: 0 })
  const [applying, setApplying]         = useState(false)

  // Calculate display size keeping aspect ratio, max 680×520
  function calcDisplay(nw: number, nh: number) {
    const maxW = 680, maxH = 520
    let w = nw, h = nh
    if (w > maxW) { h = Math.round(h * maxW / w); w = maxW }
    if (h > maxH) { w = Math.round(w * maxH / h); h = maxH }
    return { w, h }
  }

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const r = canvasRef.current!.getBoundingClientRect()
    const pos = { x: e.clientX - r.left, y: e.clientY - r.top }
    draggingRef.current = true
    startPtRef.current  = pos
    setRect({ x: pos.x, y: pos.y, w: 0, h: 0 })
  }

  // Window-level drag tracking — keeps drag alive even when cursor leaves canvas edges.
  // Coordinates are clamped to canvas bounds so the selection snaps to the image edge.
  useEffect(() => {
    function handleMouseMove(e: MouseEvent) {
      if (!draggingRef.current || !canvasRef.current) return
      const r  = canvasRef.current.getBoundingClientRect()
      const x  = Math.max(0, Math.min(e.clientX - r.left, r.width))
      const y  = Math.max(0, Math.min(e.clientY - r.top,  r.height))
      const sp = startPtRef.current
      setRect({ x: Math.min(sp.x, x), y: Math.min(sp.y, y), w: Math.abs(x - sp.x), h: Math.abs(y - sp.y) })
    }
    function handleMouseUp() { draggingRef.current = false }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup',   handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup',   handleMouseUp)
    }
  }, [])

  // Re-draw overlay whenever rect or displaySize changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || displaySize.w === 0) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (rect && rect.w > 2 && rect.h > 2) {
      // Dark veil over entire canvas
      ctx.fillStyle = 'rgba(0,0,0,0.55)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      // Punch out the selected area (shows the underlying image)
      ctx.clearRect(rect.x, rect.y, rect.w, rect.h)
      // Accent border around selection
      ctx.strokeStyle = '#7c3aed'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 4])
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h)
      // Tiny size label
      const label = `${Math.round(rect.w * (naturalSize.w / displaySize.w))} × ${Math.round(rect.h * (naturalSize.h / displaySize.h))}`
      ctx.setLineDash([])
      ctx.font = '11px monospace'
      ctx.fillStyle = '#fff'
      ctx.fillText(label, rect.x + 4, Math.max(rect.y - 4, 14))
    }
  }, [rect, displaySize, naturalSize])

  const handleApply = useCallback(() => {
    if (!rect || rect.w < 2 || rect.h < 2 || naturalSize.w === 0) return
    setApplying(true)

    const scaleX = naturalSize.w / displaySize.w
    const scaleY = naturalSize.h / displaySize.h
    const mx = Math.round(rect.x * scaleX)
    const my = Math.round(rect.y * scaleY)
    const mw = Math.round(rect.w * scaleX)
    const mh = Math.round(rect.h * scaleY)

    const mc = document.createElement('canvas')
    mc.width  = naturalSize.w
    mc.height = naturalSize.h
    const mctx = mc.getContext('2d')!
    // Black background = unmasked area
    mctx.fillStyle = 'black'
    mctx.fillRect(0, 0, mc.width, mc.height)
    // White rectangle = area to inpaint
    mctx.fillStyle = 'white'
    mctx.fillRect(mx, my, mw, mh)

    mc.toBlob(blob => {
      if (blob) onApply(new File([blob], 'rect_mask.png', { type: 'image/png' }))
      setApplying(false)
    }, 'image/png')
  }, [rect, naturalSize, displaySize, onApply])

  // Close on Escape / apply on Enter
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
      if (e.key === 'Enter') handleApply()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose, handleApply])

  const dialogRef = useRef<HTMLDivElement>(null)

  // Move focus into dialog on open
  useEffect(() => {
    dialogRef.current?.focus()
  }, [])

  const hasRect = rect && rect.w > 2 && rect.h > 2
  const isBase  = slot.slotId === 1
  const titleText = isBase
    ? 'Draw Mask — Base Image (#1)'
    : `Draw Mask — Reference #${slot.slotId} (on base image)`

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="mask-editor-title"
        tabIndex={-1}
        className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3 shadow-2xl max-w-3xl w-full mx-4 focus:outline-none"
      >

        {/* Header */}
        <div className="flex items-center justify-between">
          <span id="mask-editor-title" className="text-sm font-semibold text-white flex items-center gap-2">
            <Pencil size={14} className="text-accent" aria-hidden="true" />
            {titleText}
          </span>
          <button onClick={onClose} aria-label="Close mask editor" className="text-muted hover:text-white transition-colors">
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        {!isBase && (
          <p className="text-[11px] text-amber-400/80 bg-amber-900/20 border border-amber-800/30 rounded px-2 py-1.5">
            You are drawing on the <strong>base image (#1)</strong>. This mask defines the region that will
            be replaced in pass #{slot.slotId}, using reference #{slot.slotId} as the style guide.
          </p>
        )}

        <p className="text-[11px] text-muted">
          Click and drag to select the region to inpaint.
          The white area will be regenerated; everything else is preserved.
        </p>

        {/* Canvas area */}
        <div className="bg-bg rounded-lg overflow-hidden flex items-center justify-center"
             style={{ minHeight: 200 }}>

          {/*
            Inner wrapper sized exactly to the image — both <img> and <canvas> live here
            so `absolute top:0 left:0` on the canvas always lands precisely over the image,
            regardless of how wide the outer flex container is.
          */}
          <div className="relative" style={{ width: displaySize.w || 'auto', height: displaySize.h || 'auto' }}>

            {/* Base image (always slot #1) */}
            <img
              ref={imgRef}
              src={baseImageUrl}
              alt="base image"
              draggable={false}
              style={{
                display: 'block',
                width:  displaySize.w || 'auto',
                height: displaySize.h || 'auto',
                userSelect: 'none',
              }}
              onLoad={e => {
                const img = e.currentTarget
                const ns = { w: img.naturalWidth, h: img.naturalHeight }
                const ds = calcDisplay(ns.w, ns.h)
                setNaturalSize(ns)
                setDisplaySize(ds)
              }}
            />

            {/* Drawing overlay — perfectly aligned over the image */}
            {displaySize.w > 0 && (
              <canvas
                ref={canvasRef}
                width={displaySize.w}
                height={displaySize.h}
                className="absolute cursor-crosshair"
                style={{ top: 0, left: 0, width: displaySize.w, height: displaySize.h }}
                onMouseDown={onMouseDown}
              />
            )}

            {/* Hint overlay before first drag */}
            {displaySize.w > 0 && !rect && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <span className="bg-black/60 text-muted text-xs px-3 py-1.5 rounded-full">
                  Click and drag to select region
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 justify-between">
          <span className="text-[10px] text-muted">
            {hasRect
              ? `Selected: ${Math.round(rect!.w * (naturalSize.w / displaySize.w))} × ${Math.round(rect!.h * (naturalSize.h / displaySize.h))} px`
              : 'No region selected'}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setRect(null)}
              className="px-3 py-1.5 rounded-md bg-card border border-border text-xs text-muted
                         hover:text-white transition-colors"
            >
              Clear
            </button>
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded-md bg-card border border-border text-xs text-muted
                         hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleApply}
              disabled={!hasRect || applying}
              className="px-4 py-1.5 rounded-md bg-accent text-white text-xs font-medium
                         disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/80 transition-colors"
            >
              {applying ? 'Applying…' : 'Apply Mask'}
            </button>
          </div>
        </div>

      </div>
    </div>
  )
}

// ── SlotCard ──────────────────────────────────────────────────────────────────

interface SlotCardProps {
  slot:             RefImageSlot
  isBase:           boolean           // true for slot #1
  thumbSize:        number
  onRemove:         () => void
  onUploadMask:     (f: File) => void
  onClearMask:      () => void
  onDrawMask:       () => void
  onStrengthChange: (v: number) => void
  onDimsLoaded?:    (w: number, h: number) => void
}

function SlotCard({ slot, isBase, thumbSize, onRemove, onUploadMask, onClearMask, onDrawMask, onStrengthChange, onDimsLoaded }: SlotCardProps) {
  const maskRef  = useRef<HTMLInputElement>(null)
  const maskSize = Math.round(thumbSize * 0.7)

  return (
    <div className="shrink-0 flex flex-col gap-1">
      <div className="flex items-end gap-1.5">

        {/* Reference image */}
        <div
          className="relative rounded-lg overflow-hidden border border-border group"
          style={{ width: thumbSize, height: thumbSize }}
        >
          <img
            src={slot.imageUrl}
            alt={`ref #${slot.slotId}`}
            className="w-full h-full object-cover"
            onLoad={e => {
              const img = e.currentTarget
              onDimsLoaded?.(img.naturalWidth, img.naturalHeight)
            }}
          />

          {/* Slot role badge */}
          <div className={`absolute top-1 left-1 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full shadow
                           ${isBase ? 'bg-teal-600' : 'bg-accent'}`}>
            {isBase ? 'base' : `ref ${slot.slotId - 1}`}
          </div>

          {/* Remove button */}
          <button
            onClick={onRemove}
            title="Remove reference image"
            aria-label={isBase ? 'Remove base image' : `Remove reference image ${slot.slotId - 1}`}
            className="absolute top-1 right-1 bg-black/70 hover:bg-red-600 rounded-full p-0.5
                       opacity-0 group-hover:opacity-100 transition-all"
          >
            <X size={10} aria-hidden="true" />
          </button>

          {/* Draw mask button */}
          <button
            onClick={onDrawMask}
            title="Draw mask by selecting a rectangle"
            aria-label="Draw mask rectangle"
            className="absolute bottom-1 right-1 bg-black/70 hover:bg-accent rounded-full p-0.5
                       opacity-0 group-hover:opacity-100 transition-all"
          >
            <Pencil size={9} aria-hidden="true" />
          </button>
        </div>

        {/* Mask thumbnail or upload target */}
        <div
          className="relative rounded-lg overflow-hidden border border-dashed border-border/60 group"
          style={{ width: maskSize, height: maskSize }}
          title={slot.maskUrl ? 'Mask loaded — hover to clear' : 'Upload mask or draw rectangle on image'}
        >
          {slot.maskUrl ? (
            <>
              <img src={slot.maskUrl} alt="mask" className="w-full h-full object-cover" />
              <div className="absolute inset-x-0 top-0 text-[8px] text-center bg-black/50 text-muted py-0.5">
                mask
              </div>
              <button
                onClick={onClearMask}
                title="Remove mask"
                aria-label="Remove mask"
                className="absolute top-1 right-1 bg-black/70 hover:bg-red-600 rounded-full p-0.5
                           opacity-0 group-hover:opacity-100 transition-all"
              >
                <X size={8} aria-hidden="true" />
              </button>
            </>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center gap-0.5">
              <button
                onClick={() => maskRef.current?.click()}
                title="Upload mask file"
                className="flex flex-col items-center justify-center gap-0.5 w-full h-full
                           text-muted hover:text-white transition-colors"
              >
                <UploadCloud size={12} />
                <span className="text-[8px] leading-none">mask</span>
              </button>
            </div>
          )}
          <input
            ref={maskRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={e => {
              if (e.target.files?.[0]) onUploadMask(e.target.files[0])
              e.target.value = ''
            }}
          />
        </div>
      </div>

      {/* Per-slot strength slider */}
      <div style={{ width: thumbSize + maskSize + 6 }}>
        <div className="flex justify-between mb-0.5">
          <label htmlFor={`slot-strength-${slot.slotId}`} className="text-[9px] text-muted flex items-center gap-0.5">
            strength
            <HelpTip text="How strongly this reference image blends into the output. Slot #1 is the base image — lower values preserve more of the original." />
          </label>
          <span className="text-[9px] text-white" aria-hidden="true">{slot.strength.toFixed(2)}</span>
        </div>
        <input
          id={`slot-strength-${slot.slotId}`}
          type="range" min={0} max={1} step={0.05} value={slot.strength}
          aria-label={`Slot ${slot.slotId} inpaint strength: ${slot.strength.toFixed(2)}`}
          onChange={e => onStrengthChange(Number(e.target.value))}
          className="w-full h-1 accent-accent appearance-none bg-border rounded-full"
        />
      </div>
    </div>
  )
}

// ── RefImagesRow ──────────────────────────────────────────────────────────────

interface Props {
  slots:                RefImageSlot[]
  maskMode:             string
  modelChoice:          string
  onAddSlots:           (files: File[]) => void
  onAddSlotDirect?:     (imageId: string, imageUrl: string) => void
  onRemoveSlot:         (slotId: number) => void
  onUploadMask:         (slotId: number, file: File) => void
  onClearMask:          (slotId: number) => void
  onSlotStrengthChange: (slotId: number, strength: number) => void
  onSlotDimsLoaded?:    (slotId: number, w: number, h: number) => void
  onParamChange:        (k: keyof GenerateParams, v: unknown) => void
}

export default function RefImagesRow({
  slots, maskMode, modelChoice,
  onAddSlots, onAddSlotDirect, onRemoveSlot, onUploadMask, onClearMask,
  onSlotStrengthChange, onSlotDimsLoaded, onParamChange,
}: Props) {
  const addRef = useRef<HTMLInputElement>(null)
  const [maskEditorSlot, setMaskEditorSlot] = useState<RefImageSlot | null>(null)
  const [thumbSize, setThumbSize] = useState(80)
  const [dragOverNew, setDragOverNew] = useState(false)

  async function handleRefDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOverNew(false)

    // File drop (from OS)
    const file = e.dataTransfer.files[0]
    if (file) {
      onAddSlots([file])
      return
    }

    // Gallery drag (URL string)
    const srcUrl = e.dataTransfer.getData('text/plain')
    if (srcUrl && onAddSlotDirect) {
      try {
        const { id, url } = await uploadFromUrl(srcUrl)
        onAddSlotDirect(id, url)
      } catch (err) {
        console.error('Drop upload failed', err)
      }
    }
  }

  // The base image is always slot #1 — all masks are drawn on it
  const baseImageUrl = slots[0]?.imageUrl ?? ''

  return (
    <>
      <div className="h-full border-t border-border bg-surface px-4 py-2 overflow-y-auto relative">

        {/* Thumbnail size slider — pinned top-right */}
        <div className="absolute top-2 right-3 flex items-center gap-1.5 z-10">
          <span className="text-[9px] text-muted select-none">size</span>
          <input
            type="range" min={48} max={160} step={8} value={thumbSize}
            aria-label="Reference image thumbnail size"
            onChange={e => setThumbSize(Number(e.target.value))}
            className="w-20 h-1 accent-accent appearance-none bg-border rounded-full"
          />
        </div>

        <div className="flex items-start gap-3 overflow-x-auto pb-1">

          {/* Add ref button — also a drop zone for gallery drag */}
          <button
            onClick={() => addRef.current?.click()}
            title="Add reference image (or drop from gallery)"
            style={{ width: thumbSize, height: thumbSize }}
            onDragOver={e => { e.preventDefault(); setDragOverNew(true) }}
            onDragLeave={() => setDragOverNew(false)}
            onDrop={handleRefDrop}
            className={`shrink-0 flex flex-col items-center justify-center rounded-lg
                       border border-dashed text-muted mt-0
                       hover:border-accent hover:text-white transition-colors gap-1
                       ${dragOverNew
                         ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-white'
                         : 'border-border'}`}
          >
            <Plus size={16} />
            <span className="text-[9px] leading-none">ref img</span>
          </button>
          <input
            ref={addRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={e => {
              if (e.target.files) onAddSlots(Array.from(e.target.files))
              e.target.value = ''
            }}
          />

          {/* Slot cards */}
          {slots.map(slot => (
            <SlotCard
              key={slot.slotId}
              slot={slot}
              isBase={slot.slotId === 1}
              thumbSize={thumbSize}
              onRemove={() => onRemoveSlot(slot.slotId)}
              onUploadMask={f => onUploadMask(slot.slotId, f)}
              onClearMask={() => onClearMask(slot.slotId)}
              onDrawMask={() => setMaskEditorSlot(slot)}
              onStrengthChange={v => onSlotStrengthChange(slot.slotId, v)}
              onDimsLoaded={(w, h) => onSlotDimsLoaded?.(slot.slotId, w, h)}
            />
          ))}

          {/* Mask-mode dropdown (only when any slot has a mask) */}
          {slots.some(s => s.maskUrl) && (
            <div className="shrink-0 flex flex-col gap-1 pl-2 border-l border-border ml-1 min-w-[140px] pt-0.5">
              <span className="text-[10px] text-muted flex items-center gap-1 mb-0.5">
                Mask mode
                <HelpTip text="Controls how the drawn mask is applied during inpainting. Use Iterate Masks mode for applying different masks from different reference slots." />
              </span>
              <select
                value={maskMode}
                onChange={e => onParamChange('mask_mode', e.target.value)}
                className="w-full bg-card border border-border rounded px-1.5 py-0.5 text-[10px] text-white
                           focus:outline-none focus:border-accent"
              >
                <option>Crop & Composite (Fast)</option>
                <option>Inpainting Pipeline (Quality)</option>
              </select>
              {modelChoice.startsWith('FLUX') && maskMode === 'Inpainting Pipeline (Quality)' && (
                <p className="text-[9px] text-amber-400/80 bg-amber-900/20 border border-amber-800/30
                              rounded px-1.5 py-1 leading-tight mt-1">
                  ⓘ FLUX.2-klein doesn't support inpainting — will use img2img instead
                </p>
              )}
            </div>
          )}

          {/* Empty state hint */}
          {slots.length === 0 && (
            <span className="text-[10px] text-muted/50 select-none self-center">
              Add reference images for img2img / inpainting — #1 = base image, #2+ = style references
            </span>
          )}
        </div>
      </div>

      {/* Mask editor modal — rendered outside the scrollable row */}
      {maskEditorSlot && (
        <MaskEditorModal
          slot={maskEditorSlot}
          baseImageUrl={baseImageUrl}
          onClose={() => setMaskEditorSlot(null)}
          onApply={file => {
            onUploadMask(maskEditorSlot.slotId, file)
            setMaskEditorSlot(null)
          }}
        />
      )}
    </>
  )
}
