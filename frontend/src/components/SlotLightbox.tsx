import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, ChevronLeft, ChevronRight, ImagePlus } from 'lucide-react'
import type { RefImageSlot } from '../types'
import { neighborSlot } from '../slots'
import { keyAction } from '../previewModel'

interface Props {
  slots:      RefImageSlot[]                 // display order: base first
  slotId:     number
  onClose:    () => void
  onNavigate: (slotId: number) => void
  onReplace:  (slotId: number) => void       // opens the file picker for this slot
}

/** Full-size view of a base / reference image. Arrows step through the slots. */
export default function SlotLightbox({ slots, slotId, onClose, onNavigate, onReplace }: Props) {
  const slot = slots.find(s => s.slotId === slotId)
  const prev = neighborSlot(slots, slotId, -1)
  const next = neighborSlot(slots, slotId, 1)
  const [measured, setMeasured] = useState<{ id: number; w: number; h: number } | null>(null)
  const dims = measured?.id === slotId ? measured : null   // measured on load; stale after navigating

  // The slot can disappear while open (removed elsewhere): close instead of rendering nothing.
  useEffect(() => { if (!slot) onClose() }, [slot, onClose])

  // Remember the opener to hand focus back on close; Tab cycles inside the dialog.
  const dialogRef = useRef<HTMLDivElement>(null)
  const [opener] = useState(() => document.activeElement as HTMLElement | null)
  useEffect(() => () => { if (opener?.isConnected) opener.focus() }, [opener])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null
      const act = keyAction({
        key: e.key, metaKey: e.metaKey, ctrlKey: e.ctrlKey, altKey: e.altKey,
        targetTag: t?.tagName ?? '', targetEditable: !!t?.isContentEditable,
      })
      if (act === 'close') onClose()
      else if (act === 'prev' && prev) onNavigate(prev.slotId)
      else if (act === 'next' && next) onNavigate(next.slotId)
      else if (e.key === 'Tab' && dialogRef.current) {
        const els = [...dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled])')]
        if (!els.length) return
        const first = els[0], last = els[els.length - 1]
        if (!dialogRef.current.contains(document.activeElement)) { e.preventDefault(); first.focus() }
        else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onNavigate, prev, next])

  // Close on backdrop only when press AND release land on it.
  const downOnBackdrop = useRef(false)

  if (!slot) return null
  const title = slot.slotId === 1 ? 'img 1 · base' : `img ${slot.slotId}`

  return createPortal(
    <div
      className="fixed inset-0 z-50 bg-black/85 flex items-center justify-center p-6"
      onMouseDown={e => { downOnBackdrop.current = e.target === e.currentTarget }}
      onClick={e => { if (e.target === e.currentTarget && downOnBackdrop.current) onClose() }}
    >
      <div
        ref={dialogRef}
        role="dialog" aria-modal="true" aria-label={`${title} preview`}
        className="relative flex flex-col max-w-[92vw] max-h-[92vh] bg-surface border border-border rounded-xl overflow-hidden"
      >
        <div className="flex items-center gap-3 px-3 py-2 border-b border-border">
          <span className="text-xs font-medium text-white">{title}</span>
          {(dims || (slot.w && slot.h)) && (
            <span className="text-xs text-muted">{dims?.w ?? slot.w} × {dims?.h ?? slot.h}</span>
          )}
          <div className="ml-auto flex items-center gap-1">
            <button onClick={() => onReplace(slot.slotId)}
              className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-muted hover:text-white hover:bg-card transition-colors">
              <ImagePlus size={14} aria-hidden="true" /> Replace image
            </button>
            <button onClick={onClose} autoFocus aria-label="Close preview"
              className="p-1 rounded text-muted hover:text-white transition-colors">
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </div>

        <div className="relative flex-1 min-h-0 bg-bg flex items-center justify-center">
          <img
            src={slot.imageUrl} alt={title}
            onLoad={e => setMeasured({ id: slotId, w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            className="max-w-full max-h-[80vh] object-contain"
          />
          {prev && (
            <button onClick={() => onNavigate(prev.slotId)} aria-label="Previous image"
              className="absolute left-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/60 hover:bg-accent transition-colors">
              <ChevronLeft size={18} aria-hidden="true" />
            </button>
          )}
          {next && (
            <button onClick={() => onNavigate(next.slotId)} aria-label="Next image"
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/60 hover:bg-accent transition-colors">
              <ChevronRight size={18} aria-hidden="true" />
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
