import { useRef, useState } from 'react'
import { X, Plus, Pencil } from 'lucide-react'
import type { RefImageSlot, GenerateParams } from '../types'
import { uploadFromUrl } from '../api'
import HelpTip from './HelpTip'
import MaskEditor from './MaskEditor'

// ── SlotCard ──────────────────────────────────────────────────────────────────

interface SlotCardProps {
  slot:             RefImageSlot
  isBase:           boolean           // true for slot #1
  thumbSize:        number
  onRemove:         () => void
  maskIgnored:      boolean           // slot #2+ mask outside Inpainting Pipeline mode → not sent
  onClearMask:      () => void
  onDrawMask:       () => void
  onStrengthChange: (v: number) => void
  onDimsLoaded?:    (w: number, h: number) => void
}

const EXTRA_MASK_HINT = 'Masks on slot #2+ are only used by Iterate Masks (mask mode "Inpainting Pipeline"). Normal Generate uses only the base image mask.'

function SlotCard({ slot, isBase, thumbSize, onRemove, maskIgnored, onClearMask, onDrawMask, onStrengthChange, onDimsLoaded }: SlotCardProps) {
  const maskSize = Math.round(thumbSize * 0.7)

  const maskBox = (
    <div
      className="relative rounded-lg overflow-hidden border border-dashed border-border/60 group"
      style={{ width: maskSize, height: maskSize }}
    >
      {slot.maskUrl ? (
        <>
          <button onClick={onDrawMask} aria-label="Edit mask" className="w-full h-full">
            <img src={slot.maskUrl} alt="mask" className="w-full h-full object-cover hover:opacity-80 transition-opacity" />
          </button>
          <div className={`absolute inset-x-0 top-0 text-[8px] text-center bg-black/50 py-0.5 pointer-events-none
                           ${maskIgnored ? 'text-amber-300' : 'text-muted'}`}>
            {maskIgnored ? 'unused' : 'mask'}
          </div>
          <button
            onClick={onClearMask}
            aria-label="Remove mask"
            className="absolute top-1 right-1 bg-black/70 hover:bg-red-600 rounded-full p-0.5
                       opacity-0 group-hover:opacity-100 transition-all"
          >
            <X size={8} aria-hidden="true" />
          </button>
        </>
      ) : (
        <button
          onClick={onDrawMask}
          aria-label="Draw mask"
          className="flex flex-col items-center justify-center gap-0.5 w-full h-full
                     text-muted hover:text-white transition-colors"
        >
          <Pencil size={12} aria-hidden="true" />
          <span className="text-[8px] leading-none">draw mask</span>
        </button>
      )}
    </div>
  )

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
        </div>

        {/* Mask thumbnail (click to edit) or draw target */}
        {isBase ? maskBox : <HelpTip text={EXTRA_MASK_HINT} position="bottom">{maskBox}</HelpTip>}
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

          {/* Slot cards */}
          {slots.map(slot => (
            <SlotCard
              key={slot.slotId}
              slot={slot}
              isBase={slot.slotId === 1}
              thumbSize={thumbSize}
              onRemove={() => onRemoveSlot(slot.slotId)}
              maskIgnored={slot.slotId !== 1 && maskMode !== 'Inpainting Pipeline (Quality)'}
              onClearMask={() => onClearMask(slot.slotId)}
              onDrawMask={() => setMaskEditorSlot(slot)}
              onStrengthChange={v => onSlotStrengthChange(slot.slotId, v)}
              onDimsLoaded={(w, h) => onSlotDimsLoaded?.(slot.slotId, w, h)}
            />
          ))}

          {/* Add ref button — after the last slot; also a drop zone for gallery drag */}
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
      {maskEditorSlot && slots[0] && (
        <MaskEditor
          slot={maskEditorSlot}
          baseImageId={slots[0].imageId}
          baseImageUrl={slots[0].imageUrl}
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
