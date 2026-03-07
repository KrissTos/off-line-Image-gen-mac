import { useRef } from 'react'
import { UploadCloud, X, Plus } from 'lucide-react'
import type { RefImageSlot, GenerateParams } from '../types'

// ── SlotCard ──────────────────────────────────────────────────────────────────

interface SlotCardProps {
  slot:         RefImageSlot
  onRemove:     () => void
  onUploadMask: (f: File) => void
  onClearMask:  () => void
}

function SlotCard({ slot, onRemove, onUploadMask, onClearMask }: SlotCardProps) {
  const maskRef = useRef<HTMLInputElement>(null)

  return (
    <div className="shrink-0 flex items-end gap-1.5">
      {/* Reference image */}
      <div className="relative w-20 h-20 rounded-lg overflow-hidden border border-border group">
        <img src={slot.imageUrl} alt={`ref #${slot.slotId}`} className="w-full h-full object-cover" />

        {/* Slot number badge */}
        <div className="absolute top-1 left-1 bg-accent text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full shadow">
          #{slot.slotId}
        </div>

        {/* Remove button */}
        <button
          onClick={onRemove}
          title="Remove reference image"
          className="absolute top-1 right-1 bg-black/70 hover:bg-red-600 rounded-full p-0.5
                     opacity-0 group-hover:opacity-100 transition-all"
        >
          <X size={10} />
        </button>
      </div>

      {/* Mask thumbnail or upload target */}
      <div className="relative w-14 h-14 rounded-lg overflow-hidden border border-dashed border-border/60 group"
           title={slot.maskUrl ? 'Mask loaded — hover to clear' : 'Upload mask for this ref image'}>
        {slot.maskUrl ? (
          <>
            <img src={slot.maskUrl} alt="mask" className="w-full h-full object-cover" />
            <div className="absolute inset-x-0 top-0 text-[8px] text-center bg-black/50 text-muted py-0.5">
              mask
            </div>
            <button
              onClick={onClearMask}
              title="Remove mask"
              className="absolute top-1 right-1 bg-black/70 hover:bg-red-600 rounded-full p-0.5
                         opacity-0 group-hover:opacity-100 transition-all"
            >
              <X size={8} />
            </button>
          </>
        ) : (
          <button
            onClick={() => maskRef.current?.click()}
            className="w-full h-full flex flex-col items-center justify-center gap-0.5
                       text-muted hover:text-white hover:border-accent transition-colors"
          >
            <UploadCloud size={12} />
            <span className="text-[8px] leading-none">mask</span>
          </button>
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
  )
}

// ── RefImagesRow ──────────────────────────────────────────────────────────────

interface Props {
  slots:        RefImageSlot[]
  imgStrength:  number
  maskMode:     string
  onAddSlots:   (files: File[]) => void
  onRemoveSlot: (slotId: number) => void
  onUploadMask: (slotId: number, file: File) => void
  onClearMask:  (slotId: number) => void
  onParamChange:(k: keyof GenerateParams, v: unknown) => void
}

export default function RefImagesRow({
  slots, imgStrength, maskMode,
  onAddSlots, onRemoveSlot, onUploadMask, onClearMask, onParamChange,
}: Props) {
  const addRef = useRef<HTMLInputElement>(null)

  return (
    <div className="shrink-0 border-t border-border bg-surface px-4 py-2 min-h-[88px]">
      <div className="flex items-center gap-3 overflow-x-auto pb-1">

        {/* Add ref button */}
        <button
          onClick={() => addRef.current?.click()}
          title="Add reference image"
          className="shrink-0 flex flex-col items-center justify-center w-16 h-16 rounded-lg
                     border border-dashed border-border text-muted
                     hover:border-accent hover:text-white transition-colors gap-1"
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
            onRemove={() => onRemoveSlot(slot.slotId)}
            onUploadMask={f => onUploadMask(slot.slotId, f)}
            onClearMask={() => onClearMask(slot.slotId)}
          />
        ))}

        {/* Strength + mask-mode controls (only when refs exist) */}
        {slots.length > 0 && (
          <div className="shrink-0 flex flex-col gap-2 pl-2 border-l border-border ml-1 min-w-[130px]">
            {/* Strength */}
            <div>
              <div className="flex justify-between mb-0.5">
                <span className="text-[10px] text-muted">Strength</span>
                <span className="text-[10px] text-white">{imgStrength.toFixed(2)}</span>
              </div>
              <input
                type="range" min={0} max={1} step={0.05} value={imgStrength}
                onChange={e => onParamChange('img_strength', Number(e.target.value))}
                className="w-full h-1 accent-accent appearance-none bg-border rounded-full"
              />
            </div>

            {/* Mask mode (only when any slot has a mask) */}
            {slots.some(s => s.maskUrl) && (
              <div>
                <span className="text-[10px] text-muted block mb-0.5">Mask mode</span>
                <select
                  value={maskMode}
                  onChange={e => onParamChange('mask_mode', e.target.value)}
                  className="w-full bg-card border border-border rounded px-1.5 py-0.5 text-[10px] text-white
                             focus:outline-none focus:border-accent"
                >
                  <option>Crop & Composite (Fast)</option>
                  <option>Inpainting Pipeline (Quality)</option>
                </select>
              </div>
            )}
          </div>
        )}

        {/* Empty state hint */}
        {slots.length === 0 && (
          <span className="text-[10px] text-muted/50 select-none">
            Add reference images for img2img / inpainting — labeled #1, #2 … for use in workflows
          </span>
        )}
      </div>
    </div>
  )
}
