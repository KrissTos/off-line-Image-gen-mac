import { useRef, useState } from 'react'
import { Film, Trash2, Info, ArrowUpCircle, Loader2, RotateCcw } from 'lucide-react'
import type { OutputItem } from '../types'

interface Props {
  outputs:          OutputItem[]
  onSelect:         (item: OutputItem) => void
  onDelete:         (filename: string) => void
  onLoadParams?:    (item: OutputItem) => void
  upscaleModelPath?: string
  onUpscale?:        (item: OutputItem) => void
  upscalingItem?:    string | null   // url of item currently being upscaled
}

export default function Gallery({ outputs, onSelect, onDelete, onLoadParams, upscaleModelPath, onUpscale, upscalingItem }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [imgDims, setImgDims] = useState<Record<string, { w: number; h: number }>>({})
  const [showInfo, setShowInfo] = useState<string | null>(null)   // url of item with visible info tooltip

  function handleWheel(e: React.WheelEvent) {
    if (!scrollRef.current) return
    e.preventDefault()
    scrollRef.current.scrollLeft += e.deltaY + e.deltaX
  }

  if (outputs.length === 0) return null

  return (
    <div
      ref={scrollRef}
      onWheel={handleWheel}
      className="h-full border-t border-border bg-surface px-4 pt-2 overflow-x-auto overflow-y-hidden"
      style={{ paddingBottom: 'max(8px, env(safe-area-inset-bottom))' }}
    >
      <div className="flex gap-2 h-full items-center">
        {outputs.map((item) => (
          <div
            key={item.url}
            draggable={item.kind !== 'video'}
            onDragStart={e => {
              e.dataTransfer.setData('text/plain', item.url)
              e.dataTransfer.effectAllowed = 'copy'
            }}
            onClick={() => { setShowInfo(null); onSelect(item) }}
            role="button"
            tabIndex={0}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') onSelect(item) }}
            aria-label={item.prompt ? item.prompt.slice(0, 120) : (item.name || 'Generated image')}
            className="relative shrink-0 aspect-square h-full rounded-lg overflow-hidden border border-border
                       hover:border-accent transition-colors group cursor-grab active:cursor-grabbing"
          >
            {item.kind === 'video' ? (
              <div className="w-full h-full bg-card flex items-center justify-center">
                <Film size={18} className="text-muted" />
              </div>
            ) : (
              <img
                src={item.url}
                alt={item.prompt ? item.prompt.slice(0, 120) : (item.name || 'Generated image')}
                className="w-full h-full object-cover hover:opacity-80 transition-opacity"
                draggable={false}
                onLoad={e => {
                  const img = e.currentTarget
                  setImgDims(prev => ({ ...prev, [item.url]: { w: img.naturalWidth, h: img.naturalHeight } }))
                }}
              />
            )}

            {/* Action buttons row — top-right */}
            <div className="absolute top-1.5 right-1.5 flex gap-2 opacity-0 group-hover:opacity-100 transition-all z-10">
              {/* Info button */}
              {item.kind !== 'video' && (
                <button
                  onClick={e => { e.stopPropagation(); setShowInfo(prev => prev === item.url ? null : item.url) }}
                  aria-label="Image dimensions"
                  className="bg-black/70 hover:bg-blue-600 rounded-full p-1.5 transition-colors"
                >
                  <Info size={16} aria-hidden="true" />
                </button>
              )}

              {/* Upscale button */}
              {item.kind !== 'video' && onUpscale && (
                <button
                  onClick={e => {
                    e.stopPropagation()
                    if (!upscaleModelPath) { alert('Load an upscale model in the Upscale section first.'); return }
                    onUpscale(item)
                  }}
                  aria-label="Upscale image"
                  disabled={upscalingItem === item.url}
                  className="bg-black/70 hover:bg-accent rounded-full p-1.5 transition-colors disabled:opacity-60"
                >
                  {upscalingItem === item.url
                    ? <Loader2 size={16} className="animate-spin" aria-hidden="true" />
                    : <ArrowUpCircle size={16} aria-hidden="true" />
                  }
                </button>
              )}

              {/* Load params button */}
              {item.kind !== 'video' && onLoadParams && item.prompt !== undefined && (
                <button
                  onClick={e => { e.stopPropagation(); onLoadParams(item) }}
                  aria-label="Load generation parameters"
                  className="bg-black/70 hover:bg-emerald-600 rounded-full p-1.5 transition-colors"
                >
                  <RotateCcw size={16} aria-hidden="true" />
                </button>
              )}

              {/* Delete button */}
              <button
                onClick={e => { e.stopPropagation(); onDelete(item.name) }}
                aria-label="Delete image"
                className="bg-black/70 hover:bg-red-600 rounded-full p-1.5 transition-colors"
              >
                <Trash2 size={16} aria-hidden="true" />
              </button>
            </div>

            {/* Dimensions overlay — shown inside the thumbnail to avoid overflow-hidden clipping */}
            {showInfo === item.url && item.kind !== 'video' && (
              <div
                onClick={e => e.stopPropagation()}
                className="absolute inset-0 flex items-center justify-center bg-black/60 z-20 pointer-events-none"
              >
                <span className="bg-black/90 text-white text-xs font-mono px-2 py-1 rounded border border-white/20">
                  {imgDims[item.url]
                    ? `${imgDims[item.url].w} × ${imgDims[item.url].h} px`
                    : 'Loading…'
                  }
                </span>
              </div>
            )}

            {/* Prompt tooltip strip on hover */}
            {item.prompt && (
              <div className="absolute inset-x-0 bottom-0 bg-black/75 text-[8px] text-white px-1 py-0.5
                              truncate opacity-0 group-hover:opacity-100 transition-opacity leading-tight">
                {item.prompt}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
