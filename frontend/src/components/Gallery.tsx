import { useRef } from 'react'
import { Film, Trash2 } from 'lucide-react'
import type { OutputItem } from '../types'

interface Props {
  outputs:  OutputItem[]
  onSelect: (item: OutputItem) => void
  onDelete: (filename: string) => void
}

export default function Gallery({ outputs, onSelect, onDelete }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)

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
      className="h-full border-t border-border bg-surface px-4 py-2 overflow-x-auto overflow-y-hidden"
    >
      <div className="flex gap-2 h-full items-center">
        {outputs.map((item) => (
          <button
            key={item.url}
            onClick={() => onSelect(item)}
            title={[item.name, item.prompt].filter(Boolean).join('\n')}
            className="relative shrink-0 aspect-square h-full rounded-lg overflow-hidden border border-border
                       hover:border-accent transition-colors group"
          >
            {item.kind === 'video' ? (
              <div className="w-full h-full bg-card flex items-center justify-center">
                <Film size={18} className="text-muted" />
              </div>
            ) : (
              <img src={item.url} alt={item.prompt ? item.prompt.slice(0, 120) : (item.name || 'Generated image')} className="w-full h-full object-cover" />
            )}

            {/* Delete button */}
            <button
              onClick={e => { e.stopPropagation(); onDelete(item.name) }}
              aria-label="Delete image"
              className="absolute top-1 right-1 bg-black/70 hover:bg-red-600 rounded-full p-0.5
                         opacity-0 group-hover:opacity-100 transition-all z-10"
            >
              <Trash2 size={8} aria-hidden="true" />
            </button>

            {/* Prompt tooltip strip on hover */}
            {item.prompt && (
              <div className="absolute inset-x-0 bottom-0 bg-black/75 text-[8px] text-white px-1 py-0.5
                              truncate opacity-0 group-hover:opacity-100 transition-opacity leading-tight">
                {item.prompt}
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
