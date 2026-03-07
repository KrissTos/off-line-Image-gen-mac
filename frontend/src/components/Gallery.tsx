import { Film } from 'lucide-react'
import type { OutputItem } from '../types'

interface Props {
  outputs:  OutputItem[]
  onSelect: (item: OutputItem) => void
}

export default function Gallery({ outputs, onSelect }: Props) {
  if (outputs.length === 0) return null

  return (
    <div className="shrink-0 border-t border-border bg-surface px-4 py-3">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {outputs.map((item) => (
          <button
            key={item.url}
            onClick={() => onSelect(item)}
            title={[item.name, item.prompt].filter(Boolean).join('\n')}
            className="relative shrink-0 w-16 h-16 rounded-lg overflow-hidden border border-border
                       hover:border-accent transition-colors group"
          >
            {item.kind === 'video' ? (
              <div className="w-full h-full bg-card flex items-center justify-center">
                <Film size={18} className="text-muted" />
              </div>
            ) : (
              <img src={item.url} alt="" className="w-full h-full object-cover" />
            )}

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
