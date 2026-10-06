import { ChevronDown, ChevronRight } from 'lucide-react'

interface Props {
  label:    string
  count:    number
  open:     boolean
  onToggle: () => void
  level?:   1 | 2          // 1 = category header, 2 = nested family folder
  children: React.ReactNode
}

/** Collapsible header + body used by the Model Sources list (categories and LoRA family folders). */
export default function SourceFolder({ label, count, open, onToggle, level = 1, children }: Props) {
  return (
    <div className={level === 2 ? 'ml-2 pl-2 border-l border-border' : ''}>
      <button
        onClick={onToggle}
        aria-expanded={open}
        className={`w-full flex items-center justify-between py-1 text-left hover:text-white transition-colors ${
          level === 1 ? 'text-[10px] font-semibold uppercase tracking-wider text-muted/80'
                      : 'text-[11px] text-muted'}`}
      >
        <span className="flex items-center gap-1.5">
          {open ? <ChevronDown size={12} aria-hidden="true" /> : <ChevronRight size={12} aria-hidden="true" />}
          {label}
        </span>
        <span className="font-mono text-[10px] text-muted/60">{count}</span>
      </button>
      {open && <div className="space-y-2 mt-1">{children}</div>}
    </div>
  )
}
