// src/components/HelpTip.tsx
/**
 * Inline help tooltip. Renders a small ⓘ icon that shows a tooltip on hover.
 * Usage: <HelpTip text="Guidance scale controls…" />
 */
import { useState } from 'react'

interface Props {
  text: string
  position?: 'top' | 'bottom' | 'left' | 'right'
}

export default function HelpTip({ text, position = 'top' }: Props) {
  const [visible, setVisible] = useState(false)

  const posClass: Record<string, string> = {
    top:    'bottom-full left-1/2 -translate-x-1/2 mb-1.5',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-1.5',
    left:   'right-full top-1/2 -translate-y-1/2 mr-1.5',
    right:  'left-full top-1/2 -translate-y-1/2 ml-1.5',
  }

  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <span className="w-3.5 h-3.5 rounded-full border border-[var(--color-muted)] text-[var(--color-muted)] text-[9px] flex items-center justify-center cursor-help select-none hover:border-white hover:text-white transition-colors">
        i
      </span>
      {visible && (
        <span
          className={`absolute z-50 ${posClass[position]} w-56 p-2 rounded bg-[#2a2a2a] border border-[var(--color-border)] text-[11px] text-[var(--color-muted)] leading-relaxed shadow-lg pointer-events-none`}
        >
          {text}
        </span>
      )}
    </span>
  )
}
