// src/components/HelpTip.tsx
/**
 * Inline help tooltip. Renders a small ⓘ icon that shows a tooltip on hover.
 * Uses position:fixed + viewport-aware placement so it never overflows the window.
 * Usage: <HelpTip text="Guidance scale controls…" />
 */
import { useState, useRef, useCallback } from 'react'

interface Props {
  text: string
  position?: 'top' | 'bottom' | 'left' | 'right'
}

const TOOLTIP_W = 224 // w-56 = 14rem

export default function HelpTip({ text, position = 'top' }: Props) {
  const [tipStyle, setTipStyle] = useState<React.CSSProperties | null>(null)
  const ref = useRef<HTMLSpanElement>(null)

  const show = useCallback(() => {
    if (!ref.current) return
    const r   = ref.current.getBoundingClientRect()
    const vw  = window.innerWidth
    const vh  = window.innerHeight
    const GAP = 6

    const s: React.CSSProperties = { position: 'fixed', width: TOOLTIP_W }

    if (position === 'left' || position === 'right') {
      s.top = Math.max(4, Math.min(r.top + r.height / 2 - 40, vh - 84))
      if (position === 'left') s.right = vw - r.left + GAP
      else                     s.left  = r.right + GAP
    } else {
      // top/bottom — flip if not enough room
      const goBelow = position === 'bottom' || r.top < 110
      if (goBelow) s.top    = r.bottom + GAP
      else         s.bottom = vh - r.top + GAP
      // center over trigger, clamped to viewport
      s.left = Math.max(4, Math.min(r.left + r.width / 2 - TOOLTIP_W / 2, vw - TOOLTIP_W - 4))
    }

    setTipStyle(s)
  }, [position])

  return (
    <span
      ref={ref}
      className="relative inline-flex items-center"
      onMouseEnter={show}
      onMouseLeave={() => setTipStyle(null)}
    >
      <span className="w-3.5 h-3.5 rounded-full border border-muted text-muted text-[9px] flex items-center justify-center cursor-help select-none hover:border-white hover:text-white transition-colors">
        i
      </span>
      {tipStyle && (
        <span
          style={tipStyle}
          className="z-[9999] p-2 rounded bg-[#2a2a2a] border border-border text-[11px] text-muted leading-relaxed shadow-lg pointer-events-none"
        >
          {text}
        </span>
      )}
    </span>
  )
}
