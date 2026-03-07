import { Settings, Zap, Cpu } from 'lucide-react'
import type { AppStatus } from '../types'

interface Props {
  status:        AppStatus
  onOpenSettings: () => void
}

export default function TopBar({ status, onOpenSettings }: Props) {
  const vram = status.vram_gb.toFixed(1)
  const modelShort = status.model
    ? status.model.replace(' (4bit SDNQ - Low VRAM)', ' 4B').replace(' (4bit SDNQ - Higher Quality)', ' 9B')
        .replace(' (Quantized - Fast)', ' Q').replace(' (Full - LoRA support)', ' Full')
        .replace(' (Int8)', ' int8').replace('  (txt2video · img2video with ref)', '')
    : 'No model'

  return (
    <header className="flex items-center justify-between px-5 py-3 border-b border-border bg-surface shrink-0">
      {/* Brand */}
      <div className="flex items-center gap-2">
        <Zap size={18} className="text-accent" />
        <span className="text-sm font-semibold text-white tracking-wide">
          ultra-fast-image-gen
        </span>
      </div>

      {/* Model badge */}
      <div className="flex items-center gap-3">
        {status.loaded && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-card border border-border text-xs">
            <Cpu size={12} className="text-accent" />
            <span className="text-white font-medium">{modelShort}</span>
            {status.device && (
              <span className="text-muted">· {status.device}</span>
            )}
            {status.vram_gb > 0 && (
              <span className="text-muted">· {vram} GB</span>
            )}
          </div>
        )}
        {status.busy && (
          <span className="px-2 py-0.5 rounded-full bg-accent/20 text-accent text-xs font-medium animate-pulse">
            generating…
          </span>
        )}
        <button
          onClick={onOpenSettings}
          className="p-1.5 rounded-md text-muted hover:text-white hover:bg-card transition-colors"
        >
          <Settings size={16} />
        </button>
      </div>
    </header>
  )
}
