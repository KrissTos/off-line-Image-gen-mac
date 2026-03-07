import { useRef, useState, useCallback } from 'react'
import { ImageIcon, Loader2 } from 'lucide-react'

interface Props {
  resultUrl:     string | null
  isGenerating:  boolean
  progressMsg:   string
  progressStep?: number
  progressTotal?: number
  error?:        string | null
  onDropRef?:    (file: File) => void
}

export default function Canvas({
  resultUrl, isGenerating, progressMsg, progressStep, progressTotal, error, onDropRef,
}: Props) {
  const [dragOver, setDragOver] = useState(false)
  const dropRef = useRef<HTMLDivElement>(null)

  // Progress percentage
  const pct = progressStep && progressTotal ? Math.round((progressStep / progressTotal) * 100) : null

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (!onDropRef) return
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) onDropRef(file)
  }, [onDropRef])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (onDropRef) setDragOver(true)
  }, [onDropRef])

  const isVideo = resultUrl?.match(/\.(mp4|webm|mov)(\?|$)/i) !== null

  return (
    <div
      ref={dropRef}
      className={`flex-1 h-full flex items-center justify-center bg-bg p-6 min-h-0 transition-colors
                  ${dragOver ? 'bg-accent/5 border-2 border-dashed border-accent' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={() => setDragOver(false)}
    >
      <div className="relative w-full h-full max-w-3xl rounded-xl overflow-hidden border border-border bg-card flex items-center justify-center">

        {/* Result: video */}
        {resultUrl && isVideo && !isGenerating && (
          <video
            key={resultUrl}
            src={resultUrl}
            controls
            autoPlay
            loop
            className="w-full h-full object-contain"
          />
        )}

        {/* Result: image */}
        {resultUrl && !isVideo && !isGenerating && (
          <img
            key={resultUrl}
            src={resultUrl}
            alt="Generated output"
            className="w-full h-full object-contain"
          />
        )}

        {/* Empty state */}
        {!resultUrl && !isGenerating && !error && (
          <div className="flex flex-col items-center gap-3 text-muted select-none">
            <ImageIcon size={40} strokeWidth={1.2} />
            <p className="text-sm">Your image will appear here</p>
            {onDropRef && (
              <p className="text-xs opacity-60">Drag & drop an image to use as reference</p>
            )}
          </div>
        )}

        {/* Error state */}
        {error && !isGenerating && (
          <div className="flex flex-col items-center gap-2 text-red-400 max-w-sm text-center p-6">
            <p className="text-sm font-medium">Generation failed</p>
            <p className="text-xs opacity-80">{error}</p>
          </div>
        )}

        {/* Generating overlay */}
        {isGenerating && (
          <div className="absolute inset-0 bg-bg/80 backdrop-blur-sm flex flex-col items-center justify-center gap-4">
            <Loader2 size={36} className="text-accent animate-spin" />
            {progressMsg && (
              <p className="text-sm text-muted max-w-xs text-center">{progressMsg}</p>
            )}
            {pct !== null && (
              <div className="w-48 h-1 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent transition-all duration-300 rounded-full"
                  style={{ width: `${pct}%` }}
                />
              </div>
            )}
          </div>
        )}

        {/* Drag-over overlay */}
        {dragOver && (
          <div className="absolute inset-0 bg-accent/10 flex items-center justify-center">
            <p className="text-accent font-medium text-sm">Drop image here</p>
          </div>
        )}
      </div>
    </div>
  )
}
