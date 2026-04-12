import { useState, useRef, useEffect, useCallback } from 'react'
import { Square, Pencil, Check, X } from 'lucide-react'

interface Props {
  imageUrl:       string
  initialMaskUrl: string | null
  onClose:        () => void
  onConfirm:      (maskId: string, maskUrl: string) => void
}

type Tool = 'rect' | 'brush'

export default function EraseEditorModal({ imageUrl, initialMaskUrl, onClose, onConfirm }: Props) {
  const displayRef  = useRef<HTMLCanvasElement>(null)   // overlay canvas (user sees)
  // Detached canvas — not in DOM, always present regardless of render state
  const maskRef     = useRef<HTMLCanvasElement>(document.createElement('canvas'))
  const imgRef      = useRef<HTMLImageElement>(null)

  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 })
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 })
  const [tool,        setTool]        = useState<Tool>('rect')
  const [brushSize,   setBrushSize]   = useState(20)
  const [applying,    setApplying]    = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  // Rect tool state
  const [rect, setRect]   = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const draggingRef        = useRef(false)
  const startPtRef         = useRef({ x: 0, y: 0 })

  // Brush tool state
  const brushingRef        = useRef(false)
  const eraseMode          = useRef(false)   // true when Shift held during brush stroke

  function calcDisplay(nw: number, nh: number) {
    const maxW = 760, maxH = 560
    let w = nw, h = nh
    if (w > maxW) { h = Math.round(h * maxW / w); w = maxW }
    if (h > maxH) { w = Math.round(w * maxH / h); h = maxH }
    return { w, h }
  }

  // When image loads: set sizes, init mask canvas, load initial mask
  function onImageLoad() {
    const img = imgRef.current!
    const natural = { w: img.naturalWidth, h: img.naturalHeight }
    const display = calcDisplay(natural.w, natural.h)
    setNaturalSize(natural)
    setDisplaySize(display)

    // Init offscreen mask canvas
    const mc = maskRef.current!
    mc.width  = natural.w
    mc.height = natural.h
    const mctx = mc.getContext('2d')!
    mctx.fillStyle = 'black'
    mctx.fillRect(0, 0, mc.width, mc.height)

    // Load initial mask if provided
    if (initialMaskUrl) {
      const maskImg = new window.Image()
      maskImg.crossOrigin = 'anonymous'
      maskImg.onload = () => {
        mctx.drawImage(maskImg, 0, 0, natural.w, natural.h)
        renderOverlay()
      }
      maskImg.src = initialMaskUrl
    }
  }

  // Render overlay canvas: mask as semi-transparent red over image
  const renderOverlay = useCallback(() => {
    const canvas = displayRef.current
    const mc     = maskRef.current
    if (!canvas || !mc || displaySize.w === 0) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw scaled mask with red tint overlay
    ctx.save()
    ctx.globalAlpha = 0.45
    ctx.drawImage(mc, 0, 0, displaySize.w, displaySize.h)
    ctx.globalCompositeOperation = 'source-atop'
    ctx.fillStyle = 'rgba(220, 50, 50, 1)'
    ctx.fillRect(0, 0, displaySize.w, displaySize.h)
    ctx.restore()

    // Draw in-progress rect selection
    if (tool === 'rect' && rect && rect.w > 2 && rect.h > 2) {
      ctx.strokeStyle = '#7c3aed'
      ctx.lineWidth   = 2
      ctx.setLineDash([5, 4])
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h)
      ctx.setLineDash([])
    }
  }, [displaySize, rect, tool])

  useEffect(() => { renderOverlay() }, [renderOverlay])

  // ── Rect tool handlers ──────────────────────────────────────────────────────

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const r = displayRef.current!.getBoundingClientRect()
    const pos = { x: e.clientX - r.left, y: e.clientY - r.top }
    if (tool === 'rect') {
      draggingRef.current = true
      startPtRef.current  = pos
      setRect({ x: pos.x, y: pos.y, w: 0, h: 0 })
    } else {
      brushingRef.current = true
      eraseMode.current   = e.shiftKey
      paintBrush(pos.x, pos.y, e.shiftKey)
    }
  }

  function paintBrush(cx: number, cy: number, erase: boolean) {
    const mc   = maskRef.current!
    const mctx = mc.getContext('2d')!
    const scaleX = naturalSize.w / displaySize.w
    const scaleY = naturalSize.h / displaySize.h
    const mx = cx * scaleX
    const my = cy * scaleY
    const mr = (brushSize / 2) * Math.max(scaleX, scaleY)
    mctx.fillStyle = erase ? 'black' : 'white'
    mctx.beginPath()
    mctx.arc(mx, my, mr, 0, Math.PI * 2)
    mctx.fill()
    renderOverlay()
  }

  // Window-level mouse tracking — keeps drag/brush alive past canvas edge
  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!displayRef.current) return
      const r   = displayRef.current.getBoundingClientRect()
      const cx  = Math.max(0, Math.min(e.clientX - r.left, r.width))
      const cy  = Math.max(0, Math.min(e.clientY - r.top,  r.height))

      if (draggingRef.current) {
        const sp = startPtRef.current
        setRect({ x: Math.min(sp.x, cx), y: Math.min(sp.y, cy), w: Math.abs(cx - sp.x), h: Math.abs(cy - sp.y) })
      }
      if (brushingRef.current) {
        paintBrush(cx, cy, eraseMode.current)
      }
    }
    function onUp() {
      if (draggingRef.current && rect && rect.w > 2 && rect.h > 2) {
        commitRect()
      }
      draggingRef.current = false
      brushingRef.current = false
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup',   onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup',   onUp)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rect, displaySize, naturalSize, brushSize])

  function commitRect() {
    if (!rect || rect.w < 2 || rect.h < 2 || naturalSize.w === 0) return
    const scaleX = naturalSize.w / displaySize.w
    const scaleY = naturalSize.h / displaySize.h
    const mc   = maskRef.current!
    const mctx = mc.getContext('2d')!
    mctx.fillStyle = 'white'
    mctx.fillRect(
      Math.round(rect.x * scaleX), Math.round(rect.y * scaleY),
      Math.round(rect.w * scaleX), Math.round(rect.h * scaleY),
    )
    setRect(null)
    renderOverlay()
  }

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
      if (e.key === 'Enter')  handleConfirm()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleConfirm() {
    setApplying(true)
    setUploadError(null)
    try {
      const mc = maskRef.current!
      const blob: Blob = await new Promise(res => mc.toBlob(b => res(b!), 'image/png'))
      const fd = new FormData()
      fd.append('file', new File([blob], 'erase_mask.png', { type: 'image/png' }))
      const r = await fetch('/api/upload', { method: 'POST', body: fd })
      if (!r.ok) throw new Error(`Upload failed: ${r.status}`)
      const { id, url } = await r.json()
      onConfirm(id, url)
    } catch (err) {
      setUploadError((err as Error).message ?? 'Upload failed')
    } finally {
      setApplying(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-surface border border-border rounded-xl shadow-2xl flex flex-col gap-3 p-4 max-w-[820px] w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-white">Edit Watermark Mask</span>
          <button onClick={onClose} className="text-muted hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-3">
          <div className="flex gap-1 bg-card rounded-md p-1">
            <button
              onClick={() => setTool('rect')}
              title="Rectangle tool"
              className={`p-1.5 rounded transition-colors ${tool === 'rect' ? 'bg-accent text-white' : 'text-muted hover:text-white'}`}
            >
              <Square size={13} />
            </button>
            <button
              onClick={() => setTool('brush')}
              title="Brush tool (Shift = erase)"
              className={`p-1.5 rounded transition-colors ${tool === 'brush' ? 'bg-accent text-white' : 'text-muted hover:text-white'}`}
            >
              <Pencil size={13} />
            </button>
          </div>

          {tool === 'brush' && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted">Size</span>
              <input
                type="range" min={8} max={64} value={brushSize}
                onChange={e => setBrushSize(Number(e.target.value))}
                className="w-24 accent-violet-500"
              />
              <span className="text-xs text-muted w-6">{brushSize}</span>
            </div>
          )}

          <span className="text-xs text-muted ml-auto">
            {tool === 'brush' ? 'Paint to add mask · Shift+drag to erase' : 'Drag to add rectangle'}
          </span>
        </div>

        {/* Canvas area */}
        <div className="relative bg-black rounded-md overflow-hidden flex items-center justify-center"
             style={{ minHeight: 200 }}>
          {/* Hidden image for loading natural dimensions */}
          <img
            ref={imgRef}
            src={imageUrl}
            onLoad={onImageLoad}
            className="select-none"
            style={{ display: displaySize.w === 0 ? 'block' : 'none', maxWidth: 760, maxHeight: 560 }}
            alt="source"
          />
          {displaySize.w > 0 && (
            <div className="relative" style={{ width: displaySize.w, height: displaySize.h }}>
              {/* Background image */}
              <img
                src={imageUrl}
                className="absolute inset-0 select-none pointer-events-none"
                style={{ width: displaySize.w, height: displaySize.h }}
                alt=""
              />
              {/* Mask overlay canvas */}
              <canvas
                ref={displayRef}
                width={displaySize.w}
                height={displaySize.h}
                className="absolute inset-0 cursor-crosshair"
                onMouseDown={onMouseDown}
              />
              {/* maskRef is a detached canvas — not rendered in DOM */}
            </div>
          )}
        </div>

        {/* Footer */}
        {uploadError && (
          <p className="text-[10px] text-red-400">{uploadError}</p>
        )}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs rounded-md border border-border text-muted hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={applying}
            className="px-3 py-1.5 text-xs rounded-md bg-accent hover:bg-accent/80 text-white
                       flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <Check size={12} />
            {applying ? 'Saving…' : 'Confirm Mask'}
          </button>
        </div>
      </div>
    </div>
  )
}
