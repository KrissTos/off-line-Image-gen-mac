import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Brush, Hexagon, Info, Loader2, MousePointerClick, SquareDashed, X, ZoomIn, ZoomOut } from 'lucide-react'
import type { RefImageSlot } from '../types'
import HelpTip from './HelpTip'
import { segmentMask, segmentPrepare } from '../api'
import {
  coverage, createMask, fillPolygon, fromRgba, grow, invert, paintOverlayFull, paintOverlayRect,
  paintStroke, shrink, strokeRect, subtract, toRgba, union, type Mask, type Pt,
} from '../mask/maskOps'
import { MaskHistory } from '../mask/history'
import { fitView, imageToScreen, inImage, screenToImage, zoomAt, type View } from '../mask/viewMath'

type Tool = 'sam' | 'box' | 'brush' | 'poly'
type SamState = 'loading' | 'ready' | 'running' | 'error'
type SamPoint = { x: number; y: number; label: 0 | 1 }
type LastSam = { points: SamPoint[]; box: { x0: number; y0: number; x1: number; y1: number } | null; op: 'add' | 'sub'; before: Mask }

interface Props {
  slot:         RefImageSlot
  baseImageId:  string        // slot #1 temp id — SAM runs on this image
  baseImageUrl: string
  onClose:      () => void
  onApply:      (maskFile: File) => void
}

const BRUSH_MIN = 4, BRUSH_MAX = 200
const ZOOM_STEP = 1.25

// Hover help for the left column: name + shortcut, how to use it, modifiers.
function tip(name: string, key: string | null, lines: string[]) {
  return (
    <span className="flex flex-col gap-0.5">
      <span className="text-white font-semibold">{name}{key && <span className="text-muted font-normal"> · {key}</span>}</span>
      {lines.map(l => <span key={l}>{l}</span>)}
    </span>
  )
}

const TOOL_TIPS: Record<Tool, React.ReactNode> = {
  sam:   tip('SAM click', 'S', ['Click an object to add it to the mask.', 'Alt+click: remove an object.', 'Shift+click: refine the last object with another point.']),
  box:   tip('SAM box', 'D', ['Drag a box around an object to select it.', 'Alt+drag: remove the object instead.']),
  brush: tip('Brush', 'B', ['Paint to add to the mask.', 'Alt+paint: erase.', '[ / ] or the slider below: brush size.']),
  poly:  tip('Polygon', 'P', ['Click to place points.', 'Enter or click the first point to close.', 'Alt+close: subtract the shape. Esc: cancel.']),
}

const NAV_TIP = tip('Navigate', null, [
  'Wheel / pinch: zoom at the cursor.', '+ / −: zoom in / out.', 'Space+drag: pan.',
  '0: fit to window.', 'M: show/hide mask.', 'Cmd+Z / Shift+Cmd+Z: undo / redo.',
])

const HINTS: Record<Tool, string> = {
  sam:   'Click = add object · Alt+click = remove object · Shift+click = refine last object',
  box:   'Drag a box around an object · Alt = remove',
  brush: 'Paint to add · Alt = erase · [ ] size',
  poly:  'Click points · Enter or click first point to close · Alt+close = subtract',
}

async function blobToMask(blob: Blob, w: number, h: number): Promise<Mask> {
  const bmp = await createImageBitmap(blob)
  const c = document.createElement('canvas'); c.width = w; c.height = h
  const ctx = c.getContext('2d')!
  ctx.drawImage(bmp, 0, 0, w, h)
  return fromRgba(ctx.getImageData(0, 0, w, h).data, w, h, w, h)
}

export default function MaskEditor({ slot, baseImageId, baseImageUrl, onClose, onApply }: Props) {
  const wrapRef    = useRef<HTMLDivElement>(null)
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const imgRef     = useRef<HTMLImageElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement>(document.createElement('canvas'))
  const history    = useRef(new MaskHistory())

  const [mask, setMask]         = useState<Mask | null>(null)
  const [view, setView]         = useState<View>({ scale: 1, tx: 0, ty: 0 })
  const [box, setBox]           = useState({ w: 0, h: 0 })
  const [tool, setTool]         = useState<Tool>('sam')
  const [brush, setBrush]       = useState(24)
  const [growPx, setGrowPx]     = useState(3)
  const [showMask, setShowMask] = useState(true)
  const [sam, setSam]           = useState<SamState>('loading')
  const [samError, setSamError] = useState('')
  const [dirty, setDirty]       = useState(false)
  const [confirmDiscard, setConfirmDiscard] = useState(false)
  const [poly, setPoly]         = useState<Pt[]>([])
  const [drag, setDrag]         = useState<{ a: Pt; b: Pt } | null>(null)   // box tool, image px
  const [cursor, setCursor]     = useState<{ x: number; y: number } | null>(null) // screen px
  const [, forceRender]         = useState(0)
  const lastSam   = useRef<LastSam | null>(null)
  const samBusy   = useRef(false)
  const spaceDown = useRef(false)
  const panning   = useRef<{ x: number; y: number; v: View } | null>(null)
  const painting  = useRef<{ last: Pt; value: 0 | 255 } | null>(null)
  const maskRef   = useRef<Mask | null>(null)               // always the latest mask, for async callbacks
  const overlayImgData = useRef<ImageData | null>(null)
  const overlayMaskData = useRef<Uint8Array | null>(null)   // which mask.data the overlay buffer reflects
  const strokeDirtyRect = useRef<{ x0: number; y0: number; x1: number; y1: number } | null>(null)

  useEffect(() => { maskRef.current = mask }, [mask])

  const w = imgRef.current?.naturalWidth ?? 0
  const h = imgRef.current?.naturalHeight ?? 0

  // ── Load image + existing mask, warm SAM ────────────────────────────────────
  useEffect(() => {
    const img = new Image()
    img.onload = async () => {
      imgRef.current = img
      let m = createMask(img.naturalWidth, img.naturalHeight)
      if (slot.maskUrl) {
        try {
          const blob = await (await fetch(slot.maskUrl)).blob()
          m = await blobToMask(blob, img.naturalWidth, img.naturalHeight)
        } catch { /* start empty if the old mask can't be read */ }
      }
      setMask(m)
    }
    img.src = baseImageUrl
    segmentPrepare(baseImageId)
      .then(() => setSam('ready'))
      .catch(e => { setSam('error'); setSamError((e as Error).message) })
  }, [baseImageId, baseImageUrl, slot.maskUrl])

  // ── Size canvas to its box, fit on first layout ─────────────────────────────
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const fitted = useRef(false)
  useEffect(() => {
    if (!fitted.current && mask && box.w > 0) {
      setView(fitView(mask.w, mask.h, box.w, box.h)); fitted.current = true
    }
  }, [mask, box])

  // ── Mask → red overlay (only when the mask changes) ─────────────────────────
  // During a brush stroke `mask.data` is mutated in place (same Uint8Array across
  // pointermoves), so we repaint only the stroke's dirty rect into a persistent
  // ImageData instead of rebuilding+re-allocating the whole overlay every move.
  useEffect(() => {
    if (!mask) return
    const oc = overlayRef.current
    const sizeChanged = oc.width !== mask.w || oc.height !== mask.h
    const ctx = oc.getContext('2d')!
    const sameBuffer = overlayMaskData.current === mask.data && !sizeChanged
    const rect = strokeDirtyRect.current
    if (sameBuffer && overlayImgData.current && rect) {
      paintOverlayRect(overlayImgData.current.data, mask, rect.x0, rect.y0, rect.x1, rect.y1)
      ctx.putImageData(overlayImgData.current, 0, 0, rect.x0, rect.y0, rect.x1 - rect.x0 + 1, rect.y1 - rect.y0 + 1)
    } else {
      if (sizeChanged) { oc.width = mask.w; oc.height = mask.h }
      const id = ctx.createImageData(mask.w, mask.h)
      paintOverlayFull(id.data, mask)
      ctx.putImageData(id, 0, 0)
      overlayImgData.current = id
    }
    overlayMaskData.current = mask.data
    strokeDirtyRect.current = null
  }, [mask])

  // ── Draw ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const c = canvasRef.current, img = imgRef.current
    if (!c || !img || !mask || box.w === 0) return
    const dpr = window.devicePixelRatio || 1
    const cw = Math.round(box.w * dpr), ch = Math.round(box.h * dpr)
    if (c.width !== cw || c.height !== ch) { c.width = cw; c.height = ch }
    const ctx = c.getContext('2d')!
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, c.width, c.height)
    ctx.setTransform(dpr * view.scale, 0, 0, dpr * view.scale, dpr * view.tx, dpr * view.ty)
    ctx.imageSmoothingEnabled = view.scale < 2
    ctx.drawImage(img, 0, 0)
    if (showMask) ctx.drawImage(overlayRef.current, 0, 0)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)            // screen-space overlays
    ctx.lineWidth = 1.5
    if (drag) {
      const a = imageToScreen(view, drag.a.x, drag.a.y), b = imageToScreen(view, drag.b.x, drag.b.y)
      ctx.strokeStyle = '#7c3aed'; ctx.setLineDash([5, 4])
      ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y))
      ctx.setLineDash([])
    }
    if (poly.length) {
      ctx.strokeStyle = '#facc15'; ctx.fillStyle = '#facc15'
      ctx.beginPath()
      poly.forEach((p, i) => { const s = imageToScreen(view, p.x, p.y); if (i) ctx.lineTo(s.x, s.y); else ctx.moveTo(s.x, s.y) })
      if (cursor) ctx.lineTo(cursor.x, cursor.y)
      ctx.stroke()
      poly.forEach(p => { const s = imageToScreen(view, p.x, p.y); ctx.fillRect(s.x - 3, s.y - 3, 6, 6) })
    }
    if (tool === 'brush' && cursor) {
      ctx.strokeStyle = '#ffffff'; ctx.beginPath(); ctx.arc(cursor.x, cursor.y, brush / 2, 0, Math.PI * 2); ctx.stroke()
    }
  }, [mask, view, box, showMask, drag, poly, cursor, tool, brush])

  // ── Edit helpers ─────────────────────────────────────────────────────────────
  const commit = useCallback((next: Mask, before: Mask, keepSam = false) => {
    history.current.push(before)
    if (!keepSam) lastSam.current = null
    setMask(next); setDirty(true); forceRender(n => n + 1)
  }, [])

  const runSam = useCallback(async (points: SamPoint[], b: LastSam['box'], op: 'add' | 'sub', refine: boolean) => {
    if (!maskRef.current || samBusy.current || sam !== 'ready') return
    const mw = maskRef.current.w, mh = maskRef.current.h
    samBusy.current = true; setSam('running'); setSamError('')
    try {
      const obj = await blobToMask(await segmentMask(baseImageId, points, b), mw, mh)
      // Read the mask fresh: brush/polygon/invert/grow/clear/undo may have run during
      // the 150–400 ms round trip, and their result must not be clobbered. Those edits
      // also null out lastSam.current directly, so a still-truthy lastSam here means no
      // intervening edit happened and the refine base is still valid.
      const current = maskRef.current
      if (!current) { setSam('ready'); return }
      const base = refine && lastSam.current ? lastSam.current.before : current
      const next = op === 'add' ? union(base, obj) : subtract(base, obj)
      if (refine && lastSam.current) {
        setMask(next); setDirty(true)                     // replaces the previous result of the same object
      } else {
        commit(next, current, true)
      }
      lastSam.current = { points, box: b, op, before: base }
      setSam('ready')
    } catch (e) {
      setSamError((e as Error).message)
      setSam('ready')                                       // a click failure isn't fatal — only prepare failure locks SAM
    } finally {
      samBusy.current = false
    }
  }, [sam, baseImageId, commit])

  const closePolygon = useCallback((subtractIt: boolean) => {
    if (!mask) return
    if (poly.length >= 3) {
      const next = { ...mask, data: mask.data.slice() }
      fillPolygon(next, poly, subtractIt ? 0 : 255)
      commit(next, mask)
    }
    setPoly([])
  }, [mask, poly, commit])

  const apply = useCallback(() => {
    if (!mask) return
    const c = document.createElement('canvas'); c.width = mask.w; c.height = mask.h
    c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(toRgba(mask)), mask.w, mask.h), 0, 0)
    c.toBlob(b => { if (b) onApply(new File([b], 'mask.png', { type: 'image/png' })) }, 'image/png')
  }, [mask, onApply])

  const doUndo = useCallback((redo: boolean) => {
    if (!mask) return
    const m = redo ? history.current.redo(mask) : history.current.undo(mask)
    if (m) { lastSam.current = null; setMask(m); setDirty(true) }
  }, [mask])

  const act = useCallback((f: (m: Mask) => Mask) => { if (mask) commit(f(mask), mask) }, [mask, commit])

  // Import a mask PNG (white = masked) as the new mask, scaled to the image; undoable.
  const importRef = useRef<HTMLInputElement>(null)
  const importMask = useCallback(async (file: File) => {
    const before = maskRef.current
    if (!before) return
    try { commit(await blobToMask(file, before.w, before.h), maskRef.current ?? before) }
    catch { alert('Could not read that image as a mask.') }
  }, [commit])

  const maskCoverage = useMemo(() => (mask ? coverage(mask) : 0), [mask])

  // ── Pointer ──────────────────────────────────────────────────────────────────
  const local = (e: { clientX: number; clientY: number }) => {
    const r = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  function onPointerDown(e: React.PointerEvent) {
    if (!mask) return
    const s = local(e), p = screenToImage(view, s.x, s.y)
    if (spaceDown.current || e.button === 1) { panning.current = { x: s.x, y: s.y, v: view }; return }
    if (e.button !== 0) return
    ;(e.target as Element).setPointerCapture(e.pointerId)
    if (tool === 'sam') {
      if (!inImage(p, mask.w, mask.h)) return
      const pt: SamPoint = { x: p.x, y: p.y, label: e.altKey ? 0 : 1 }
      if (e.shiftKey && lastSam.current) runSam([...lastSam.current.points, pt], lastSam.current.box, lastSam.current.op, true)
      else runSam([{ ...pt, label: 1 }], null, e.altKey ? 'sub' : 'add', false)
    } else if (tool === 'box') {
      setDrag({ a: p, b: p })
    } else if (tool === 'brush') {
      const value: 0 | 255 = e.altKey ? 0 : 255
      const r = brush / 2 / view.scale
      history.current.push(mask); lastSam.current = null
      const next = { ...mask, data: mask.data.slice() }
      paintStroke(next, [p], r, value)
      strokeDirtyRect.current = strokeRect([p], r, mask.w, mask.h)
      painting.current = { last: p, value }
      setMask(next); setDirty(true)
    } else if (tool === 'poly') {
      if (poly.length >= 3) {
        const first = imageToScreen(view, poly[0].x, poly[0].y)
        if (Math.hypot(first.x - s.x, first.y - s.y) < 8) { closePolygon(e.altKey); return }
      }
      setPoly([...poly, p])
    }
  }

  function onPointerMove(e: React.PointerEvent) {
    const s = local(e); setCursor(s)
    if (panning.current) {
      const pn = panning.current
      setView({ ...pn.v, tx: pn.v.tx + s.x - pn.x, ty: pn.v.ty + s.y - pn.y }); return
    }
    const p = screenToImage(view, s.x, s.y)
    if (drag) setDrag({ ...drag, b: p })
    if (painting.current && mask) {
      const r = brush / 2 / view.scale
      const next = { ...mask, data: mask.data }            // same buffer: stroke in progress
      paintStroke(next, [painting.current.last, p], r, painting.current.value)
      strokeDirtyRect.current = strokeRect([painting.current.last, p], r, mask.w, mask.h)
      painting.current.last = p
      setMask({ ...next })
    }
  }

  function onPointerUp(e: React.PointerEvent) {
    panning.current = null
    painting.current = null
    if (drag && mask) {
      const a = drag.a, b = drag.b
      const x0 = Math.max(0, Math.min(a.x, b.x)), y0 = Math.max(0, Math.min(a.y, b.y))
      const x1 = Math.min(mask.w, Math.max(a.x, b.x)), y1 = Math.min(mask.h, Math.max(a.y, b.y))
      setDrag(null)
      if ((x1 - x0) * view.scale > 4 && (y1 - y0) * view.scale > 4) runSam([], { x0, y0, x1, y1 }, e.altKey ? 'sub' : 'add', false)
    }
  }

  // ── Wheel zoom (non-passive) ──────────────────────────────────────────────────
  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = c.getBoundingClientRect()
      setView(v => zoomAt(v, e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015)))
    }
    c.addEventListener('wheel', onWheel, { passive: false })
    return () => c.removeEventListener('wheel', onWheel)
  }, [])

  // Buttons/keys zoom around the view centre; the wheel zooms at the cursor.
  const zoomBy = useCallback((f: number) => setView(v => zoomAt(v, box.w / 2, box.h / 2, f)), [box])
  const fit    = () => { if (mask && box.w) setView(fitView(mask.w, mask.h, box.w, box.h)) }

  // ── Keyboard ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLInputElement
      if (t.tagName === 'INPUT' && t.type !== 'range') return   // sliders keep shortcuts working
      if (e.code === 'Space') { spaceDown.current = e.type === 'keydown'; e.preventDefault(); return }
      if (e.type !== 'keydown') return
      const k = e.key.toLowerCase()
      if ((e.metaKey || e.ctrlKey) && k === 'z') { e.preventDefault(); doUndo(e.shiftKey); return }
      if (k === 'escape') {
        if (poly.length) setPoly([])
        else if (confirmDiscard) setConfirmDiscard(false)
        else if (dirty) setConfirmDiscard(true)
        else onClose()
        return
      }
      if (k === 'enter') { if (poly.length) closePolygon(e.altKey); else apply(); return }
      if (e.metaKey || e.ctrlKey) return                    // don't hijack Cmd/Ctrl shortcuts (e.g. Cmd+S)
      if (k === 's') setTool('sam')
      else if (k === 'd') setTool('box')
      else if (k === 'b') setTool('brush')
      else if (k === 'p') setTool('poly')
      else if (k === 'i') act(invert)
      else if (k === 'm') setShowMask(v => !v)
      else if (k === '0') fit()
      else if (k === '=' || k === '+') zoomBy(ZOOM_STEP)
      else if (k === '-') zoomBy(1 / ZOOM_STEP)
      else if (k === '[') setBrush(b => Math.max(BRUSH_MIN, b - 4))
      else if (k === ']') setBrush(b => Math.min(BRUSH_MAX, b + 4))
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKey)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('keyup', onKey) }
  }, [poly, dirty, confirmDiscard, onClose, closePolygon, apply, act, doUndo, mask, box, zoomBy])

  // ── UI ───────────────────────────────────────────────────────────────────────
  const toolBtn = (t: Tool, label: string, key: string, icon: React.ReactNode) => (
    <HelpTip key={t} text={TOOL_TIPS[t]} position="right">
      <button
        onClick={() => setTool(t)} aria-label={`${label} (${key})`} aria-pressed={tool === t}
        className={`w-10 h-10 rounded flex items-center justify-center transition-colors
          ${tool === t ? 'bg-accent text-white' : 'text-muted hover:text-white hover:bg-card'}`}
      >{icon}</button>
    </HelpTip>
  )
  const actBtn = (label: string, onClick: () => void, help: React.ReactNode, disabled = false) => (
    <HelpTip text={help} position="right" className="flex w-full">
      <button onClick={onClick} aria-label={label} disabled={disabled}
        className="w-full px-1 py-1 rounded text-[10px] text-muted hover:text-white hover:bg-card disabled:opacity-40">
        {label}
      </button>
    </HelpTip>
  )
  const iconBtn = (label: string, onClick: () => void, icon: React.ReactNode) => (
    <button onClick={onClick} aria-label={label} disabled={!mask}
      className="w-7 h-7 rounded flex items-center justify-center text-muted hover:text-white hover:bg-card disabled:opacity-40">
      {icon}
    </button>
  )
  const numInput = 'w-full bg-card border border-border rounded text-[11px] text-center text-white py-0.5'
  // sam === 'error' means segmentPrepare failed and SAM is locked; a click failure
  // (caught in runSam) leaves sam 'ready' — usable again — but still shows samError
  // until the next click clears it.
  const samShowsError = sam === 'error' || (sam === 'ready' && !!samError)
  const samLabel = sam === 'loading' ? 'SAM loading…' : sam === 'running' ? 'SAM running…'
    : samShowsError ? `SAM error: ${samError}` : 'SAM ready'

  return (
    <div className="fixed inset-0 z-50 bg-bg flex flex-col" role="dialog" aria-modal="true" aria-label="Mask editor">
      <div className="flex items-center justify-between px-4 h-10 border-b border-border text-xs">
        <span className="text-white font-semibold">
          Mask — {slot.slotId === 1 ? 'base image (#1)' : `pass #${slot.slotId}, drawn on base image (#1)`}
        </span>
        <div className="flex items-center gap-2">
          {confirmDiscard && (
            <span className="flex items-center gap-2 text-amber-300">
              Discard changes?
              <button className="px-2 py-0.5 rounded bg-card border border-border hover:text-white" onClick={onClose}>Discard</button>
              <button className="px-2 py-0.5 rounded bg-card border border-border hover:text-white" onClick={() => setConfirmDiscard(false)}>Keep editing</button>
            </span>
          )}
          <button onClick={apply} disabled={!mask} className="px-3 py-1 rounded bg-accent text-white disabled:opacity-40">Apply (Enter)</button>
          <button onClick={() => (dirty ? setConfirmDiscard(true) : onClose())} aria-label="Close mask editor" className="text-muted hover:text-white">
            <X size={16} />
          </button>
        </div>
      </div>
      <div className="flex flex-1 min-h-0">
        <div className="w-24 shrink-0 border-r border-border flex flex-col items-center gap-1 py-2 px-2 overflow-y-auto">
          {toolBtn('sam', 'SAM click', 'S', <MousePointerClick size={18} />)}
          {toolBtn('box', 'SAM box', 'D', <SquareDashed size={18} />)}
          {toolBtn('brush', 'Brush', 'B', <Brush size={18} />)}
          {toolBtn('poly', 'Polygon', 'P', <Hexagon size={18} />)}
          {tool === 'brush' && (
            <div className="w-full flex flex-col gap-1 pt-1">
              <span className="text-[10px] text-muted">Brush px</span>
              <input type="range" min={BRUSH_MIN} max={BRUSH_MAX} value={brush} aria-label="Brush size"
                onChange={e => setBrush(Number(e.target.value))}
                className="w-full h-1 accent-accent appearance-none bg-border rounded-full" />
              <input type="number" min={BRUSH_MIN} max={BRUSH_MAX} value={brush} aria-label="Brush size in pixels"
                onChange={e => setBrush(Math.max(BRUSH_MIN, Math.min(BRUSH_MAX, Number(e.target.value) || BRUSH_MIN)))}
                className={numInput} />
            </div>
          )}
          <div className="w-full border-t border-border my-1" />
          {actBtn('Invert', () => act(invert), tip('Invert', 'I', ['Swap masked and unmasked areas.']), !mask)}
          <span className="w-full text-[10px] text-muted pt-1">Grow/shrink px</span>
          <input type="number" min={1} max={50} value={growPx} aria-label="Grow/shrink pixels"
            onChange={e => setGrowPx(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
            className={numInput} />
          {actBtn('Grow', () => act(m => grow(m, growPx)), tip('Grow', null, [`Expand the mask edge by ${growPx} px.`]), !mask)}
          {actBtn('Shrink', () => act(m => shrink(m, growPx)), tip('Shrink', null, [`Pull the mask edge in by ${growPx} px.`]), !mask)}
          {actBtn('Clear', () => act(m => createMask(m.w, m.h)), tip('Clear', null, ['Empty the whole mask (undoable).']), !mask)}
          {actBtn('Import…', () => importRef.current?.click(),
            tip('Import mask', null, ['Load a PNG (white = masked) as the mask, e.g. from Photoshop.', 'Replaces the current mask; stretched to the image size. Undoable.']), !mask)}
          <input ref={importRef} type="file" accept="image/*" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) importMask(f); e.target.value = '' }} />
          {actBtn('Undo', () => doUndo(false), tip('Undo', 'Cmd+Z', ['Step back one mask edit.']), !history.current.canUndo)}
          {actBtn('Redo', () => doUndo(true), tip('Redo', 'Shift+Cmd+Z', ['Re-apply an undone edit.']), !history.current.canRedo)}
          <div className="w-full border-t border-border my-1" />
          <span className="w-full text-[10px] text-muted">Zoom</span>
          <div className="w-full flex items-center justify-between">
            {iconBtn('Zoom out (−)', () => zoomBy(1 / ZOOM_STEP), <ZoomOut size={15} />)}
            <span className="text-[10px] text-white tabular-nums">{Math.round(view.scale * 100)}%</span>
            {iconBtn('Zoom in (+)', () => zoomBy(ZOOM_STEP), <ZoomIn size={15} />)}
          </div>
          <div className="w-full flex gap-1">
            {actBtn('Fit', fit, tip('Fit', '0', ['Fit the whole image in the window.']), !mask)}
            {actBtn('100%', () => zoomBy(1 / view.scale), tip('100%', null, ['One image pixel per screen point.']), !mask)}
          </div>
        </div>
        <div ref={wrapRef} className="flex-1 min-w-0 relative overflow-hidden">
          <canvas
            ref={canvasRef}
            style={{ width: box.w, height: box.h, cursor: tool === 'brush' ? 'none' : 'crosshair' }}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
            onPointerLeave={() => setCursor(null)}
          />
          {!mask && <div className="absolute inset-0 flex items-center justify-center text-muted text-sm">Loading image…</div>}
          <HelpTip text={NAV_TIP} position="left" className="absolute top-2 right-2 inline-flex">
            <span className="p-1 rounded bg-card/80 border border-border text-muted hover:text-white cursor-help" aria-label="Navigation help">
              <Info size={14} aria-hidden="true" />
            </span>
          </HelpTip>
        </div>
      </div>
      <div className="h-7 px-4 border-t border-border flex items-center gap-4 text-[11px] text-muted">
        <span className="text-white">{tool.toUpperCase()}</span>
        <span className={`flex items-center gap-1 ${samShowsError ? 'text-red-400' : sam === 'ready' ? 'text-green-400' : 'text-amber-300'}`}>
          {(sam === 'loading' || sam === 'running') && <Loader2 size={11} className="animate-spin" />}
          {samLabel}
        </span>
        <span>Mask {mask ? (maskCoverage * 100).toFixed(1) : '0'}%</span>
        <span>Zoom {Math.round(view.scale * 100)}%</span>
        {tool === 'brush' && <span>Brush {brush}px</span>}
        <span className="truncate">{HINTS[tool]} · Wheel/+/− zoom · Space+drag pan · 0 fit · M mask · I invert</span>
        <span className="ml-auto">{w}×{h}</span>
      </div>
    </div>
  )
}
