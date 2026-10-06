import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Copy, Check, Trash2, ChevronLeft, ChevronRight, Download } from 'lucide-react'
import type { OutputItem } from '../types'
import type { WorkflowData } from '../workflow'
import { loadRun } from '../api'
import { neighbor, paramRows, loraRows, upscaleSource, keyAction } from '../previewModel'

interface Props {
  item:       OutputItem
  outputs:    OutputItem[]
  canLoad:    boolean
  onClose:    () => void
  onNavigate: (item: OutputItem) => void
  onLoad:     (item: OutputItem) => void
  onDelete:   (item: OutputItem) => Promise<boolean>
}

export default function GalleryPreview({ item, outputs, canLoad, onClose, onNavigate, onLoad, onDelete }: Props) {
  const [wf, setWf]         = useState<WorkflowData | null>(null)
  const [wfError, setWfError] = useState(false)
  const [copied, setCopied] = useState(false)
  const [deleteFailed, setDeleteFailed] = useState(false)
  const prev = neighbor(outputs, item.url, -1)
  const next = neighbor(outputs, item.url, 1)

  // One fetch per run (outputs of the same run share it); `cancelled` drops a slow earlier response.
  useEffect(() => {
    setWf(null); setWfError(false)
    if (!item.run) return
    let cancelled = false
    loadRun(item.run)
      .then(d => { if (!cancelled) setWf(d) })
      .catch(() => { if (!cancelled) setWfError(true) })
    return () => { cancelled = true }
  }, [item.run])

  useEffect(() => { setDeleteFailed(false) }, [item.url])

  // Focus: remember the opener to hand focus back on close; Tab cycles inside the dialog.
  const dialogRef = useRef<HTMLDivElement>(null)
  // Captured during the first render: autoFocus on the close button has already run by effect time.
  const [opener] = useState(() => document.activeElement as HTMLElement | null)
  useEffect(() => () => { if (opener?.isConnected) opener.focus() }, [opener])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null
      const act = keyAction({
        key: e.key, metaKey: e.metaKey, ctrlKey: e.ctrlKey, altKey: e.altKey,
        targetTag: t?.tagName ?? '', targetEditable: !!t?.isContentEditable,
      })
      if (act === 'close') onClose()
      else if (act === 'prev' && prev) onNavigate(prev)
      else if (act === 'next' && next) onNavigate(next)
      else if (e.key === 'Tab' && dialogRef.current) {
        const els = [...dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), a[href], video[controls]')]
        if (!els.length) return
        const first = els[0], last = els[els.length - 1]
        if (!dialogRef.current.contains(document.activeElement)) { e.preventDefault(); first.focus() }
        else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onNavigate, prev, next])

  // Close on backdrop only when press AND release land on it (a text selection that ends there must not close).
  const downOnBackdrop = useRef(false)

  const slots   = wf?.ref_slots ?? []
  const source  = upscaleSource(outputs, wf, item)
  const params  = paramRows(item, slots.some(s => s.maskUrl))
  const loras   = loraRows(item)
  const loadOk  = canLoad && !!item.run && !!wf && !wfError

  async function copyPrompt() {
    try {
      await navigator.clipboard.writeText(item.prompt ?? '')
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch { /* clipboard unavailable */ }
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-6"
      onMouseDown={e => { downOnBackdrop.current = e.target === e.currentTarget }}
      onClick={e => { if (e.target === e.currentTarget && downOnBackdrop.current) onClose() }}
    >
      <div
        ref={dialogRef}
        role="dialog" aria-modal="true" aria-label="Output preview"
        className="relative flex w-full max-w-6xl h-[85vh] bg-surface border border-border rounded-xl overflow-hidden"
      >
        {/* Image */}
        <div className="relative flex-1 min-w-0 bg-bg flex items-center justify-center">
          {item.kind === 'video'
            ? <video src={item.url} controls className="max-w-full max-h-full" />
            : <img src={item.url} alt={item.prompt?.slice(0, 120) || item.name} className="max-w-full max-h-full object-contain" />}
          {prev && (
            <button onClick={() => onNavigate(prev)} aria-label="Previous output"
              className="absolute left-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/60 hover:bg-accent transition-colors">
              <ChevronLeft size={18} aria-hidden="true" />
            </button>
          )}
          {next && (
            <button onClick={() => onNavigate(next)} aria-label="Next output"
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-full bg-black/60 hover:bg-accent transition-colors">
              <ChevronRight size={18} aria-hidden="true" />
            </button>
          )}
        </div>

        {/* Details */}
        <div className="w-80 shrink-0 flex flex-col border-l border-border">
          <div className="flex items-center justify-between px-3 py-2 border-b border-border">
            <span className="text-xs text-muted truncate">{item.run ?? item.name}</span>
            <button onClick={onClose} autoFocus aria-label="Close preview"
              className="p-1 rounded text-muted hover:text-white transition-colors">
              <X size={16} aria-hidden="true" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-4 text-sm">
            <section>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-[11px] uppercase tracking-wide text-muted">Prompt</h3>
                {item.prompt && (
                  <button onClick={copyPrompt} aria-label="Copy prompt"
                    className="flex items-center gap-1 text-xs text-muted hover:text-white transition-colors">
                    {copied ? <Check size={12} aria-hidden="true" /> : <Copy size={12} aria-hidden="true" />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                )}
              </div>
              <p className="whitespace-pre-wrap break-words text-white/90">{item.prompt || '—'}</p>
            </section>

            {params.length > 0 && (
              <section>
                <h3 className="text-[11px] uppercase tracking-wide text-muted mb-1">Parameters</h3>
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
                  {params.map(r => (
                    <div key={r.label} className="contents">
                      <dt className="text-muted">{r.label}</dt>
                      <dd className="text-white/90 font-mono break-all">{r.value}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            )}

            {loras.length > 0 && (
              <section>
                <h3 className="text-[11px] uppercase tracking-wide text-muted mb-1">LoRAs</h3>
                <ul className="text-xs space-y-0.5">
                  {loras.map(r => (
                    <li key={r.label} className="flex justify-between gap-2">
                      <span className="truncate">{r.label}</span>
                      <span className="font-mono text-muted">{r.value}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {source && (
              <button onClick={() => onNavigate(source)}
                className="text-xs text-accent hover:underline">
                Upscaled from {source.file?.split('/').pop()}
              </button>
            )}

            {slots.length > 0 && (
              <section>
                <h3 className="text-[11px] uppercase tracking-wide text-muted mb-1">Reference images</h3>
                <div className="flex flex-wrap gap-2">
                  {slots.map((s, i) => (
                    <div key={s.imageUrl} className="relative w-16 h-16 rounded border border-border overflow-hidden bg-card">
                      <img src={s.imageUrl} alt={i === 0 ? 'Base image' : `Reference ${i}`} className="w-full h-full object-cover" />
                      <span className="absolute bottom-0 left-0 bg-black/70 text-[9px] px-1">{i === 0 ? 'base' : `ref ${i}`}</span>
                      {s.maskUrl && <span className="absolute top-0 right-0 bg-accent text-[9px] px-1">mask</span>}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {deleteFailed && (
              <p role="alert" className="text-xs text-red-400">Delete failed. The file is still there.</p>
            )}

            {(!item.run || wfError) && (
              <p className="text-xs text-muted">
                {item.run ? 'Could not read this run\'s workflow.' : 'No workflow saved for this output.'}
              </p>
            )}
          </div>

          <div className="p-3 border-t border-border flex gap-2">
            <button onClick={() => onLoad(item)} disabled={!loadOk}
              className="flex-1 px-3 py-1.5 rounded bg-accent text-white text-sm hover:opacity-90 transition-opacity
                         disabled:opacity-40 disabled:cursor-not-allowed">
              Load in workflow
            </button>
            <a href={item.url} download aria-label="Download"
              className="p-2 rounded bg-card border border-border text-muted hover:text-white hover:border-accent transition-colors">
              <Download size={16} aria-hidden="true" />
            </a>
            <button onClick={async () => setDeleteFailed(!(await onDelete(item)))} aria-label="Delete output"
              className="p-2 rounded bg-card border border-border text-muted hover:text-white hover:bg-red-600 transition-colors">
              <Trash2 size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}
