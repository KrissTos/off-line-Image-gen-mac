import { useEffect, useRef, useState } from 'react'
import { Download, Trash2, RefreshCw, CheckCircle2 } from 'lucide-react'
import { startCivitaiDownload, fetchCivitaiJob, deleteCivitai, type ModelSource, type CivitaiJob } from '../api'
import { civitaiRowState, formatSize, progressPct } from '../sourceGroups'

interface Props { src: ModelSource; onChanged: () => void }

/** Download / Installed / Update / Delete controls for one CivitAI LoRA row. */
export default function CivitaiRowActions({ src, onChanged }: Props) {
  const vid = src.civitai?.versionId
  const [job, setJob] = useState<CivitaiJob | null>(null)
  const [confirming, setConfirming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)
  const state = civitaiRowState(src)
  const busy = job?.state === 'queued' || job?.state === 'downloading' || job?.state === 'verifying'

  useEffect(() => () => { if (timer.current) window.clearInterval(timer.current) }, [])

  function notify() {
    window.dispatchEvent(new Event('lora-library-changed'))
    onChanged()
  }

  function poll() {
    if (vid == null) return
    if (timer.current) window.clearInterval(timer.current)
    timer.current = window.setInterval(async () => {
      const j = await fetchCivitaiJob(vid).catch(() => null)
      if (!j) return
      setJob(j)
      if (j.state === 'done' || j.state === 'error') {
        if (timer.current) window.clearInterval(timer.current)
        timer.current = null
        if (j.state === 'error') setError(j.error ?? 'Download failed')
        else { setJob(null); notify() }
      }
    }, 1000)
  }

  async function download() {
    if (vid == null) return
    setError(null)
    try { setJob(await startCivitaiDownload(vid)); poll() }
    catch (e: any) { setError(e.message || 'Download failed') }
  }

  async function remove() {
    const iv = src.installed_version
    if (iv == null) return
    setConfirming(false); setError(null)
    try { await deleteCivitai(iv); notify() }
    catch (e: any) { setError(e.message || 'Delete failed') }
  }

  const size = formatSize(src.civitai?.sizeKB)
  return (
    <div className="mt-1.5">
      <div className="flex items-center gap-1.5 text-[10px]">
        {size && <span className="text-muted/60 font-mono">{size}</span>}
        <span className="flex-1" />
        {busy ? (
          <span className="flex items-center gap-1 text-muted"><RefreshCw size={11} className="animate-spin" /> {job?.state === 'verifying' ? 'Verifying' : `${progressPct(job?.bytes, job?.total)}%`}</span>
        ) : confirming ? (
          <>
            <span className="text-muted">Delete file?</span>
            <button onClick={remove} className="px-2 py-0.5 rounded bg-red-900/40 text-red-300 hover:bg-red-900/60">Delete</button>
            <button onClick={() => setConfirming(false)} className="px-2 py-0.5 rounded bg-card border border-border text-muted hover:text-white">Cancel</button>
          </>
        ) : (
          <>
            {state !== 'download' && (
              <span className="flex items-center gap-1 text-green-400"><CheckCircle2 size={11} /> Installed</span>
            )}
            {state !== 'installed' && (
              <button onClick={download} className="flex items-center gap-1 px-2 py-0.5 rounded bg-accent/20 text-accent border border-accent/30 hover:bg-accent/30">
                <Download size={11} /> {state === 'update' ? 'Update' : 'Download'}
              </button>
            )}
            {state !== 'download' && (
              <button onClick={() => setConfirming(true)} title="Delete the downloaded file" aria-label="Delete the downloaded file"
                className="p-1 rounded text-muted hover:text-red-400 hover:bg-red-900/20"><Trash2 size={11} /></button>
            )}
          </>
        )}
      </div>
      {busy && (
        <div className="w-full h-1 mt-1 bg-border rounded-full overflow-hidden">
          <div className="h-full bg-accent transition-all duration-200" style={{ width: `${progressPct(job?.bytes, job?.total)}%` }} />
        </div>
      )}
      {error && <p className="text-[10px] text-red-400 mt-1">{error}</p>}
    </div>
  )
}
