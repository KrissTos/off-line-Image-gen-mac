import { useEffect, useState } from 'react'
import { KeyRound, CheckCircle2 } from 'lucide-react'
import { fetchCivitaiStatus, setCivitaiKey, clearCivitaiKey, updateSettings } from '../api'

interface Props { onNsfwChange: () => void }

/** CivitAI API key (never shown again after save) and the "Show NSFW LoRAs" toggle. */
export default function CivitaiKeyPanel({ onNsfwChange }: Props) {
  const [hasKey, setHasKey] = useState(false)
  const [showNsfw, setShowNsfw] = useState(false)
  const [keyInput, setKeyInput] = useState('')
  const [msg, setMsg] = useState<string | null>(null)

  useEffect(() => {
    fetchCivitaiStatus().then(s => { setHasKey(s.has_key); setShowNsfw(s.show_nsfw) }).catch(() => {})
  }, [])

  async function save() {
    if (!keyInput.trim()) return
    try {
      await setCivitaiKey(keyInput)
      setHasKey(true); setKeyInput(''); setMsg(null)
    } catch (e: any) { setMsg(e.message || 'Could not save the key') }
  }

  async function remove() {
    await clearCivitaiKey().catch(() => {})
    setHasKey(false)
  }

  async function toggleNsfw(next: boolean) {
    setShowNsfw(next)
    await updateSettings({ civitai_show_nsfw: next }).catch(() => {})
    onNsfwChange()
  }

  return (
    <div className="mb-3 p-2 rounded-lg bg-card border border-border space-y-2">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted/80">
        <KeyRound size={11} /> CivitAI
        {hasKey && <span className="flex items-center gap-1 text-green-400 normal-case font-normal"><CheckCircle2 size={11} /> key set</span>}
      </div>
      <div className="flex gap-2">
        <input
          type="password" value={keyInput} onChange={e => setKeyInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && save()}
          placeholder={hasKey ? 'Replace API key' : 'CivitAI API key'}
          aria-label="CivitAI API key"
          className="flex-1 min-w-0 bg-bg border border-border rounded-md px-2 py-1 text-xs text-white placeholder-muted focus:outline-none focus:border-accent"
        />
        <button onClick={save} className="px-2 py-1 rounded-md bg-accent text-white text-[10px] hover:bg-accent/80">Save</button>
        {hasKey && <button onClick={remove} className="px-2 py-1 rounded-md bg-card border border-border text-muted hover:text-white text-[10px]">Remove</button>}
      </div>
      <p className="text-[10px] text-muted/60">Needed for models that require login.</p>
      {msg && <p className="text-[10px] text-red-400">{msg}</p>}
      <label className="flex items-center gap-2 text-[11px] text-muted cursor-pointer">
        <input type="checkbox" checked={showNsfw} onChange={e => toggleNsfw(e.target.checked)} />
        Show NSFW LoRAs <span className="text-muted/60">(then run Update)</span>
      </label>
    </div>
  )
}
