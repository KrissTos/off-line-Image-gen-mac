import { X, HardDrive, LogIn, LogOut, CheckCircle2, Download } from 'lucide-react'
import { useState, useEffect } from 'react'
import { fetchStorage, fetchSettings, fetchModels } from '../api'

interface Props {
  open:    boolean
  onClose: () => void
}

export default function SettingsDrawer({ open, onClose }: Props) {
  const [storage, setStorage]     = useState<{ models: unknown[]; summary: string } | null>(null)
  const [hfToken, setHfToken]     = useState('')
  const [hfStatus, setHfStatus]   = useState<string | null>(null)
  const [statusMsg, setStatusMsg] = useState('')
  const [modelChoices, setModelChoices] = useState<string[]>([])
  const [availableModels, setAvailableModels] = useState<string[]>([])

  useEffect(() => {
    if (!open) return
    fetchStorage().then(setStorage).catch(() => {})
    fetchSettings().catch(() => {})
    // Check HF login status
    fetch('/api/hf/status')
      .then(r => r.json())
      .then(d => setHfStatus(d.status ?? null))
      .catch(() => setHfStatus(null))
    // Load model list to show download status
    fetchModels()
      .then(d => { setModelChoices(d.choices); setAvailableModels(d.available) })
      .catch(() => {})
  }, [open])

  async function handleHFLogin() {
    if (!hfToken.trim()) return
    try {
      const r = await fetch('/api/hf/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: hfToken }),
      })
      const d = await r.json()
      setStatusMsg(d.status)
      setHfToken('')
      // Refresh HF status after login
      fetch('/api/hf/status').then(r => r.json()).then(d => setHfStatus(d.status)).catch(() => {})
    } catch (e: unknown) {
      setStatusMsg(`Error: ${(e as Error).message}`)
    }
  }

  async function handleHFLogout() {
    const r = await fetch('/api/hf/logout', { method: 'POST' })
    const d = await r.json()
    setStatusMsg(d.status)
    setHfStatus(null)
  }

  if (!open) return null

  const isLoggedIn = hfStatus && !hfStatus.toLowerCase().includes('not') && !hfStatus.toLowerCase().includes('error')

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />

      {/* Drawer */}
      <div className="fixed right-0 top-0 h-full w-96 bg-surface border-l border-border z-50 flex flex-col shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="text-sm font-semibold text-white">Settings</h2>
          <button onClick={onClose} className="text-muted hover:text-white"><X size={16} /></button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6">

          {/* HuggingFace */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3 flex items-center gap-1.5">
              <LogIn size={13} /> HuggingFace
            </h3>

            {/* Login status */}
            {hfStatus && (
              <div className={`flex items-center gap-2 mb-3 text-xs px-3 py-2 rounded-md
                ${isLoggedIn ? 'bg-green-900/30 text-green-400 border border-green-800/40'
                             : 'bg-card border border-border text-muted'}`}>
                {isLoggedIn
                  ? <><CheckCircle2 size={12} /> Logged in · {hfStatus}</>
                  : <>{hfStatus}</>
                }
              </div>
            )}

            <div className="space-y-2">
              <input
                type="password"
                value={hfToken}
                onChange={e => setHfToken(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleHFLogin()}
                placeholder="hf_…  token"
                className="w-full bg-card border border-border rounded-md px-3 py-2 text-sm text-white
                           placeholder-muted focus:outline-none focus:border-accent"
              />
              <div className="flex gap-2">
                <button onClick={handleHFLogin}
                  className="flex-1 py-2 rounded-md bg-accent text-white text-xs font-medium hover:bg-accent/80 transition-colors">
                  {isLoggedIn ? 'Re-login' : 'Login'}
                </button>
                <button onClick={handleHFLogout}
                  className="py-2 px-3 rounded-md bg-card border border-border text-muted hover:text-white text-xs transition-colors flex items-center gap-1">
                  <LogOut size={13} /> Logout
                </button>
              </div>
              {statusMsg && <p className="text-xs text-muted">{statusMsg}</p>}
            </div>
          </section>

          {/* Models */}
          {modelChoices.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3 flex items-center gap-1.5">
                <Download size={13} /> Models
              </h3>
              <div className="space-y-1.5">
                {modelChoices.map(m => {
                  const isLocal = availableModels.includes(m)
                  return (
                    <div key={m} className="flex items-center justify-between text-xs px-3 py-2 rounded-md bg-card border border-border">
                      <span className={`truncate max-w-[220px] ${isLocal ? 'text-white' : 'text-muted'}`}>{m}</span>
                      <span className={`shrink-0 ml-2 flex items-center gap-1 ${isLocal ? 'text-green-400' : 'text-muted/50'}`}>
                        {isLocal
                          ? <><CheckCircle2 size={11} /> cached</>
                          : <><Download size={11} /> not downloaded</>
                        }
                      </span>
                    </div>
                  )
                })}
                {!isLoggedIn && modelChoices.some(m => !availableModels.includes(m)) && (
                  <p className="text-[10px] text-muted/60 pt-1">
                    Log in to HuggingFace to auto-download gated models on first use.
                  </p>
                )}
              </div>
            </section>
          )}

          {/* Storage */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3 flex items-center gap-1.5">
              <HardDrive size={13} /> Storage
            </h3>
            {storage ? (
              <div className="space-y-2">
                {(storage.models as Array<{ name: string; size: string; choice: string }>).map((m, i) => (
                  <div key={i} className="flex justify-between text-xs">
                    <span className="text-white truncate max-w-[220px]">{m.choice || m.name}</span>
                    <span className="text-muted shrink-0 ml-2">{m.size}</span>
                  </div>
                ))}
                {storage.summary && (
                  <p className="text-xs text-muted pt-1 border-t border-border">{storage.summary}</p>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted">Loading…</p>
            )}
          </section>

        </div>
      </div>
    </>
  )
}
