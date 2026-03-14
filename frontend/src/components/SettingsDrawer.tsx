import {
  X, HardDrive, LogIn, LogOut, CheckCircle2, Download, Trash2,
  RefreshCw, AlertCircle, FolderOpen, Save, CloudDownload, ArrowDownCircle, Palette,
} from 'lucide-react'
import { useState, useEffect, useCallback } from 'react'
import {
  fetchStorage, fetchSettings, updateSettings, fetchModels, deleteModel,
  checkModelUpdates, updateModel, openFolderDialog, openOutputFolder,
  fetchModelExtras, deleteUpscaleModel,
  type ModelUpdateResult, type ModelExtras,
} from '../api'
import { applyThemeColors } from '../App'

interface Props {
  open:    boolean
  onClose: () => void
}

interface StorageModel {
  name:   string
  size:   string
  choice: string
}

const DEFAULT_OUTPUT_DIR = '~/Pictures/ultra-fast-image-gen'

export default function SettingsDrawer({ open, onClose }: Props) {
  const [storage, setStorage]           = useState<{ models: StorageModel[]; summary: string } | null>(null)
  const [hfToken, setHfToken]           = useState('')
  const [hfStatus, setHfStatus]         = useState<string | null>(null)
  const [statusMsg, setStatusMsg]       = useState('')
  const [modelChoices, setModelChoices] = useState<string[]>([])
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [deletingModel, setDeletingModel]     = useState<string | null>(null)
  const [deleteConfirm, setDeleteConfirm]     = useState<string | null>(null)
  const [refreshing, setRefreshing]           = useState(false)
  // Output folder
  const [outputDir, setOutputDir]         = useState('')
  const [outputDirSaved, setOutputDirSaved]   = useState(false)
  const [outputDirSaving, setOutputDirSaving] = useState(false)
  const [folderPicking, setFolderPicking]     = useState(false)
  // Model updates
  const [updateResults, setUpdateResults]     = useState<ModelUpdateResult[] | null>(null)
  const [checkingUpdates, setCheckingUpdates] = useState(false)
  const [updatingModel, setUpdatingModel]     = useState<string | null>(null)
  // Extra models (upscale)
  const [extras, setExtras]                   = useState<ModelExtras | null>(null)
  const [deletingUpscale, setDeletingUpscale] = useState<string | null>(null)
  // Theme colors
  const DEFAULT_THEME: Record<string, string> = {
    bg: '#0a0a0a', surface: '#141414', card: '#1c1c1c',
    border: '#2a2a2a', accent: '#7c3aed', muted: '#6b7280', label: '#6b7280',
  }
  const THEME_LABELS: Record<string, string> = {
    bg: 'Background', surface: 'Surface', card: 'Card',
    border: 'Border', accent: 'Accent', muted: 'Muted text', label: 'Section labels',
  }
  const [themeColors, setThemeColors] = useState<Record<string, string>>(DEFAULT_THEME)
  const [themeSaved, setThemeSaved]   = useState(false)
  const [themeSaving, setThemeSaving] = useState(false)

  const loadData = useCallback(async () => {
    setRefreshing(true)
    try {
      await Promise.all([
        fetchStorage().then(setStorage).catch(() => {}),
        fetchSettings()
          .then(s => {
            setOutputDir((s.output_dir as string) || DEFAULT_OUTPUT_DIR)
            if (s.theme_colors && typeof s.theme_colors === 'object') {
              setThemeColors({ ...DEFAULT_THEME, ...(s.theme_colors as Record<string, string>) })
            }
          })
          .catch(() => setOutputDir(DEFAULT_OUTPUT_DIR)),
        fetch('/api/hf/status')
          .then(r => r.json())
          .then(d => setHfStatus(d.status ?? null))
          .catch(() => setHfStatus(null)),
        fetchModels()
          .then(d => { setModelChoices(d.choices); setAvailableModels(d.available) })
          .catch(() => {}),
        fetchModelExtras().then(setExtras).catch(() => {}),
      ])
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    if (!open) return
    loadData()
    setUpdateResults(null)
  }, [open, loadData])

  // ── HF login/logout ──────────────────────────────────────────────────────
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

  // ── Delete model ─────────────────────────────────────────────────────────
  async function handleDeleteModel(modelName: string) {
    if (deleteConfirm !== modelName) { setDeleteConfirm(modelName); return }
    setDeleteConfirm(null)
    setDeletingModel(modelName)
    try {
      await deleteModel(modelName)
      await Promise.all([
        fetchModels().then(d => { setModelChoices(d.choices); setAvailableModels(d.available) }).catch(() => {}),
        fetchStorage().then(setStorage).catch(() => {}),
      ])
    } catch (e: unknown) {
      setStatusMsg(`Delete failed: ${(e as Error).message}`)
    } finally {
      setDeletingModel(null)
    }
  }

  // ── Check for updates ────────────────────────────────────────────────────
  async function handleCheckUpdates() {
    setCheckingUpdates(true)
    setUpdateResults(null)
    try {
      const data = await checkModelUpdates()
      setUpdateResults(data.results)
    } catch (e: unknown) {
      setStatusMsg(`Check failed: ${(e as Error).message}`)
    } finally {
      setCheckingUpdates(false)
    }
  }

  async function handleUpdateModel(choice: string) {
    setUpdatingModel(choice)
    try {
      const data = await updateModel(choice)
      setStatusMsg(data.status)
      // Re-check after update
      const updated = await checkModelUpdates()
      setUpdateResults(updated.results)
      await fetchModels().then(d => { setModelChoices(d.choices); setAvailableModels(d.available) }).catch(() => {})
    } catch (e: unknown) {
      setStatusMsg(`Update failed: ${(e as Error).message}`)
    } finally {
      setUpdatingModel(null)
    }
  }

  // ── Extra model delete ────────────────────────────────────────────────────
  async function handleDeleteUpscale(name: string) {
    setDeletingUpscale(name)
    try {
      await deleteUpscaleModel(name)
      const e = await fetchModelExtras()
      setExtras(e)
      await fetchStorage().then(setStorage).catch(() => {})
    } catch (e: unknown) {
      setStatusMsg(`Delete failed: ${(e as Error).message}`)
    } finally {
      setDeletingUpscale(null)
    }
  }

  // ── Output folder ────────────────────────────────────────────────────────
  async function handlePickFolder() {
    setFolderPicking(true)
    try {
      const data = await openFolderDialog()
      if (!data.cancelled && data.path) {
        setOutputDir(data.path)
        setOutputDirSaved(false)
      }
    } catch (e: unknown) {
      setStatusMsg(`Folder picker failed: ${(e as Error).message}`)
    } finally {
      setFolderPicking(false)
    }
  }

  async function handleSaveOutputDir() {
    setOutputDirSaving(true)
    try {
      await updateSettings({ output_dir: outputDir.trim() || DEFAULT_OUTPUT_DIR })
      setOutputDirSaved(true)
      setTimeout(() => setOutputDirSaved(false), 2000)
    } catch (e: unknown) {
      setStatusMsg(`Save failed: ${(e as Error).message}`)
    } finally {
      setOutputDirSaving(false)
    }
  }

  async function handleSaveTheme() {
    setThemeSaving(true)
    try {
      await updateSettings({ theme_colors: themeColors })
      applyThemeColors(themeColors)
      setThemeSaved(true)
      setTimeout(() => setThemeSaved(false), 2000)
    } catch (e: unknown) {
      setStatusMsg(`Theme save failed: ${(e as Error).message}`)
    } finally {
      setThemeSaving(false)
    }
  }

  function handleResetTheme() {
    setThemeColors(DEFAULT_THEME)
    applyThemeColors(DEFAULT_THEME)
  }

  if (!open) return null

  const isLoggedIn = hfStatus && !hfStatus.toLowerCase().includes('not') && !hfStatus.toLowerCase().includes('error')

  // Map choice name → storage size
  const storageByChoice: Record<string, string> = {}
  if (storage?.models) {
    for (const m of storage.models) {
      if (m.choice) storageByChoice[m.choice] = m.size
    }
  }

  // Merge update results into per-model map
  const updateByChoice: Record<string, ModelUpdateResult> = {}
  if (updateResults) {
    for (const r of updateResults) updateByChoice[r.choice] = r
  }

  const UPDATE_LABEL: Record<string, { text: string; ariaLabel: string; color: string }> = {
    up_to_date:       { text: '✓ Up to date',       ariaLabel: 'Up to date',       color: 'text-green-400' },
    update_available: { text: '🔄 Update available', ariaLabel: 'Update available', color: 'text-amber-400' },
    not_downloaded:   { text: '⬇ Not downloaded',   ariaLabel: 'Not downloaded',   color: 'text-muted' },
    error:            { text: '⚠ Error',             ariaLabel: 'Error',            color: 'text-red-400' },
  }

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />

      {/* Drawer */}
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-drawer-title"
        className="fixed right-0 top-0 h-full w-96 bg-surface border-l border-border z-50 flex flex-col shadow-2xl"
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 id="settings-drawer-title" className="text-sm font-semibold text-white">Settings</h2>
          <div className="flex items-center gap-2">
            <button onClick={loadData} disabled={refreshing} title="Refresh" aria-label="Refresh settings"
              className="text-muted hover:text-white transition-colors disabled:opacity-40">
              <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} aria-hidden="true" />
            </button>
            <button onClick={onClose} aria-label="Close settings" className="text-muted hover:text-white">
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6">

          {/* ── Output Folder ── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-label mb-3 flex items-center gap-1.5">
              <FolderOpen size={13} /> Output Folder
            </h3>
            <p className="text-[10px] text-muted/70 mb-2">
              Where generated images and videos are saved on your Mac.
            </p>
            {/* Row 1: action buttons */}
            <div className="flex gap-2 mb-2">
              <button
                onClick={handlePickFolder}
                disabled={folderPicking}
                title="Browse for folder"
                className="flex-1 px-3 py-2 rounded-md bg-card border border-border text-muted
                           hover:text-white text-xs transition-colors disabled:opacity-40 flex items-center justify-center gap-1.5"
              >
                {folderPicking ? <RefreshCw size={12} className="animate-spin" /> : <FolderOpen size={12} />}
                Browse…
              </button>
              <button
                onClick={handleSaveOutputDir}
                disabled={outputDirSaving}
                className={`flex-1 px-3 py-2 rounded-md text-xs font-medium transition-colors flex items-center justify-center gap-1.5
                  ${outputDirSaved
                    ? 'bg-green-700/60 text-green-300 border border-green-700/50'
                    : 'bg-accent text-white hover:bg-accent/80'
                  } disabled:opacity-40`}
              >
                {outputDirSaving ? <RefreshCw size={12} className="animate-spin" /> : outputDirSaved ? <CheckCircle2 size={12} /> : <Save size={12} />}
                {outputDirSaved ? 'Saved' : 'Save'}
              </button>
              <button
                onClick={() => openOutputFolder().catch(() => {})}
                title="Open output folder in Finder"
                aria-label="Open output folder in Finder"
                className="flex-1 px-3 py-2 rounded-md bg-card border border-border text-muted
                           hover:text-white text-xs transition-colors flex items-center justify-center gap-1.5"
              >
                <FolderOpen size={12} /> Open
              </button>
            </div>
            {/* Row 2: full-width path field */}
            <input
              type="text"
              value={outputDir}
              onChange={e => { setOutputDir(e.target.value); setOutputDirSaved(false) }}
              onKeyDown={e => e.key === 'Enter' && handleSaveOutputDir()}
              placeholder={DEFAULT_OUTPUT_DIR}
              className="w-full bg-card border border-border rounded-md px-3 py-2 text-xs text-white
                         placeholder-muted focus:outline-none focus:border-accent font-mono"
            />
          </section>

          {/* ── HuggingFace ── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-label mb-3 flex items-center gap-1.5">
              <LogIn size={13} /> HuggingFace
            </h3>
            {hfStatus && (
              <div className={`flex items-center gap-2 mb-3 text-xs px-3 py-2 rounded-md
                ${isLoggedIn ? 'bg-green-900/30 text-green-400 border border-green-800/40'
                             : 'bg-card border border-border text-muted'}`}>
                {isLoggedIn ? <><CheckCircle2 size={12} /> Logged in · {hfStatus}</> : <>{hfStatus}</>}
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

          {/* ── Models ── */}
          {modelChoices.length > 0 && (
            <section>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-label flex items-center gap-1.5">
                  <Download size={13} /> Models
                </h3>
                <button
                  onClick={handleCheckUpdates}
                  disabled={checkingUpdates}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-card border border-border
                             text-muted hover:text-white text-[10px] transition-colors disabled:opacity-40"
                >
                  {checkingUpdates
                    ? <RefreshCw size={11} className="animate-spin" />
                    : <CloudDownload size={11} />
                  }
                  {checkingUpdates ? 'Checking…' : 'Check for updates'}
                </button>
              </div>

              <div className="space-y-2">
                {modelChoices.map(m => {
                  const isLocal      = availableModels.includes(m)
                  const sizeLabel    = storageByChoice[m]
                  const isDeleting   = deletingModel === m
                  const needsConfirm = deleteConfirm === m
                  const upd          = updateByChoice[m]
                  const isUpdating   = updatingModel === m

                  return (
                    <div key={m} className={`rounded-md border text-xs px-3 py-2.5 transition-colors
                      ${isLocal ? 'bg-card border-border' : 'bg-bg border-border/50 opacity-60'}`}>

                      {/* Row 1: name + cached/not badge */}
                      <div className="flex items-start justify-between gap-2">
                        <span className={`leading-snug break-words ${isLocal ? 'text-white' : 'text-muted'}`}
                          style={{ maxWidth: '190px' }}>
                          {m}
                        </span>
                        <span className={`shrink-0 flex items-center gap-1 mt-0.5
                          ${isLocal ? 'text-green-400' : 'text-muted/50'}`}>
                          {isLocal
                            ? <><CheckCircle2 size={11} /> cached</>
                            : <><Download size={11} /> not downloaded</>
                          }
                        </span>
                      </div>

                      {/* Row 2: update status (when checked) */}
                      {upd && (
                        <div className="flex items-center justify-between mt-1.5">
                          <span
                            aria-label={UPDATE_LABEL[upd.status]?.ariaLabel}
                            className={`text-[10px] ${UPDATE_LABEL[upd.status]?.color ?? 'text-muted'}`}
                          >
                            {UPDATE_LABEL[upd.status]?.text}
                            {upd.local_hash && upd.online_hash && upd.local_hash !== upd.online_hash && (
                              <span className="opacity-60 ml-1">
                                (local {upd.local_hash} → {upd.online_hash})
                              </span>
                            )}
                          </span>
                          {upd.status === 'update_available' && (
                            <button
                              onClick={() => handleUpdateModel(m)}
                              disabled={isUpdating}
                              className="flex items-center gap-1 px-2 py-0.5 rounded bg-accent text-white
                                         hover:bg-accent/80 text-[10px] font-medium transition-colors disabled:opacity-40"
                            >
                              {isUpdating
                                ? <RefreshCw size={10} className="animate-spin" />
                                : <ArrowDownCircle size={10} />
                              }
                              {isUpdating ? 'Updating…' : 'Update'}
                            </button>
                          )}
                        </div>
                      )}

                      {/* Row 3: size + delete (cached only) */}
                      {isLocal && (
                        <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/50">
                          <span className="text-muted">{sizeLabel ?? '…'}</span>
                          {needsConfirm ? (
                            <div className="flex items-center gap-1.5">
                              <span className="text-amber-400 flex items-center gap-1">
                                <AlertCircle size={11} /> Sure?
                              </span>
                              <button onClick={() => handleDeleteModel(m)} disabled={isDeleting}
                                className="px-2 py-0.5 rounded bg-red-600 text-white hover:bg-red-500 text-[10px] font-medium transition-colors">
                                Yes, delete
                              </button>
                              <button onClick={() => setDeleteConfirm(null)}
                                className="px-2 py-0.5 rounded bg-card border border-border text-muted hover:text-white text-[10px] transition-colors">
                                Cancel
                              </button>
                            </div>
                          ) : (
                            <button onClick={() => handleDeleteModel(m)} disabled={isDeleting}
                              title={`Delete ${m} from disk`}
                              className="flex items-center gap-1 text-muted hover:text-red-400 transition-colors disabled:opacity-40">
                              {isDeleting ? <RefreshCw size={12} className="animate-spin" /> : <Trash2 size={12} />}
                              <span>{isDeleting ? 'Deleting…' : 'Delete'}</span>
                            </button>
                          )}
                        </div>
                      )}
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

          {/* ── Other Models (upscale) ── */}
          {extras && extras.upscale_models.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-label mb-3 flex items-center gap-1.5">
                <HardDrive size={13} /> Other Models
              </h3>
              <div className="space-y-2">

                {/* Upscale models */}
                {extras.upscale_models.map(m => (
                  <div key={m.name} className="rounded-md border bg-card border-border text-xs px-3 py-2.5">
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-white leading-snug break-words" style={{ maxWidth: '190px' }}>{m.name}</span>
                      <span className="shrink-0 flex items-center gap-1 mt-0.5 text-green-400">
                        <CheckCircle2 size={11} /> upscaler
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/50">
                      <span className="text-muted">{m.size}</span>
                      <button onClick={() => handleDeleteUpscale(m.name)} disabled={deletingUpscale === m.name}
                        title={`Delete ${m.name} from disk`}
                        className="flex items-center gap-1 text-muted hover:text-red-400 transition-colors disabled:opacity-40">
                        {deletingUpscale === m.name ? <RefreshCw size={12} className="animate-spin" /> : <Trash2 size={12} />}
                        <span>{deletingUpscale === m.name ? 'Deleting…' : 'Delete'}</span>
                      </button>
                    </div>
                  </div>
                ))}

              </div>
            </section>
          )}

          {/* ── Theme Colors ── */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-label flex items-center gap-1.5">
                <Palette size={13} /> Theme Colors
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={handleResetTheme}
                  className="px-2.5 py-1 rounded-md bg-card border border-border text-muted hover:text-white text-[10px] transition-colors"
                >
                  Reset
                </button>
                <button
                  onClick={handleSaveTheme}
                  disabled={themeSaving}
                  className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors flex items-center gap-1 disabled:opacity-40
                    ${themeSaved
                      ? 'bg-green-700/60 text-green-300 border border-green-700/50'
                      : 'bg-accent text-white hover:bg-accent/80'}`}
                >
                  {themeSaving ? <RefreshCw size={10} className="animate-spin" /> : themeSaved ? <CheckCircle2 size={10} /> : <Save size={10} />}
                  {themeSaved ? 'Saved' : 'Save'}
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(THEME_LABELS).map(([key, label]) => (
                <div key={key} className="flex items-center gap-2 bg-card border border-border rounded-md px-3 py-2">
                  <input
                    type="color"
                    value={themeColors[key] ?? DEFAULT_THEME[key]}
                    onChange={e => {
                      const next = { ...themeColors, [key]: e.target.value }
                      setThemeColors(next)
                      applyThemeColors(next)
                    }}
                    className="w-7 h-7 rounded cursor-pointer border-0 bg-transparent p-0"
                    title={label}
                    aria-label={label}
                  />
                  <div>
                    <div className="text-xs text-white leading-none">{label}</div>
                    <div className="text-[10px] text-muted font-mono mt-0.5">{themeColors[key] ?? DEFAULT_THEME[key]}</div>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-muted/60 mt-2">Colors preview live — click Save to persist.</p>
          </section>

          {/* ── Storage ── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-label mb-3 flex items-center gap-1.5">
              <HardDrive size={13} /> Storage
            </h3>
            {storage ? (
              <div className="space-y-2">
                {storage.models.map((m) => (
                  <div key={m.choice || m.name} className="flex justify-between text-xs">
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
