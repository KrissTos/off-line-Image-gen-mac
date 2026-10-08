import { LogIn, LogOut, CheckCircle2 } from 'lucide-react'

interface Props {
  token:      string
  onToken:    (v: string) => void
  status:     string | null
  loggedIn:   boolean
  message:    string
  onLogin:    () => void
  onLogout:   () => void
}

/** HuggingFace token login row (inside the Accounts & API keys card). */
export default function HuggingFaceLogin({ token, onToken, status, loggedIn, message, onLogin, onLogout }: Props) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted/80">
        <LogIn size={11} /> HuggingFace
        {loggedIn && status && (
          <span className="flex items-center gap-1 text-green-400 normal-case font-normal">
            <CheckCircle2 size={11} /> {status}
          </span>
        )}
        {!loggedIn && status && <span className="normal-case font-normal text-muted/70">{status}</span>}
      </div>
      <div className="flex gap-2">
        <input
          type="password" value={token} onChange={e => onToken(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && onLogin()}
          placeholder="hf_…  token"
          aria-label="HuggingFace token"
          className="flex-1 min-w-0 bg-bg border border-border rounded-md px-2 py-1 text-xs text-white placeholder-muted focus:outline-none focus:border-accent"
        />
        <button onClick={onLogin} className="px-2 py-1 rounded-md bg-accent text-white text-[10px] hover:bg-accent/80">
          {loggedIn ? 'Re-login' : 'Login'}
        </button>
        <button onClick={onLogout}
          className="px-2 py-1 rounded-md bg-card border border-border text-muted hover:text-white text-[10px] flex items-center gap-1">
          <LogOut size={11} /> Logout
        </button>
      </div>
      <p className="text-[10px] text-muted/60">Needed for gated models and to auto-download them.</p>
      {message && <p className="text-[10px] text-muted">{message}</p>}
    </div>
  )
}
