"""CivitAI downloads: API key, installed-file registry, safe download into lora_uploads/, delete.

The key is only ever sent to civitai.com (never to the storage host it redirects to) and never
appears in any returned or logged text. Files land in lora_uploads/ so the LoRA panel sees them.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
CIVITAI_HOST = "civitai.com"
MAX_REDIRECTS = 5
CHUNK = 1 << 20
PARTIAL_MAX_AGE_S = 3600
_ACTIVE = ("queued", "downloading", "verifying")
log = logging.getLogger(__name__)


class DownloadError(Exception):
    """Message is user-facing (shown on the row)."""


def _key_path() -> Path:
    return BASE_DIR / "civitai" / "token"


def _registry_path() -> Path:
    return BASE_DIR / "civitai_installed.json"


def _lora_dir() -> Path:
    return BASE_DIR / "lora_uploads"


# ── API key ──────────────────────────────────────────────────────────────────

def get_key() -> str | None:
    try:
        return _key_path().read_text().strip() or None
    except OSError:
        return None


def set_key(key: str) -> None:
    key = (key or "").strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Invalid API key")
    p = _key_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key)
    os.chmod(p, 0o600)


def clear_key() -> None:
    _key_path().unlink(missing_ok=True)


# ── registry of downloaded files ─────────────────────────────────────────────

def load_registry() -> dict:
    try:
        data = json.loads(_registry_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


_reg_lock = threading.RLock()       # guards every registry read-modify-write (download, delete)


def _save_registry(reg: dict) -> None:
    p = _registry_path()
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    os.replace(tmp, p)


def installed_map() -> dict:
    """{(modelId, family): {**entry, "file": name}} for every downloaded file."""
    return {(e.get("modelId"), e.get("family")): {**e, "file": name} for name, e in load_registry().items()}


def trained_trigger(file_name: str) -> str | None:
    words = [str(w).strip() for w in (load_registry().get(file_name) or {}).get("trained", []) if str(w).strip()]
    return ", ".join(words) or None


# ── names ────────────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """A plain `.safetensors` file name from an untrusted string; raises ValueError otherwise."""
    base = Path(str(name or "").replace("\\", "/")).name
    base = re.sub(r"[^\w.\- ()\[\]]", "_", base).strip(" .")
    ext = ".safetensors"
    if not base.lower().endswith(ext) or len(base) <= len(ext):
        raise ValueError("Not a .safetensors file name")
    if len(base) > 120:
        base = base[:120 - len(ext)] + ext
    return base[:-len(ext)] + ext                  # /api/lora/list matches the suffix case-sensitively


def _dest_name(name: str, c: dict, fam: str, reg: dict, lora_dir: Path) -> str:
    own = reg.get(name)
    if not (lora_dir / name).exists() or (own and own.get("modelId") == c["modelId"] and own.get("family") == fam):
        return name
    return f"{name[:-len('.safetensors')]}__civ{c['versionId']}.safetensors"


# ── network ──────────────────────────────────────────────────────────────────

def explain_status(status: int, has_key: bool) -> str:
    if status == 401:
        return "Key rejected or model needs login" if has_key else "Needs a CivitAI API key"
    if status == 403:
        return "Not available (early access or removed)"
    if status == 404:
        return "Version removed"
    if status == 429:
        return "Rate limited, retry later"
    return f"CivitAI returned HTTP {status}"


def open_stream(url: str, key: str | None, get=None):
    """GET `url` following redirects by hand: the key goes to civitai.com only, redirects must be https."""
    if get is None:
        import requests
        get = requests.get
    for _ in range(MAX_REDIRECTS + 1):
        p = urlparse(url)
        if p.scheme != "https":
            raise DownloadError("Refusing a non-https download")
        headers = {"Authorization": f"Bearer {key}"} if key and p.hostname == CIVITAI_HOST else {}
        r = get(url, headers=headers, stream=True, allow_redirects=False, timeout=(10, 60))
        if r.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, r.headers.get("Location", ""))
            r.close()
            continue
        return r
    raise DownloadError("Too many redirects")


def verify_lora(path: str, family: str) -> None:
    """Header-only check that the file is a LoRA for `family`; raises RuntimeError with a user message."""
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception as e:
        raise RuntimeError(f"Could not read LoRA file: {e}")
    if not keys:
        raise RuntimeError("LoRA file appears to be empty.")
    if family in ("klein-9B", "klein-4B"):
        from core.lora_flux2 import check_lora_compatibility, lora_variant
        check_lora_compatibility(path)
        want = family.split("-")[1].lower()
        got = lora_variant(path)
        if got and got != want:
            raise RuntimeError(f"This LoRA is for klein-{got.upper()}, not klein-{want.upper()}")


# ── download job ─────────────────────────────────────────────────────────────

_jobs: dict[int, dict] = {}
_lock = threading.Lock()


def _set(vid: int, **kw) -> None:
    with _lock:
        _jobs.setdefault(vid, {}).update(kw)


def job_state(version_id: int) -> dict:
    with _lock:
        return dict(_jobs.get(version_id, {"state": "idle"}))


def _fail(vid: int, msg: str, key: str | None) -> None:
    _set(vid, state="error", error=msg.replace(key, "***") if key else msg)


def _sweep_partials(lora_dir: Path) -> None:
    for p in lora_dir.glob(".tmp_*.part"):
        try:
            if time.time() - p.stat().st_mtime > PARTIAL_MAX_AGE_S:
                p.unlink()
        except OSError:
            pass


def start_download(row: dict, *, open_stream=None, verify=None, loaded_paths=None, threaded: bool = True) -> dict:
    vid = int(row["civitai"]["versionId"])
    with _lock:
        cur = _jobs.get(vid)
        if cur and cur.get("state") in _ACTIVE:
            return dict(cur)
        _jobs[vid] = {"state": "queued", "bytes": 0, "total": 0, "error": None}
    args = (vid, row, open_stream or globals()["open_stream"], verify or verify_lora, loaded_paths or (lambda: []))
    if threaded:
        threading.Thread(target=_run, args=args, daemon=True).start()
    else:
        _run(*args)
    return job_state(vid)


def _run(vid: int, row: dict, opener, verify, loaded_paths) -> None:
    c, fam = row["civitai"], row.get("family", "")
    key = get_key()
    tmp = None
    try:
        name = safe_filename(c["file"])
        lora_dir = _lora_dir()
        lora_dir.mkdir(exist_ok=True)
        dest_name = _dest_name(name, c, fam, load_registry(), lora_dir)
        _sweep_partials(lora_dir)
        tmp = lora_dir / f".tmp_{dest_name}.part"
        resp = opener(f"https://{CIVITAI_HOST}/api/download/models/{vid}", key)
        try:
            if resp.status_code != 200:
                raise DownloadError(explain_status(resp.status_code, bool(key)))
            total = int(resp.headers.get("Content-Length") or 0) or int((c.get("sizeKB") or 0) * 1024)
            _set(vid, state="downloading", bytes=0, total=total)
            h, n = hashlib.sha256(), 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(CHUNK):
                    if chunk:
                        fh.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
                        _set(vid, bytes=n)
        finally:
            resp.close()
        _set(vid, state="verifying")
        want = (c.get("sha256") or "").lower()
        if want and h.hexdigest() != want:
            raise DownloadError("Checksum mismatch, file discarded")
        try:
            verify(str(tmp), fam)
        except RuntimeError as e:
            raise DownloadError(str(e))
        with _reg_lock:
            reg = load_registry()
            olds = [o for o, e in reg.items()                  # a newer version replaces the old file
                    if e.get("modelId") == c["modelId"] and e.get("family") == fam and o != dest_name]
            in_use = {os.path.basename(p) for p in loaded_paths()}
            busy = next((o for o in olds if o in in_use), None)
            if busy:
                raise DownloadError(f"{busy} is loaded: unload it first, then update")
            os.replace(tmp, lora_dir / dest_name)
            tmp = None
            for o in olds:
                (lora_dir / Path(o).name).unlink(missing_ok=True)
                reg.pop(o)
            reg[dest_name] = {"modelId": c["modelId"], "versionId": vid, "family": fam, "sha256": want,
                              "name": row.get("name", ""), "trained": list(c.get("trained") or []),
                              "installedAt": int(time.time())}
            _save_registry(reg)
        _set(vid, state="done", file=dest_name)
    except DownloadError as e:
        _fail(vid, str(e), key)
    except OSError as e:
        _fail(vid, f"Disk error: {e.strerror or e}", key)
    except Exception as e:
        log.warning("civitai download %s failed: %s", vid, str(e).replace(key, "***") if key else e)
        _fail(vid, "Download failed (network error)", key)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


# ── delete ───────────────────────────────────────────────────────────────────

def delete_installed(version_id: int, loaded_paths=()) -> str:
    """Delete a downloaded file (registry files only). LookupError if unknown, RuntimeError if loaded."""
    with _reg_lock:
        reg = load_registry()
        name = next((n for n, e in reg.items() if e.get("versionId") == version_id), None)
        if name is None or Path(name).name != name:
            raise LookupError("Not an installed CivitAI LoRA")
        if any(os.path.basename(p) == name for p in loaded_paths):
            raise RuntimeError("This LoRA is loaded: unload it first")
        (_lora_dir() / name).unlink(missing_ok=True)
        reg.pop(name)
        _save_registry(reg)
    return name
