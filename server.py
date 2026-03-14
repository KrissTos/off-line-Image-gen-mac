"""
FastAPI backend for ultra-fast-image-gen.

Serves the React SPA from frontend/dist/ and exposes /api/* endpoints.

Usage:
    python server.py              # production (port 7860)
    python server.py --port 7861  # dev mode (Vite proxy points here)
    python server.py --reload     # uvicorn auto-reload for development
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

# ── Suppress semaphore-leak warning ───────────────────────────────────────────
# The warning is emitted by Python's multiprocessing.resource_tracker *daemon*
# — a separate child process that does NOT inherit warnings.filterwarnings().
# Setting PYTHONWARNINGS in os.environ before that daemon is spawned (which
# happens on the first torch/multiprocessing semaphore creation) is the only
# reliable way to silence it in the daemon's process.
_pw = os.environ.get("PYTHONWARNINGS", "")
if "resource_tracker" not in _pw:
    os.environ["PYTHONWARNINGS"] = (
        f"{_pw},ignore::UserWarning:multiprocessing.resource_tracker"
    ).lstrip(",")
del _pw
# Belt-and-suspenders: also suppress in the main process.
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be",
    category=UserWarning,
)

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent
DIST    = ROOT / "frontend" / "dist"
TEMP_DIR = ROOT / ".tmp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ultra-fast-image-gen API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React SPA (only if built)
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

# ── Single-line access logging ────────────────────────────────────────────────
# Uvicorn prints one INFO line per request, which makes the terminal scroll
# continuously.  This handler rewrites the *same* terminal line instead
# (using CR + ANSI "clear to EOL") when connected to a real TTY.
# Falls back to normal newline output when piped / redirected.

class _SingleLineHandler(logging.StreamHandler):
    """Overwrites the current terminal line for every log record."""

    _tty: bool = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if self._tty:
                # \r  → go to column 0
                # \033[K → erase from cursor to end of line
                sys.stderr.write(f"\r\033[K{msg}")
            else:
                sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


async def _startup_patch_access_log() -> None:
    """
    Replace uvicorn.access's default StreamHandler with our single-line variant.
    Runs on startup — AFTER uvicorn has already configured its own handlers.
    """
    alog = logging.getLogger("uvicorn.access")
    if alog.handlers:
        fmt = alog.handlers[0].formatter   # keep uvicorn's existing formatter
        h = _SingleLineHandler()
        h.setFormatter(fmt)
        alog.handlers = [h]
        alog.propagate = False


app.add_event_handler("startup", _startup_patch_access_log)

# ── Browser heartbeat (auto-shutdown when browser closes) ─────────────────────

_last_ping:          float = time.time()   # updated by POST /api/ping
_auto_shutdown:      bool  = True          # set to False via --no-auto-shutdown
_HEARTBEAT_TIMEOUT:  float = 15.0          # seconds of silence → shutdown
_HEARTBEAT_INTERVAL: float = 5.0          # check interval


async def _heartbeat_watcher() -> None:
    """Shut down the server if no browser ping arrives within the timeout."""
    # Give the browser time to open on first launch
    await asyncio.sleep(_HEARTBEAT_TIMEOUT + 5)
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        if time.time() - _last_ping > _HEARTBEAT_TIMEOUT:
            print("\n⏹  No browser heartbeat received — shutting down server.", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return


@app.on_event("startup")
async def _start_heartbeat() -> None:
    if _auto_shutdown:
        asyncio.create_task(_heartbeat_watcher())


# ── Lazy imports (avoid loading torch at import time) ─────────────────────────

def _app():
    import app as _a
    return _a

def _mgr():
    from pipeline import manager
    return manager

def _wu():
    from core.workflow_utils import load_any_workflow, get_locally_available_models
    return load_any_workflow, get_locally_available_models

from core.ip_adapter_flux import (
    is_downloaded        as ipa_is_downloaded,
    load_ip_adapter      as ipa_load,
    unload_ip_adapter    as ipa_unload,
    set_scale            as ipa_set_scale,
    download             as ipa_download,
)

_ipa_loaded: bool = False   # tracks whether IP-Adapter is injected into current pipe

# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:             str   = ""
    height:             int   = 512
    width:              int   = 512
    steps:              int   = 20
    seed:               int   = -1
    guidance:           float = 7.0
    device:             str   = "mps"
    model_choice:       str   = ""
    model_source:       str   = "Local"
    input_image_ids:    list[str] = Field(default_factory=list)  # temp file IDs
    mask_image_id:      str | None = None
    lora_file:          str | None = None
    lora_strength:      float = 1.0
    img_strength:       float = 1.0
    repeat_count:       int   = 1
    auto_save:          bool  = True
    output_dir:         str   = ""
    upscale_enabled:    bool  = False
    upscale_model_path: str   = ""
    num_frames:         int   = 25
    fps:                int   = 24
    mask_mode:          str   = "Crop & Composite (Fast)"
    outpaint_align:     str   = "center"
    ip_adapter_image_ids: list[str] = Field(default_factory=list)
    ip_adapter_scales:    list[float] = Field(default_factory=list)
    ip_adapter_enabled:   bool = False


class LoadModelRequest(BaseModel):
    model_choice: str
    device:       str = "mps"


class LoadLoraRequest(BaseModel):
    lora_path: str
    strength:  float = 1.0
    device:    str   = "mps"


class SaveWorkflowRequest(BaseModel):
    name:               str
    prompt:             str   = ""
    height:             int   = 512
    width:              int   = 512
    steps:              int   = 20
    seed:               int   = -1
    guidance:           float = 7.0
    device:             str   = "mps"
    model_choice:       str   = ""
    model_source:       str   = "Local"
    lora_strength:      float = 1.0
    img_strength:       float = 1.0
    repeat_count:       int   = 1
    upscale_enabled:    bool  = False
    upscale_model_path: str   = ""
    num_frames:         int   = 25
    fps:                int   = 24


class UpdateSettingsRequest(BaseModel):
    settings: dict[str, Any]


# ── Utility ───────────────────────────────────────────────────────────────────

def _temp_path(file_id: str) -> Path:
    return TEMP_DIR / file_id

def _load_pil(file_id: str):
    """Load a PIL Image from a temp-upload file ID."""
    from PIL import Image
    p = _temp_path(file_id)
    if not p.exists():
        raise HTTPException(404, f"Temp file {file_id} not found")
    return Image.open(p).convert("RGB")

def _output_dir() -> str:
    a = _app()
    default = getattr(a, "DEFAULT_OUTPUT_DIR",
                      os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen"))
    try:
        settings = a.load_settings()
        return settings.get("output_dir") or default
    except Exception:
        return default


# ── Routes: Heartbeat ─────────────────────────────────────────────────────────

@app.post("/api/ping")
async def api_ping() -> dict:
    """Browser heartbeat — resets the auto-shutdown timer."""
    global _last_ping
    _last_ping = time.time()
    return {"ok": True}


# ── Routes: Status / devices / models ────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    return _mgr().current_status()


@app.get("/api/devices")
async def api_devices():
    return {"devices": _app().get_available_devices()}


@app.get("/api/models")
async def api_models():
    from core.workflow_utils import get_locally_available_models
    a = _app()
    choices    = a.MODEL_CHOICES
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    available  = get_locally_available_models(models_dir)
    return {
        "choices":   choices,
        "available": available,
        "current":   getattr(a, "current_model", None),
    }


@app.post("/api/models/load")
async def api_load_model(req: LoadModelRequest):
    global _ipa_loaded
    status = await _mgr().load_model(req.model_choice, req.device)
    _ipa_loaded = False  # IP-Adapter must be re-injected into new pipeline
    return {"status": status}


@app.delete("/api/models/{model_name:path}")
async def api_delete_model(model_name: str):
    a = _app()
    # model_name is URL-encoded MODEL_CHOICES string
    try:
        result = a.delete_model(model_name)
        # delete_model returns a 3-tuple in Gradio; extract status message
        msg = result[2] if isinstance(result, (list, tuple)) and len(result) >= 3 else str(result)
        return {"status": msg}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/models/check-updates")
async def api_check_model_updates():
    """Query HuggingFace Hub for latest commit hashes of all known models."""
    a = _app()
    try:
        table, _ = await asyncio.get_event_loop().run_in_executor(None, a.check_online_versions)
        # Return per-model status list for the UI
        results = []
        for repo_id, display_name in a.KNOWN_MODELS.items():
            from app import online_version_cache, get_model_snapshot_info, get_local_models_dir
            local_hash, _ = get_model_snapshot_info(get_local_models_dir(), repo_id)
            online_hash   = online_version_cache.get(repo_id)
            if online_hash is None:
                status = "error"
            elif local_hash is None:
                status = "not_downloaded"
            elif local_hash == online_hash:
                status = "up_to_date"
            else:
                status = "update_available"
            results.append({
                "choice":       display_name,
                "repo_id":      repo_id,
                "local_hash":   local_hash[:8]  if local_hash  else None,
                "online_hash":  online_hash[:8] if online_hash else None,
                "status":       status,
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/models/update")
async def api_update_model(req: LoadModelRequest):
    """Download / update a model from HuggingFace Hub."""
    a = _app()
    try:
        msg, _, _ = await asyncio.get_event_loop().run_in_executor(
            None, a.download_model_update, req.model_choice
        )
        return {"status": msg}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/open-folder-dialog")
async def api_open_folder_dialog():
    """Open a native macOS folder picker dialog and return the selected path."""
    import platform
    if platform.system() != "Darwin":
        raise HTTPException(400, "Folder picker only supported on macOS")
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e",
            'POSIX path of (choose folder with prompt "Select output folder")',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            return {"path": None, "cancelled": True}
        if proc.returncode != 0:
            err = stderr.decode().strip()
            # "User cancelled" is normal — anything else is a real error worth surfacing
            if err and "cancelled" not in err.lower() and "cancel" not in err.lower():
                raise HTTPException(500, f"osascript error: {err}")
            return {"path": None, "cancelled": True}
        path = stdout.decode().strip().rstrip("/")
        return {"path": path, "cancelled": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/open-output-folder")
async def api_open_output_folder():
    """Reveal the current output folder in Finder (macOS) or file manager."""
    import subprocess, platform
    folder = os.path.expanduser(_output_dir())
    os.makedirs(folder, exist_ok=True)
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", folder])
        elif platform.system() == "Windows":
            subprocess.Popen(["explorer", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Routes: Generation (SSE) ─────────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """
    Server-Sent Events stream.  Each event is a JSON object:
      {"type": "progress", "message": "..."}
      {"type": "image", "url": "/api/output/...", "info": "..."}
      {"type": "video", "url": "/api/output/..."}
      {"type": "done"}
      {"type": "error", "message": "..."}
    """
    if _mgr().is_busy:
        raise HTTPException(423, "A generation is already running")

    # Resolve uploaded images → PIL
    from PIL import Image as PILImage
    input_images = None
    if req.input_image_ids:
        try:
            input_images = [_load_pil(fid) for fid in req.input_image_ids]
        except HTTPException:
            raise

    mask_image = None
    if req.mask_image_id:
        try:
            p = _temp_path(req.mask_image_id)
            mask_image = PILImage.open(p).convert("L")
        except Exception:
            pass

    # IP-Adapter reference images
    ip_adapter_images = None
    if req.ip_adapter_enabled and req.ip_adapter_image_ids:
        ip_adapter_images = [_load_pil(fid) for fid in req.ip_adapter_image_ids]
        global _ipa_loaded
        import app as _app_module
        if not _ipa_loaded and _app_module.pipe is not None:
            await asyncio.get_event_loop().run_in_executor(None, ipa_load, _app_module.pipe)
            _ipa_loaded = True

    params = req.model_dump()
    params["input_images"] = input_images
    params["mask_image"]   = mask_image
    params["output_dir"]   = req.output_dir or _output_dir()
    params["ip_adapter_images"]  = ip_adapter_images
    params["ip_adapter_scales"]  = req.ip_adapter_scales or [0.6] * len(req.ip_adapter_image_ids)
    params["ip_adapter_enabled"] = req.ip_adapter_enabled

    async def event_stream():
        try:
            async for event in _mgr().generate(params):
                # Save sidecar JSON alongside output file so Gallery can reload prompt/model
                if event.get("type") in ("image", "video"):
                    url = event.get("url", "")
                    if url.startswith("/api/output/"):
                        filename = url[len("/api/output/"):]
                        out_path  = Path(params["output_dir"]) / filename
                        sidecar   = out_path.with_suffix(".json")
                        try:
                            sidecar.write_text(json.dumps({
                                "prompt":       req.prompt,
                                "model_choice": req.model_choice,
                                "width":        req.width,
                                "height":       req.height,
                                "steps":        req.steps,
                                "guidance":     req.guidance,
                                "seed":         req.seed,
                            }, indent=2))
                        except Exception:
                            pass  # non-critical
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── Routes: File serving ──────────────────────────────────────────────────────

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Upload a reference or mask image.  Returns a temp file ID."""
    ext     = Path(file.filename or "upload.png").suffix or ".png"
    file_id = f"{uuid.uuid4().hex}{ext}"
    dest    = _temp_path(file_id)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"id": file_id, "url": f"/api/temp/{file_id}"}


@app.get("/api/temp/{file_id:path}")
async def api_serve_temp(file_id: str):
    p = _temp_path(file_id)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p))


@app.delete("/api/output/{filename:path}")
async def api_delete_output(filename: str):
    """Delete an output file and its sidecar JSON."""
    p = Path(_output_dir()) / filename
    if not p.exists():
        raise HTTPException(404, detail="File not found")
    p.unlink()
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        sidecar.unlink()
    return {"deleted": filename}


@app.get("/api/output/{filename:path}")
async def api_serve_output(filename: str):
    p = Path(_output_dir()) / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p))


def _read_sidecar(f: Path) -> dict:
    """Read .json sidecar saved alongside an output file (prompt, model, etc.)."""
    s = f.with_suffix(".json")
    if s.exists():
        try:
            return json.loads(s.read_text())
        except Exception:
            pass
    return {}


@app.get("/api/outputs")
async def api_list_outputs(limit: int = 20):
    """List most recent output images/videos."""
    out = Path(_output_dir())
    if not out.exists():
        return {"files": []}
    files = sorted(
        [f for f in out.iterdir() if f.suffix.lower() in {".png", ".jpg", ".mp4", ".webm"}],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:limit]
    result = []
    for f in files:
        meta = _read_sidecar(f)
        result.append({
            "name":         f.name,
            "url":          f"/api/output/{f.name}",
            "mtime":        f.stat().st_mtime,
            "kind":         "video" if f.suffix.lower() in {".mp4", ".webm", ".mov"} else "image",
            "prompt":       meta.get("prompt"),
            "model_choice": meta.get("model_choice"),
        })
    return {"files": result}


# ── Routes: Workflows ─────────────────────────────────────────────────────────

@app.get("/api/workflows")
async def api_list_workflows():
    a = _app()
    names = a.list_saved_workflows() or []
    return {"workflows": names}


@app.post("/api/workflows/save")
async def api_save_workflow(req: SaveWorkflowRequest):
    a    = _app()
    data = req.model_dump()
    name = data.pop("name")
    try:
        result = a.save_workflow(
            data.get("prompt", ""),
            data.get("height", 512),
            data.get("width", 512),
            data.get("steps", 20),
            data.get("seed", -1),
            data.get("guidance", 7.0),
            data.get("device", "mps"),
            data.get("model_choice", ""),
            data.get("model_source", "Local"),
            data.get("lora_strength", 1.0),
            data.get("img_strength", 1.0),
            data.get("repeat_count", 1),
            data.get("upscale_enabled", False),
            data.get("upscale_model_path", ""),
            data.get("num_frames", 25),
            data.get("fps", 24),
            None,    # input_images not serialised here
            name,
        )
        status_msg = result[0] if isinstance(result, (tuple, list)) else str(result)
        return {"status": status_msg}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/workflows/{name:path}")
async def api_load_workflow(name: str):
    a = _app()
    try:
        result = a.load_workflow(name)
        # result is an 18-tuple matching _wf_load_outputs; last element is status
        keys = ["prompt", "height", "width", "steps", "seed", "guidance",
                "device", "model_choice", "model_source", "lora_strength",
                "img_strength", "repeat_count", "upscale_enabled",
                "upscale_model_path", "num_frames", "fps", "input_images", "status"]
        return dict(zip(keys, result))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/workflows/import")
async def api_import_comfyui(file: UploadFile = File(...)):
    """Import a ComfyUI workflow JSON; returns parsed params."""
    # Write to temp, parse, delete
    tmp = Path(tempfile.mktemp(suffix=".json"))
    try:
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        load_any_workflow, get_locally_available_models = _wu()
        wf = load_any_workflow(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    if wf.get("_source") != "comfyui":
        raise HTTPException(400, "File is a native workflow — use /api/workflows/{name} instead")

    from core.workflow_utils import get_locally_available_models as glam
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    available  = glam(models_dir)

    matched = wf.get("model_choice")
    ckpt    = os.path.basename(wf.get("_comfyui_ckpt_name") or "") or "(unknown)"
    if matched and available and matched not in available:
        matched = available[0]
    elif not matched:
        matched = available[0] if available else (_app().MODEL_CHOICES[0])

    return {
        "prompt":             wf.get("prompt", ""),
        "height":             wf.get("height", 512),
        "width":              wf.get("width",  512),
        "steps":              wf.get("steps",  20),
        "seed":               wf.get("seed",   -1),
        "guidance":           wf.get("guidance", 7.0),
        "device":             wf.get("device", "mps"),
        "model_choice":       matched,
        "model_source":       wf.get("model_source", "Local"),
        "lora_strength":      wf.get("lora_strength", 1.0),
        "img_strength":       wf.get("img_strength", 1.0),
        "repeat_count":       wf.get("repeat_count", 1),
        "upscale_enabled":    wf.get("upscale_enabled", False),
        "upscale_model_path": wf.get("upscale_model_path", ""),
        "num_frames":         wf.get("num_frames", 25),
        "fps":                wf.get("fps", 24),
        "checkpoint_name":    ckpt,
        "unknown_nodes":      wf.get("_unknown_nodes", []),
        "lora_file":          wf.get("lora_file", ""),
    }


# ── Routes: LoRA ──────────────────────────────────────────────────────────────

@app.post("/api/lora/load")
async def api_load_lora(req: LoadLoraRequest):
    status = await _mgr().load_lora(req.lora_path, req.strength, req.device)
    return {"status": status}


@app.delete("/api/lora")
async def api_clear_lora():
    return {"status": _mgr().clear_lora()}


@app.post("/api/lora/upload")
async def api_upload_lora(file: UploadFile = File(...)):
    """Upload a LoRA .safetensors file and return its saved path."""
    fname = Path(file.filename or "lora.safetensors").name
    dest  = ROOT / "lora_uploads" / fname
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"path": str(dest), "name": fname}


# ── IP-Adapter ──────────────────────────────────────────────────────────────

@app.get("/api/ip-adapter/status")
async def ipa_status():
    return {
        "downloaded": ipa_is_downloaded(),
        "loaded":     _ipa_loaded,
    }


@app.post("/api/ip-adapter/download")
async def ipa_download_endpoint():
    """SSE stream of download progress. Same pattern as batch upscale."""
    if ipa_is_downloaded():
        async def already_done():
            yield f"data: {json.dumps({'type':'done','message':'Already downloaded'})}\n\n"
        return StreamingResponse(already_done(), media_type="text/event-stream")

    async def _stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _progress(downloaded: int, total: int):
            pct = int(downloaded / total * 100) if total else 0
            dl_mb  = downloaded / 1024 / 1024
            tot_mb = total      / 1024 / 1024
            evt = {"type": "progress", "downloaded": downloaded, "total": total,
                   "pct": pct, "message": f"Downloading… {dl_mb:.0f} / {tot_mb:.0f} MB ({pct}%)"}
            loop.call_soon_threadsafe(queue.put_nowait, evt)

        def _run():
            try:
                ipa_download(progress_cb=_progress)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_run)
        executor.shutdown(wait=False)

        while True:
            evt = await queue.get()
            yield f"data: {json.dumps(evt)}\n\n"
            if evt["type"] in ("done", "error"):
                break

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.delete("/api/ip-adapter")
async def ipa_delete():
    """Unload from pipeline and delete weights."""
    global _ipa_loaded
    import app as _app_module
    from core.ip_adapter_flux import IP_ADAPTER_FILE
    if _ipa_loaded and _app_module.pipe is not None:
        await asyncio.get_event_loop().run_in_executor(None, ipa_unload, _app_module.pipe)
        _ipa_loaded = False
    if IP_ADAPTER_FILE.exists():
        IP_ADAPTER_FILE.unlink()
    return {"status": "deleted"}


@app.post("/api/upscale/upload")
async def api_upload_upscale(file: UploadFile = File(...)):
    """Upload an upscale model (.pth/.safetensors/etc.) and return its saved path."""
    fname = Path(file.filename or "upscale_model.pth").name
    dest  = ROOT / "upscale_models" / fname
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"path": str(dest), "name": fname}


class BatchUpscaleRequest(BaseModel):
    input_folder:  str
    output_folder: str = ""
    scale_choice:  str = "×4"
    model_path:    str


@app.post("/api/upscale/batch")
async def api_batch_upscale(req: BatchUpscaleRequest):
    """Stream batch upscale progress as SSE log events."""

    def event_stream():
        try:
            for log_line in _app().batch_upscale_folder(
                req.input_folder,
                req.output_folder,
                req.scale_choice,
                req.model_path,
            ):
                # batch_upscale_folder yields cumulative log strings;
                # send only the last appended line for efficiency
                last = log_line.rstrip("\n").split("\n")[-1]
                yield f"data: {json.dumps({'type': 'log', 'message': last})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Routes: Settings ──────────────────────────────────────────────────────────

@app.get("/api/settings")
async def api_get_settings():
    return _app().load_settings()


@app.post("/api/settings")
async def api_update_settings(req: UpdateSettingsRequest):
    a = _app()
    for key, val in req.settings.items():
        a.save_setting(key, val)
    return {"status": "ok", "settings": a.load_settings()}


# ── Routes: Storage ───────────────────────────────────────────────────────────

@app.get("/api/storage")
async def api_storage():
    a = _app()
    models, summary = a.scan_downloaded_models()
    normalized = [
        {"choice": m.get("display_name", ""), "name": m.get("cache_name", ""), "size": m.get("size_str", "")}
        for m in models
    ]
    return {"models": normalized, "summary": summary}


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@app.get("/api/models/extras")
async def api_models_extras():
    """Return info about non-HF-cache models: upscale files and IP-Adapter weights."""
    from core.ip_adapter_flux import IP_ADAPTER_FILE

    upscale_exts = {".pth", ".safetensors", ".ckpt", ".pt", ".bin"}
    upscale_dir  = ROOT / "upscale_models"
    upscale_list = []
    if upscale_dir.exists():
        for f in sorted(upscale_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in upscale_exts:
                upscale_list.append({"name": f.name, "size": _fmt_bytes(f.stat().st_size)})

    ipa: dict = {"downloaded": IP_ADAPTER_FILE.exists(), "size": None}
    if IP_ADAPTER_FILE.exists():
        ipa["size"] = _fmt_bytes(IP_ADAPTER_FILE.stat().st_size)

    return {"upscale_models": upscale_list, "ip_adapter": ipa}


@app.delete("/api/upscale/{filename:path}")
async def api_delete_upscale(filename: str):
    """Delete an upscale model file from disk."""
    target = (ROOT / "upscale_models" / Path(filename).name).resolve()
    # Safety: must stay inside upscale_models/
    if not str(target).startswith(str((ROOT / "upscale_models").resolve())):
        raise HTTPException(400, "Invalid path")
    if not target.exists():
        raise HTTPException(404, "File not found")
    target.unlink()
    return {"status": "deleted"}


# ── Routes: HuggingFace auth ──────────────────────────────────────────────────

class HFLoginRequest(BaseModel):
    token: str

@app.get("/api/hf/status")
async def api_hf_status():
    return {"status": _app().hf_get_status()}

@app.post("/api/hf/login")
async def api_hf_login(req: HFLoginRequest):
    return {"status": _app().hf_login_token(req.token)}

@app.post("/api/hf/logout")
async def api_hf_logout():
    return {"status": _app().hf_logout()}


# ── Routes: SPA (MUST be last — wildcard would shadow /api/* if registered earlier) ──

@app.get("/", include_in_schema=False)
async def serve_spa():
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Frontend not built. Run: cd frontend && npm run build"},
                        status_code=503)


@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """Catch-all: return index.html for any non-API route (React Router support)."""
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(404)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ultra-fast-image-gen FastAPI server")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   type=str, default="127.0.0.1")
    parser.add_argument("--reload", action="store_true",
                        help="Enable uvicorn auto-reload (dev mode)")
    parser.add_argument("--no-auto-shutdown", action="store_true",
                        help="Keep server running even when the browser is closed")
    args = parser.parse_args()

    if args.no_auto_shutdown:
        _auto_shutdown = False  # type: ignore[assignment]

    print(f"Starting ultra-fast-image-gen API on http://{args.host}:{args.port}")
    if DIST.exists():
        print(f"Serving React SPA from {DIST}")
    else:
        print("⚠  frontend/dist not found — run: cd frontend && npm run build")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
