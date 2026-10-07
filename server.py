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
import re
import shutil
import signal
import sys
import tempfile
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from core import civitai, civitai_install, model_sources, run_store

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
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent
DIST    = ROOT / "frontend" / "dist"
TEMP_DIR = ROOT / ".tmp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# ── Model Sources Registry ─────────────────────────────────────────────────
MODEL_SOURCES_FILE = ROOT / "model_sources.json"

DEFAULT_SOURCES: list[dict] = [
    # Base models — model_choice must exactly match MODEL_CHOICES in app.py
    {"id": "src-001", "name": "FLUX.2-klein-4B (4bit SDNQ)",  "model_choice": "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)",              "url": "https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",           "type": "base",     "vram_gb": 8,  "description": "~8 GB VRAM · MPS optimized · recommended for 16 GB Mac"},
    {"id": "src-002", "name": "FLUX.2-klein-9B (4bit SDNQ)",  "model_choice": "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)",        "url": "https://huggingface.co/Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",   "type": "base",     "vram_gb": 12, "description": "~12 GB VRAM · high quality"},
    {"id": "src-003", "name": "FLUX.2-klein-4B (Int8)",        "model_choice": "FLUX.2-klein-4B (Int8)",                              "url": "https://huggingface.co/aydin99/FLUX.2-klein-4B-int8",                        "type": "base",     "vram_gb": 16, "description": "~16 GB VRAM · MPS explicit · 4B int8"},
    {"id": "src-005", "name": "Z-Image Turbo (Full)",           "model_choice": "Z-Image Turbo (Full - LoRA support)",                "url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",                           "type": "base",     "vram_gb": 24, "description": "~24 GB VRAM · LoRA support"},
    {"id": "src-006", "name": "Z-Image Turbo (Quantized)",      "model_choice": "Z-Image Turbo (Quantized - Fast)",                   "url": "https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",            "type": "base",     "vram_gb": 6,  "description": "~6 GB VRAM · fast"},
    {"id": "src-007", "name": "LTX-Video",                      "model_choice": "LTX-Video  (txt2video · img2video with ref)",        "url": "https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled",            "type": "base",     "vram_gb": 26, "description": "LTX-Video 0.9.8 13B distilled — multiscale video gen"},
    # LoRAs
    {"id": "src-008", "name": "Outpaint LoRA (klein 4B)",       "url": "https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora",                  "type": "lora",     "description": "Outpainting — add green border to image"},
    {"id": "src-009", "name": "Zoom LoRA (klein 4B)",            "url": "https://huggingface.co/fal/flux-2-klein-4B-zoom-lora",                     "type": "lora",     "description": "Zoom into red-highlighted region"},
    {"id": "src-010", "name": "Spritesheet LoRA (klein 4B)",     "url": "https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora",              "type": "lora",     "description": "Single object → 2×2 sprite sheet"},
    {"id": "src-011", "name": "Virtual Try-on (klein 9B)",       "url": "https://huggingface.co/fal/flux-klein-9b-virtual-tryon-lora",              "type": "lora",     "description": "Clothing swap with reference images"},
    {"id": "src-012", "name": "360 Outpaint (klein 4B)",         "url": "https://huggingface.co/nomadoor/flux-2-klein-4B-360-erp-outpaint-lora",    "type": "lora",     "description": "Equirectangular panorama outpainting"},
    {"id": "src-013", "name": "360 Outpaint (klein 9B)",         "url": "https://huggingface.co/nomadoor/flux-2-klein-9B-360-erp-outpaint-lora",    "type": "lora",     "description": "Equirectangular panorama outpainting 9B"},
    {"id": "src-014", "name": "Style Pack (klein 9B)",           "url": "https://huggingface.co/DeverStyle/Flux.2-Klein-Loras",                     "type": "lora",     "description": "Arcane, DMC, flat-vector styles"},
    {"id": "src-015", "name": "Anime→Real (klein)",              "url": "https://huggingface.co/WarmBloodAban/Flux2_Klein_Anything_to_Real_Characters", "type": "lora", "description": "Anime to photorealistic conversion"},
    {"id": "src-016", "name": "Relight (klein 9B)",              "url": "https://huggingface.co/linoyts/Flux2-Klein-Delight-LoRA",                  "type": "lora",     "description": "Remove and replace lighting"},
    {"id": "src-017", "name": "Consistency (klein 9B)",          "url": "https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency",                 "type": "lora",     "description": "Improve edit coherence"},
    {"id": "src-018", "name": "Enhanced Details (klein 9B)",     "url": "https://huggingface.co/dx8152/Flux2-Klein-9B-Enhanced-Details",            "type": "lora",     "description": "Realism and texture boost"},
    {"id": "src-019", "name": "Distillation LoRA (klein 9B)",   "url": "https://huggingface.co/vafipas663/flux2-klein-base-9b-distill-lora",        "type": "lora",     "description": "Better CFG handling + fine detail"},
    {"id": "src-020", "name": "AC Style (klein)",                "url": "https://huggingface.co/valiantcat/FLUX.2-klein-AC-Style-LORA",             "type": "lora",     "description": "Comics and cyber neon style"},
    {"id": "src-021", "name": "Unified Reward (klein 9B)",       "url": "https://huggingface.co/CodeGoat24/FLUX.2-klein-base-9B-UnifiedReward-Flex-lora", "type": "lora", "description": "Quality preference alignment"},
    # Upscalers
    {"id": "src-022", "name": "Real-ESRGAN x4",                  "url": "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged",                  "type": "upscaler", "description": "Safetensors · general purpose"},
    {"id": "src-023", "name": "4xNomosWebPhoto RealPLKSR",        "url": "https://huggingface.co/Phips/4xNomosWebPhoto_RealPLKSR",                  "type": "upscaler", "description": "Best for web / JPEG photos"},
    {"id": "src-024", "name": "4xNomosWebPhoto ATD",              "url": "https://huggingface.co/Phips/4xNomosWebPhoto_atd",                        "type": "upscaler", "description": "ATD architecture · web photos"},
    {"id": "src-025", "name": "4xRealWebPhoto v4 DRCT-L",         "url": "https://huggingface.co/Phips/4xRealWebPhoto_v4_drct-l",                   "type": "upscaler", "description": "Latest DRCT · real photos"},
    {"id": "src-026", "name": "4x-UltraSharp",                    "url": "https://huggingface.co/Kim2091/UltraSharp",                               "type": "upscaler", "description": "JPEG artifact recovery · detail"},
    {"id": "src-027", "name": "4x-Remacri",                       "url": "https://huggingface.co/OzzyGT/4xRemacri",                                 "type": "upscaler", "description": "Safetensors · general use"},
    {"id": "src-028", "name": "gyre upscalers (SwinIR + HAT)",    "url": "https://huggingface.co/halffried/gyre_upscalers",                         "type": "upscaler", "description": "SwinIR + HAT in safetensors format"},
    {"id": "src-029", "name": "SwinIR collection",                 "url": "https://huggingface.co/GraydientPlatformAPI/safetensor-upscalers",        "type": "upscaler", "description": "SwinIR-L and SwinIR-M safetensors"},
    {"id": "src-030", "name": "uwg upscaler collection",           "url": "https://huggingface.co/uwg/upscaler",                                    "type": "upscaler", "description": "Large multi-architecture collection"},
    {"id": "src-031", "name": "OpenModelDB",                       "url": "https://openmodeldb.info",                                               "type": "upscaler", "description": "Browse all community upscale models"},
]

# ── Server log capture ────────────────────────────────────────────────────────
# Tee stdout + stderr to logs/server.log so the Settings drawer can download it.
# Overwrites on each server start so the log always reflects the current session.

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE  = LOGS_DIR / "server.log"

_log_fh = open(LOG_FILE, "w", encoding="utf-8", buffering=1)  # 'w' = fresh each run


import re as _re
_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\[K|\r")


class _LogTee:
    """Tee writes to both the original stream and the shared log file (ANSI-stripped)."""
    def __init__(self, original: Any) -> None:
        self._orig = original

    def write(self, text: str) -> int:
        self._orig.write(text)
        clean = _ANSI_RE.sub("", text)
        if clean:
            _log_fh.write(clean)
            _log_fh.flush()
        return len(text)

    def flush(self) -> None:
        self._orig.flush()

    def isatty(self) -> bool:
        return getattr(self._orig, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._orig.fileno()


sys.stdout = _LogTee(sys.__stdout__)  # type: ignore[assignment]
sys.stderr = _LogTee(sys.__stderr__)  # type: ignore[assignment]

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

_last_ping:           float = time.time()   # updated by POST /api/ping
# --no-auto-shutdown → env var: uvicorn.run("server:app") re-imports this module,
# so a global set in __main__ would never reach the served app.
_auto_shutdown:       bool  = os.environ.get("IMAGEGEN_NO_AUTO_SHUTDOWN") != "1"
_HEARTBEAT_TIMEOUT:   float = 60.0          # seconds of silence → shutdown (60 s handles background-tab throttling)
_HEARTBEAT_INTERVAL:  float = 5.0           # check interval
_shutdown_task:       asyncio.Task | None = None  # pending graceful-shutdown countdown


async def _heartbeat_watcher() -> None:
    """Shut down the server if no browser ping arrives within the timeout.
    Never shuts down while a generation is actively running.
    Detects Mac sleep/wake by measuring wall-clock jumps: if the event loop
    was frozen for longer than the timeout (i.e. system sleep), reset the
    ping timer so a brief wake doesn't immediately trigger shutdown."""
    # Give the browser time to open on first launch
    await asyncio.sleep(_HEARTBEAT_TIMEOUT + 5)
    _last_tick = time.time()
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        now = time.time()
        elapsed_since_tick = now - _last_tick
        _last_tick = now
        # If the event loop was suspended for longer than the timeout, it's
        # almost certainly a system sleep/wake cycle — reset the ping clock.
        if elapsed_since_tick > _HEARTBEAT_TIMEOUT:
            global _last_ping
            _last_ping = now
            print(f"⏸  System sleep detected ({elapsed_since_tick:.0f}s gap) — heartbeat timer reset.", flush=True)
            continue
        if time.time() - _last_ping > _HEARTBEAT_TIMEOUT:
            # Don't kill the server mid-generation — the browser may just be throttled
            try:
                from pipeline import manager
                if manager.is_busy:
                    _last_ping_ref = time.time()  # reset so we don't re-check immediately
                    continue
            except Exception:
                pass
            print("\n⏹  No browser heartbeat received — shutting down server.", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return


@app.on_event("startup")
async def _start_heartbeat() -> None:
    if _auto_shutdown:
        asyncio.create_task(_heartbeat_watcher())


def _prune_temp_dir(max_age_s: float = 86400, now: float | None = None) -> int:
    """Trash temp uploads older than max_age_s. Direct child files only."""
    now = time.time() if now is None else now
    n = 0
    for f in TEMP_DIR.iterdir():
        if f.is_file() and now - f.stat().st_mtime > max_age_s:
            try:
                run_store.trash(f)
                n += 1
            except Exception as e:   # one stuck file must not stop the rest
                print(f"[tmp_uploads] could not trash {f.name}: {e}")
    return n


@app.on_event("startup")
async def _prune_stale_uploads() -> None:
    # One `trash` process per file: run it in a thread, in the background, so startup never waits.
    task = asyncio.create_task(asyncio.to_thread(_prune_temp_dir))
    _RUN_TASKS.add(task)
    task.add_done_callback(_RUN_TASKS.discard)


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
    lora_files:         list[dict] = Field(default_factory=list)  # [{path, strength}, ...]
    lora_file:          str | None = None   # legacy single-LoRA (backward compat)
    lora_strength:      float = 1.0         # legacy single-LoRA strength
    img_strength:       float = 1.0
    repeat_count:       int   = 1
    auto_save:          bool  = True
    output_dir:         str   = ""
    upscale_enabled:    bool  = False
    upscale_model_path: str   = ""
    num_frames:         int   = 25
    fps:                int   = 24
    fast_preview:       bool  = False   # LTX-Video: single-pass distilled render (skips upsampler)
    mask_mode:          str   = "Crop & Composite (Fast)"
    outpaint_align:     str   = "center"
    ref_slots:          list[dict] = Field(default_factory=list)  # [{imageId, maskId, strength}] — recorded in the run

class BatchGenerateRequest(GenerateRequest):
    input_folder: str = ""  # local path to folder of images

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
    lora_files:         list[dict] = []     # [{path, strength}, ...]
    lora_file:          str | None = None   # legacy single-LoRA
    lora_strength:      float = 1.0         # legacy single-LoRA strength
    img_strength:       float = 1.0
    repeat_count:       int   = 1
    upscale_enabled:    bool  = False
    upscale_model_path: str   = ""
    num_frames:         int   = 25
    fps:                int   = 24
    fast_preview:       bool  = False
    ref_slots:          list[dict] = []
    mask_mode:          str        = ""
    outpaint_align:     str        = ""
    overwrite:          str | None = None   # existing folder name → rewrite in place


class UpdateSettingsRequest(BaseModel):
    settings: dict[str, Any]


# ── Utility ───────────────────────────────────────────────────────────────────

def _temp_path(file_id: str) -> Path:
    return TEMP_DIR / file_id

def _load_pil(file_id: str):
    """Load a PIL Image from a temp-upload file ID (EXIF orientation applied, like the UI)."""
    from PIL import Image, ImageOps
    p = _temp_path(file_id)
    if not p.exists():
        raise HTTPException(404, f"Temp file {file_id} not found")
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")

def _output_dir() -> str:
    a = _app()
    default = getattr(a, "DEFAULT_OUTPUT_DIR",
                      os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen"))
    try:
        settings = a.load_settings()
        return settings.get("output_dir") or default
    except Exception:
        return default


_SEED_RE = re.compile(r"Seed:\s*(\d+)")


def _guarded_temp(file_id: str | None) -> Path | None:
    """Existing file inside TEMP_DIR for a client-supplied id, else None."""
    if not file_id:
        return None
    try:
        p = (TEMP_DIR / file_id).resolve()
    except OSError:
        return None
    return p if p.is_relative_to(TEMP_DIR.resolve()) and p.is_file() else None


def _request_slots(req) -> list[dict]:
    """Slots to record in the run: req.ref_slots when sent, else derived from input_image_ids
    (the /generate skill's shape): slot 1 gets the mask and img_strength, others strength 1.0."""
    if req.ref_slots:
        raw = [(s.get("imageId"), s.get("maskId"), s.get("strength", 1.0)) for s in req.ref_slots]
    else:
        raw = [(fid, req.mask_image_id if i == 0 else None, req.img_strength if i == 0 else 1.0)
               for i, fid in enumerate(req.input_image_ids)]
    slots = []
    for img_id, mask_id, strength in raw:
        img = _guarded_temp(img_id)
        if img is not None:
            slots.append({"image_path": img, "mask_path": _guarded_temp(mask_id), "strength": float(strength)})
    return slots


def _record_output(run_dir: Path, event: dict, requested_seed: int) -> None:
    """Rename a fresh output to the run naming rule, list it in workflow.json, point the event at it."""
    src = Path(event.get("path") or "")
    if not src.is_file():
        return
    m = _SEED_RE.search(event.get("info") or "")
    seed = int(m.group(1)) if m else (requested_seed if requested_seed >= 0 else None)
    dst = src
    try:
        target = run_store.output_filename(run_dir, seed, src.suffix.lower() or ".png")
        src.rename(target)
        dst = target
        run_store.add_output(run_dir, dst, event["type"], seed=seed)
    except Exception as e:
        print(f"[run_store] could not record {src.name}: {e}")
    if event.get("info"):
        event["info"] = event["info"].replace(str(src), str(dst))
    event["path"] = str(dst)
    event["url"]  = f"/api/output/{run_dir.name}/outputs/{dst.name}"


_RUN_TASKS: set[asyncio.Task] = set()   # generations still running, possibly past their SSE client


async def _run_events(run_dir: Path, params: dict, requested_seed: int):
    """Generate into run_dir/outputs, yielding events with run paths. The generation runs in its
    own task, so outputs finishing after the client disconnects (reload, Stop) are still recorded.
    A run that produced nothing (stopped, failed) is moved to the Trash once generation has ended."""
    params["output_dir"] = str(run_dir / "outputs")
    queue: asyncio.Queue = asyncio.Queue()

    async def drive():
        produced = 0
        try:
            async for event in _mgr().generate(params):
                if event.get("type") in ("image", "video"):
                    _record_output(run_dir, event, requested_seed)
                    produced += 1
                queue.put_nowait(event)
        except Exception as e:
            import traceback
            traceback.print_exc()   # full traceback goes to server.log via _LogTee
            queue.put_nowait(e)
        finally:
            if not produced:
                try:
                    run_store.trash(run_dir)
                except Exception as e:
                    print(f"[run_store] could not trash empty run {run_dir.name}: {e}")
            queue.put_nowait(None)

    task = asyncio.create_task(drive())
    _RUN_TASKS.add(task)
    task.add_done_callback(_RUN_TASKS.discard)
    while (event := await queue.get()) is not None:
        if isinstance(event, Exception):
            raise event
        yield event


# ── Routes: Heartbeat ─────────────────────────────────────────────────────────

@app.post("/api/ping")
async def api_ping() -> dict:
    """Browser heartbeat — resets the auto-shutdown timer and cancels any pending shutdown."""
    global _last_ping, _shutdown_task
    _last_ping = time.time()
    if _shutdown_task and not _shutdown_task.done():
        _shutdown_task.cancel()
        _shutdown_task = None
    return {"ok": True}


@app.post("/api/shutdown")
async def api_shutdown() -> dict:
    """Graceful shutdown — called by browser beforeunload via sendBeacon.
    Waits 4 s before killing: a page refresh will cancel it via the next /api/ping."""
    global _shutdown_task
    if not _auto_shutdown:
        return {"ok": False, "reason": "auto-shutdown disabled"}
    if _shutdown_task and not _shutdown_task.done():
        return {"ok": True, "reason": "shutdown already pending"}
    async def _delayed():
        await asyncio.sleep(4.0)
        print("\n⏹  Browser closed — shutting down server.", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
    _shutdown_task = asyncio.create_task(_delayed())
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
    models_dir = a.get_local_models_dir()
    available  = get_locally_available_models(models_dir)
    return {
        "choices":   choices,
        "available": available,
        "current":   getattr(a, "current_model", None),
    }


@app.post("/api/models/load")
async def api_load_model(req: LoadModelRequest):
    status = await _mgr().load_model(req.model_choice, req.device)
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
    """Download / update a model from HuggingFace Hub. Streams SSE download progress."""
    import queue as _q
    import threading as _threading

    a = _app()
    progress_q = _q.Queue()
    loop = asyncio.get_event_loop()

    async def generate():
        def run():
            a.download_model_update_stream(req.model_choice, progress_q)

        t = _threading.Thread(target=run, daemon=True)
        t.start()
        while True:
            event = await loop.run_in_executor(None, progress_q.get)
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@app.get("/api/open-file-dialog")
async def api_open_file_dialog():
    """Open a native macOS file picker (image files) and return the selected path."""
    import platform
    if platform.system() != "Darwin":
        raise HTTPException(400, "File picker only supported on macOS")
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e",
            'POSIX path of (choose file with prompt "Select an image to upscale"'
            ' of type {"public.image"})',
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

@app.post("/api/stop")
async def api_stop():
    """Signal the running generation to stop at the next step boundary."""
    _mgr().request_stop()
    return {"status": "stop requested"}


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """
    Server-Sent Events stream.  Each event is a JSON object:
      {"type": "progress", "message": "..."}
      {"type": "image", "url": "/api/output/<run>/outputs/<file>", "path": "<abs path>", "info": "..."}
      {"type": "video", "url": "/api/output/<run>/outputs/<file>", "path": "<abs path>", "info": "..."}
    Each generation is saved as one run folder (core/run_store.py) under the output dir.
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

    params = req.model_dump()
    params["input_images"] = input_images
    params["mask_image"]   = mask_image
    base_dir = Path(req.output_dir or _output_dir())
    slots    = _request_slots(req)

    async def event_stream():
        try:
            run_dir = run_store.create_run(base_dir, req.model_dump(), slots)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Could not create run folder: {e}'})}\n\n"
            return
        try:
            async for event in _run_events(run_dir, params, req.seed):
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


def _output_path(rel: str) -> Path:
    p = run_store.safe_join(Path(_output_dir()), rel)
    if p is None:
        raise HTTPException(400, "Invalid path")
    return p


@app.delete("/api/output/{filename:path}")
def api_delete_output(filename: str):
    """Delete one output. Inside a run: drop it from workflow.json; the last one trashes the run."""
    base = Path(_output_dir()).resolve()
    p = _output_path(filename)
    if not p.is_file():
        raise HTTPException(404, detail="File not found")
    parts = p.relative_to(base).parts
    run_dir = base / parts[0]
    in_run = len(parts) > 1 and (run_dir / "workflow.json").is_file()
    if in_run and not (len(parts) > 2 and parts[1] == "outputs"):
        raise HTTPException(400, detail="Only files in outputs/ can be deleted")
    if in_run:
        if run_store.remove_output(run_dir, "/".join(parts[1:])):
            run_store.trash(run_dir)
    else:
        run_store.trash(p)          # pre-migration flat file
    return {"deleted": filename}


@app.get("/api/output/{filename:path}")
def api_serve_output(filename: str):
    p = _output_path(filename)
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(str(p))


@app.get("/api/outputs")
def api_list_outputs(limit: int = 20):
    """Most recent outputs across all run folders."""
    return {"files": run_store.list_outputs(Path(_output_dir()), limit)}


@app.get("/api/runs/{run:path}")
def api_load_run(run: str):
    base = Path(_output_dir())
    run_dir = run_store.safe_join(base, run)
    if run_dir is None or run_dir == base.resolve():
        raise HTTPException(400, "Invalid run name")
    if not (run_dir / "workflow.json").is_file():
        raise HTTPException(404, f"Run not found: {run}")
    try:
        return run_store.load(run_dir, f"/api/output/{run}")
    except ValueError as e:
        raise HTTPException(500, f"Failed to read workflow.json: {e}")


# ── Routes: Workflows ─────────────────────────────────────────────────────────

@app.get("/api/workflows")
async def api_list_workflows():
    a = _app()
    names = (a.list_saved_workflows() or [])[:15]
    return {"workflows": names}


@app.get("/api/open-workflow-folder-dialog")
async def api_open_workflow_folder_dialog():
    """Open a native macOS folder picker starting at WORKFLOWS_DIR."""
    import platform
    if platform.system() != "Darwin":
        raise HTTPException(400, "Folder picker only supported on macOS")
    a        = _app()
    wf_dir   = str(Path(a.WORKFLOWS_DIR).resolve())
    script   = (
        f'POSIX path of (choose folder with prompt "Select workflow folder"'
        f' default location POSIX file "{wf_dir}")'
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
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
            if err and "cancelled" not in err.lower() and "cancel" not in err.lower():
                raise HTTPException(500, f"osascript error: {err}")
            return {"path": None, "cancelled": True}
        path = stdout.decode().strip().rstrip("/")
        return {"path": path, "cancelled": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/workflows/save")
def api_save_workflow(req: SaveWorkflowRequest):
    a = _app()
    slots = []
    for s in req.ref_slots:
        img = _guarded_temp(s.get("imageId"))
        if img is not None:
            slots.append({"image_path": img, "mask_path": _guarded_temp(s.get("maskId")),
                          "strength": float(s.get("strength", 1.0))})
    try:
        name = run_store.save_workflow(Path(a.WORKFLOWS_DIR), req.model_dump(), slots, req.name,
                                       overwrite=req.overwrite)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": f"✓ Saved: {name}", "name": name}


@app.get("/api/workflows/{name:path}")
def api_load_workflow(name: str):
    a = _app()
    base = Path(a.WORKFLOWS_DIR)
    wf_dir = run_store.safe_join(base, name)
    if wf_dir is None or wf_dir == base.resolve():
        raise HTTPException(400, "Invalid workflow name")
    if not (wf_dir / "workflow.json").is_file():
        raise HTTPException(404, f"Workflow not found: {name}")
    try:
        return run_store.load(wf_dir, f"/api/workflow-assets/{name}")
    except ValueError as e:
        raise HTTPException(500, f"Failed to read workflow.json: {e}")


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
    models_dir = _app().get_local_models_dir()
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

def _detect_lora_type(path: str) -> str:
    """
    Detect model compatibility of a LoRA file by inspecting its header only.
    Returns: "flux" | "zimage" | "unknown"
    """
    try:
        from core.lora_flux2 import check_lora_compatibility
        check_lora_compatibility(path)
        return "flux"
    except RuntimeError as e:
        if "No FLUX LoRA keys found" in str(e):
            return "zimage"
        # Block-count exceeded → full FLUX.1, not compatible with klein
        return "unknown"
    except Exception:
        return "unknown"


@app.get("/api/lora/list")
def api_list_loras():
    """Return all .safetensors/.pt/.bin files in lora_uploads/ with model_type tag and, for
    FLUX LoRAs, the klein size ("4b" | "9b" | null) so the UI can grey out mismatches."""
    from core.lora_flux2 import lora_variant
    from core.lora_triggers import get_trigger_info
    lora_dir = ROOT / "lora_uploads"
    if not lora_dir.exists():
        return {"files": []}
    files = [
        {"name": f.name, "path": str(f), "model_type": (mt := _detect_lora_type(str(f))),
         "variant": lora_variant(str(f)) if mt == "flux" else None,
         **{f"trigger_{k}" if k != "trigger" else k: v
            for k, v in get_trigger_info(str(f)).items()}}
        for f in sorted(lora_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
        if f.suffix in (".safetensors", ".pt", ".bin") and f.is_file()
    ]
    return {"files": files}


class LoraTriggerRequest(BaseModel):
    name:    str   # file name inside lora_uploads/
    trigger: str   # "" clears the user override


@app.put("/api/lora/trigger")
def api_set_lora_trigger(req: LoraTriggerRequest):
    """Save the user's trigger word/prompt for a LoRA (shown as a hint in the LoRA panel)."""
    from core.lora_triggers import set_trigger, get_trigger_info
    try:
        set_trigger(req.name, req.trigger)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return get_trigger_info(str(ROOT / "lora_uploads" / req.name))


@app.post("/api/lora/load")
async def api_load_lora(req: LoadLoraRequest):
    status = await _mgr().load_lora(req.lora_path, req.strength, req.device)
    return {"status": status}


@app.delete("/api/lora")
async def api_clear_lora():
    return {"status": _mgr().clear_lora()}


@app.post("/api/lora/upload")
async def api_upload_lora(file: UploadFile = File(...)):
    """Upload a LoRA .safetensors file. Rejects files incompatible with FLUX.2-klein."""
    from core.lora_flux2 import check_lora_compatibility

    fname = Path(file.filename or "lora.safetensors").name
    dest  = ROOT / "lora_uploads" / fname
    tmp   = ROOT / "lora_uploads" / f".tmp_{fname}"
    dest.parent.mkdir(exist_ok=True)

    # Write to temp path first so we can delete on validation failure
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        check_lora_compatibility(str(tmp))
    except RuntimeError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    tmp.rename(dest)
    return {"path": str(dest), "name": fname}


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


class SingleUpscaleRequest(BaseModel):
    source:       str        # "gallery" | "path"
    filename:     str = ""   # for source="gallery": filename in output dir
    file_path:    str = ""   # for source="path": absolute path on disk
    model_path:   str
    scale_choice: str = "×4"


@app.post("/api/upscale/single")
async def api_upscale_single(req: SingleUpscaleRequest):
    """Upscale one image and save next to the original as <stem>_<W>x<H><ext> (recorded in its run)."""
    from PIL import Image as PILImage

    if not req.model_path:
        raise HTTPException(400, "No upscale model — load one in the Upscale section first")

    # Resolve source path
    if req.source == "gallery":
        if not req.filename:
            raise HTTPException(400, "filename required for gallery source")
        src_path = _output_path(req.filename)
    elif req.source == "path":
        if not req.file_path:
            raise HTTPException(400, "file_path required for path source")
        src_path = Path(req.file_path)
    else:
        raise HTTPException(400, f"Unknown source: {req.source!r}")

    if not src_path.exists():
        raise HTTPException(404, f"Image not found: {src_path}")

    # Run upscale in thread (CPU/GPU-bound)
    def _run() -> tuple:
        a = _app()
        device = a.get_available_devices()[0]
        scale_map = {"×2": 2, "×3": 3, "×4": 4}
        target = scale_map.get(req.scale_choice, 4)
        img = PILImage.open(src_path).convert("RGB")
        upscaled = a.upscale_image(img, req.model_path, device)
        # Resize down if ×2 or ×3 (model outputs ×4)
        if target < 4:
            orig_w, orig_h = img.size
            upscaled = upscaled.resize(
                (orig_w * target, orig_h * target),
                PILImage.LANCZOS,
            )
        w, h = upscaled.size
        out_name = f"{src_path.stem}_{w}x{h}{src_path.suffix or '.png'}"
        out_path = src_path.parent / out_name
        upscaled.save(out_path)
        return str(out_path), out_name, w, h

    import asyncio as _aio
    from concurrent.futures import ThreadPoolExecutor
    loop = _aio.get_event_loop()
    try:
        saved_path, out_name, w, h = await loop.run_in_executor(
            ThreadPoolExecutor(max_workers=1), _run
        )
    except Exception as e:
        raise HTTPException(500, f"Upscale failed: {e}")

    run_dir = Path(saved_path).parent.parent
    if Path(saved_path).parent.name == "outputs" and (run_dir / "workflow.json").is_file():
        try:
            run_store.add_output(run_dir, saved_path, "image", upscaled_from=f"outputs/{src_path.name}")
        except Exception as e:
            print(f"[run_store] could not record upscale {out_name}: {e}")

    # Build URL if saved inside output dir
    url: str | None = None
    try:
        rel = Path(saved_path).resolve().relative_to(Path(_output_dir()).resolve())
        url = f"/api/output/{rel}"
    except ValueError:
        pass

    return {"saved_path": saved_path, "filename": out_name, "url": url, "width": w, "height": h}


# ── Routes: Depth Map ─────────────────────────────────────────────────────────

class DepthMapRequest(BaseModel):
    file_path:  str | None = None   # absolute path from file picker
    filename:   str | None = None   # legacy: output-dir filename
    model_repo: str = "istiakiat/DA3MONO-LARGE"


@app.post("/api/depth-map")
async def api_depth_map(req: DepthMapRequest):
    """Generate a depth map for an image file."""
    from pipeline import manager
    if manager.is_busy:
        raise HTTPException(503, "Pipeline is busy — wait for generation to finish")

    if req.file_path:
        src_path = Path(req.file_path)
    elif req.filename:
        src_path = Path(_output_dir()) / Path(req.filename).name
    else:
        raise HTTPException(400, "Provide either file_path or filename")

    if not src_path.exists():
        raise HTTPException(400, f"File not found: {src_path}")

    out_name = src_path.stem + "_depth.png"
    out_path = Path(_output_dir()) / out_name

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor(1) as pool:
            png_bytes = await loop.run_in_executor(
                pool,
                lambda: _run_depth_map(str(src_path), req.model_repo),
            )
    except Exception as e:
        import traceback
        traceback.print_exc()   # full traceback goes to server.log via _LogTee
        raise HTTPException(500, f"Depth map generation failed: {e}")

    out_path.write_bytes(png_bytes)
    url = f"/api/output/{out_name}"
    return {"url": url, "filename": out_name}


def _run_depth_map(image_path: str, repo_id: str) -> bytes:
    from core.depth_map import generate_depth_map
    # DA2 outputs disparity (larger = nearer) — no inversion needed.
    # DA3 and others output true depth (larger = farther) — invert for white=near.
    invert = not repo_id.startswith("depth-anything/Depth-Anything-V2")
    return generate_depth_map(image_path, repo_id=repo_id, invert=invert)


# ── Routes: Watermark Remover ─────────────────────────────────────────────────

class EraseDetectRequest(BaseModel):
    file_path: str


class EraseRequest(BaseModel):
    file_path: str
    mask_id:   str


@app.post("/api/erase/detect")
async def api_erase_detect(req: EraseDetectRequest):
    """Run heuristic watermark detection. Returns temp URLs for image + mask."""
    src_path = Path(req.file_path)
    if not src_path.exists():
        raise HTTPException(400, f"File not found: {src_path}")

    # Copy source image to temp so frontend can preview it
    img_id = f"{uuid.uuid4().hex}{src_path.suffix or '.png'}"
    shutil.copy2(src_path, _temp_path(img_id))

    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor(1) as pool:
            mask_bytes = await loop.run_in_executor(
                pool, lambda: _run_erase_detect(str(src_path))
            )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Detection failed: {e}")

    mask_id = f"{uuid.uuid4().hex}.png"
    _temp_path(mask_id).write_bytes(mask_bytes)

    return {
        "image_id":  img_id,
        "image_url": f"/api/temp/{img_id}",
        "mask_id":   mask_id,
        "mask_url":  f"/api/temp/{mask_id}",
    }


@app.post("/api/erase")
async def api_erase(req: EraseRequest):
    """Fill the masked region with LaMa inpainting. Saves result to output dir."""
    src_path = Path(req.file_path)
    if not src_path.exists():
        raise HTTPException(400, f"File not found: {src_path}")

    mask_path = _temp_path(req.mask_id)
    if not mask_path.resolve().is_relative_to(TEMP_DIR.resolve()):
        raise HTTPException(400, "Invalid mask_id")
    if not mask_path.exists():
        raise HTTPException(400, f"Mask not found: {req.mask_id}")

    mask_bytes = mask_path.read_bytes()

    base_name = src_path.stem + "_erased"
    out_name  = f"{base_name}.png"
    counter   = 2
    while (Path(_output_dir()) / out_name).exists():
        out_name = f"{base_name}_{counter}.png"
        counter += 1
    out_path = Path(_output_dir()) / out_name

    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor(1) as pool:
            result_bytes = await loop.run_in_executor(
                pool, lambda: _run_erase(str(src_path), mask_bytes)
            )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Watermark removal failed: {e}")

    out_path.write_bytes(result_bytes)
    out_path.with_suffix(".json").write_text(
        json.dumps({"source": str(src_path), "operation": "watermark_removal"}, indent=2)
    )

    return {"url": f"/api/output/{out_name}", "filename": out_name}


def _run_erase_detect(image_path: str) -> bytes:
    from core.erase import detect_watermark
    return detect_watermark(Path(image_path))


def _run_erase(image_path: str, mask_bytes: bytes) -> bytes:
    from core.erase import remove_watermark
    return remove_watermark(Path(image_path), mask_bytes)


# ── SAM click-to-mask (mask editor) ───────────────────────────────────────────

class SegmentPrepareRequest(BaseModel):
    image_id: str


class SegmentPoint(BaseModel):
    x: float
    y: float
    label: int = 1          # 1 = include, 0 = exclude


class SegmentBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class SegmentRequest(BaseModel):
    image_id: str
    points: list[SegmentPoint] = Field(default_factory=list)
    box: SegmentBox | None = None


# One thread: SAM calls run one at a time and share the embedding cache.
from concurrent.futures import ThreadPoolExecutor as _SegPool
_segment_pool = _SegPool(max_workers=1)


def _segment_image_loader(image_id: str):
    """Guarded temp path → loader returning the image as the browser shows it
    (EXIF orientation applied), RGB."""
    path = _temp_path(image_id)
    if not path.resolve().is_relative_to(TEMP_DIR.resolve()) or not path.is_file():
        raise HTTPException(400, f"Invalid image_id: {image_id}")

    def load():
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    return load


@app.get("/api/segment/status")
def api_segment_status():
    """Whether SAM weights are already loaded (no load triggered)."""
    import core.segment as seg
    return {"loaded": seg.get_segmenter().loaded}


@app.post("/api/segment/prepare")
async def api_segment_prepare(req: SegmentPrepareRequest):
    """Load SAM (first call) and cache this image's embedding so clicks are instant."""
    import core.segment as seg
    loader = _segment_image_loader(req.image_id)
    t0 = time.perf_counter()
    try:
        await asyncio.get_event_loop().run_in_executor(
            _segment_pool, lambda: seg.get_segmenter().prepare(req.image_id, loader))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"SAM failed: {e}")
    return {"ready": True, "ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/api/segment")
async def api_segment(req: SegmentRequest):
    """One SAM object mask for points (+1/0 labels) and/or one box → PNG (255 = object)."""
    import io
    import numpy as np
    from PIL import Image
    import core.segment as seg
    loader = _segment_image_loader(req.image_id)
    prompts = {"points": [p.model_dump() for p in req.points],
               "box": req.box.model_dump() if req.box else None}
    try:
        mask = await asyncio.get_event_loop().run_in_executor(
            _segment_pool, lambda: seg.get_segmenter().segment(req.image_id, loader, prompts))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"SAM failed: {e}")
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


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


# ── Routes: Model Sources ──────────────────────────────────────────────────────

def _show_nsfw() -> bool:
    return bool(_app().load_settings().get("civitai_show_nsfw"))


def _loaded_lora_paths() -> list[str]:
    return [p["path"] for p in _app().current_lora_paths]


def _stored_sources() -> list[dict]:
    """Rows as stored (pruned, sorted), including NSFW CivitAI rows the list view hides."""
    _defaults_by_id = {s["id"]: s for s in DEFAULT_SOURCES}
    sources = _read_sources_file().get("sources", DEFAULT_SOURCES)
    # Merge model_choice from DEFAULT_SOURCES for any entry that lacks it
    for s in sources:
        if "model_choice" not in s and s.get("id") in _defaults_by_id:
            s["model_choice"] = _defaults_by_id[s["id"]].get("model_choice", "")
    return _sort_sources(model_sources.prune_list(_drop_unusable_base(sources)))


@app.get("/api/model-sources")
def api_get_model_sources():
    """Return the model sources list. Seeds from DEFAULT_SOURCES if file missing.
    Always merges model_choice from DEFAULT_SOURCES by ID so existing saved files
    get the field even if they were written before it existed. LoRAs this app cannot use
    (wrong model family) are pruned unless the user added them by hand (`custom`).
    CivitAI rows are annotated installed/update; NSFW ones are hidden unless the setting is on."""
    return {"sources": civitai.annotate(_stored_sources(), civitai_install.installed_map(), _show_nsfw())}


def _read_sources_file() -> dict:
    """Raw model_sources.json ({} when missing or unreadable)."""
    if MODEL_SOURCES_FILE.exists():
        try:
            data = json.loads(MODEL_SOURCES_FILE.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def _write_sources_file(sources: list[dict], ignored: list[str]) -> None:
    data = {"version": 1, "sources": sources}
    if ignored:
        data["ignored"] = ignored
    MODEL_SOURCES_FILE.write_text(json.dumps(data, indent=2))


_TYPE_ORDER = {"base": 0, "lora": 1, "upscaler": 2}

def _sort_sources(sources: list[dict]) -> list[dict]:
    return sorted(sources, key=lambda s: _TYPE_ORDER.get(s.get("type", "base"), 3))


def _drop_unusable_base(sources: list[dict]) -> list[dict]:
    """Remove base-type entries whose repo the loader can't load.

    The model loader only handles repos in app.KNOWN_MODELS — a base source for any
    other repo can neither download in-app nor load, so it's dead weight in the UI.
    LoRAs/upscalers are link-only by design and always kept."""
    supported = {f"https://huggingface.co/{rid}" for rid in _app().KNOWN_MODELS}
    return [s for s in sources if s.get("type") != "base" or s.get("url") in supported]


def _discover_candidates(existing_urls: set[str], next_id: int) -> list[dict]:
    """Search HuggingFace for Apple Silicon-compatible repos not already in the source list."""
    from huggingface_hub import HfApi

    # Repo IDs that are incompatible with Apple Silicon — never add these
    BLOCKLIST = {
        "black-forest-labs/FLUX.2-klein-9b-fp8",  # FP8: no native MPS support
    }

    BASE_ORGS     = {"Disty0", "Tongyi-MAI", "Lightricks", "aydin99", "black-forest-labs"}
    LORA_ORGS     = {"fal", "nomadoor", "DeverStyle", "WarmBloodAban", "linoyts",
                     "dx8152", "vafipas663", "valiantcat", "CodeGoat24"}
    UPSCALER_ORGS = {"Comfy-Org", "Phips", "Kim2091", "OzzyGT", "halffried",
                     "GraydientPlatformAPI", "uwg"}

    # Keywords that must appear in repo name or tags for a model to be relevant
    BASE_KW     = {"flux", "z-image", "zimage", "klein", "sdnq", "ltx", "turbo", "diffusion"}
    LORA_KW     = {"lora"}
    UPSCALER_KW = {"esrgan", "drct", "swinir", "hat", "upscal", "plksr", "atd",
                   "remacri", "ultrasharp", "4x-", "4xreal", "4xnomos", "gyre"}

    def _is_relevant(repo_id: str, tags: list, org_type: str) -> bool:
        combined = (repo_id + " " + " ".join(tags)).lower()
        if org_type == "upscaler":
            return any(k in combined for k in UPSCALER_KW)
        if org_type == "lora":
            return any(k in combined for k in LORA_KW | BASE_KW)
        return any(k in combined for k in BASE_KW)

    def _infer_type(repo_id: str, tags: list) -> str:
        author = repo_id.split("/")[0]
        if author in UPSCALER_ORGS:
            return "upscaler"
        tag_str = " ".join(tags).lower()
        if author in LORA_ORGS or "lora" in repo_id.lower() or "lora" in tag_str:
            return "lora"
        return "base"

    api  = HfApi()
    candidates: list[dict] = []
    seen_urls: set[str]    = set()

    all_orgs_typed = (
        [(org, "base")     for org in BASE_ORGS] +
        [(org, "lora")     for org in LORA_ORGS] +
        [(org, "upscaler") for org in UPSCALER_ORGS]
    )

    for org, org_type in all_orgs_typed:
        try:
            for m in api.list_models(author=org, limit=50):
                url = f"https://huggingface.co/{m.id}"
                if url in existing_urls or url in seen_urls or m.id in BLOCKLIST:
                    continue
                model_tags = list(m.tags or [])
                if not _is_relevant(m.id, model_tags, org_type):
                    continue
                seen_urls.add(url)
                candidates.append({
                    "id":           f"src-{next_id:03d}",
                    "name":         m.id.split("/")[-1],
                    "url":          url,
                    "type":         _infer_type(m.id, model_tags),
                    "description":  "",
                    "model_choice": "",
                })
                next_id += 1
        except Exception:
            continue

    # MPS tag search — keyword filter required (very broad tag)
    try:
        for m in api.list_models(tags="mps", limit=50):
            url = f"https://huggingface.co/{m.id}"
            if url in existing_urls or url in seen_urls or m.id in BLOCKLIST:
                continue
            model_tags = list(m.tags or [])
            combined = (m.id + " " + " ".join(model_tags)).lower()
            if not any(k in combined for k in BASE_KW | LORA_KW | UPSCALER_KW):
                continue
            seen_urls.add(url)
            candidates.append({
                "id":           f"src-{next_id:03d}",
                "name":         m.id.split("/")[-1],
                "url":          url,
                "type":         _infer_type(m.id, model_tags),
                "description":  "",
                "model_choice": "",
            })
            next_id += 1
    except Exception:
        pass

    return candidates


@app.get("/api/model-sources/discover")
def api_discover_model_sources():
    """Update: find new repos, keep only those this app can use, fill in what each one does."""
    current = _stored_sources()
    ignored = list(_read_sources_file().get("ignored", []))
    existing_urls = {s["url"] for s in current}
    existing_ids = [
        int(s["id"].replace("src-", ""))
        for s in current if s.get("id", "").startswith("src-") and s["id"][4:].isdigit()
    ]
    next_id = max(existing_ids, default=0) + 1

    candidates = _drop_unusable_base(_discover_candidates(existing_urls, next_id))  # never surface base models the loader can't load
    merged, ignored, report = model_sources.merge_discovery(
        current, candidates, ignored, model_sources.hf_fetch_files, model_sources.hf_fetch_card)
    fresh, failed_bases = civitai.discover(civitai.http_fetch_models, _show_nsfw())
    failed_fam = {civitai.BASE_FAMILY[b] for b in failed_bases}
    installed = civitai_install.installed_map()
    merged, c_added = civitai.merge_rows(merged, fresh, failed_fam, set(installed))
    updated = _sort_sources(merged)
    _write_sources_file(updated, ignored)
    visible = civitai.annotate(updated, installed, _show_nsfw())
    return {**report, "civitai": {"added": c_added, "failed": sorted(failed_fam),
                                  "updates": sum(1 for s in visible if s.get("update"))},
            "sources": visible}


@app.post("/api/model-sources")
def api_save_model_sources(payload: dict = Body(...)):
    """Save the model sources list."""
    sources = payload.get("sources", [])
    valid_types = {"base", "lora", "upscaler"}
    for s in sources:
        if not s.get("name") or not s.get("url"):
            raise HTTPException(400, "Each source must have a non-empty name and url.")
        if s.get("type") not in valid_types:
            raise HTTPException(400, f"Invalid type '{s.get('type')}'. Must be base, lora, or upscaler.")
    sources = [{k: v for k, v in s.items() if k not in ("installed", "update", "installed_version")}
               for s in sources]
    if not _show_nsfw():
        in_ids = {s.get("id") for s in sources}
        installed = civitai_install.installed_map()
        for s in _read_sources_file().get("sources", []):
            c = s.get("civitai") or {}
            if (s.get("provider") == "civitai" and s.get("nsfw") and s.get("id") not in in_ids
                    and (c.get("modelId"), s.get("family")) not in installed):
                sources.append(s)                  # hidden by the toggle, not removed by the user
    _write_sources_file(sources, list(_read_sources_file().get("ignored", [])))
    return {"ok": True}


# ── Routes: CivitAI ───────────────────────────────────────────────────────────

class CivitaiKeyRequest(BaseModel):
    key: str


class CivitaiDownloadRequest(BaseModel):
    version_id: int


@app.get("/api/civitai/status")
def api_civitai_status():
    return {"has_key": civitai_install.get_key() is not None, "show_nsfw": _show_nsfw()}


@app.post("/api/civitai/key")
def api_civitai_set_key(req: CivitaiKeyRequest):
    try:
        civitai_install.set_key(req.key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"has_key": True}


@app.delete("/api/civitai/key")
def api_civitai_clear_key():
    civitai_install.clear_key()
    return {"has_key": False}


@app.post("/api/civitai/download")
def api_civitai_download(req: CivitaiDownloadRequest):
    for s in _stored_sources():
        if s.get("provider") == "civitai" and (s.get("civitai") or {}).get("versionId") == req.version_id:
            return civitai_install.start_download(s)
    raise HTTPException(status_code=404, detail="Unknown CivitAI version: run Update first")


@app.get("/api/civitai/download/{version_id}")
def api_civitai_download_state(version_id: int):
    return civitai_install.job_state(version_id)


@app.delete("/api/civitai/{version_id}")
def api_civitai_delete(version_id: int):
    try:
        name = civitai_install.delete_installed(version_id, _loaded_lora_paths())
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"deleted": name}


# ── Routes: Storage ───────────────────────────────────────────────────────────

@app.get("/api/storage")
def api_storage():
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
    """Return info about non-HF-cache models: upscale files."""
    upscale_exts = {".pth", ".safetensors", ".ckpt", ".pt", ".bin"}
    upscale_dir  = ROOT / "upscale_models"
    upscale_list = []
    if upscale_dir.exists():
        for f in sorted(upscale_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in upscale_exts:
                upscale_list.append({"name": f.name, "size": _fmt_bytes(f.stat().st_size)})
    return {"upscale_models": upscale_list}


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
def api_hf_status():
    # Sync def — FastAPI runs this in a thread pool so the blocking whoami()
    # network call never freezes the event loop (and other /api/* requests).
    return {"status": _app().hf_get_status()}

@app.post("/api/hf/login")
def api_hf_login(req: HFLoginRequest):
    return {"status": _app().hf_login_token(req.token)}

@app.post("/api/hf/logout")
def api_hf_logout():
    return {"status": _app().hf_logout()}


# ── Routes: Logs ──────────────────────────────────────────────────────────────

@app.post("/api/logs/save")
async def api_save_log():
    """Save a timestamped snapshot of the current session log inside logs/."""
    import datetime
    if not LOG_FILE.exists():
        raise HTTPException(404, "No log file found for this session")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = LOGS_DIR / f"server_log_{timestamp}.txt"
    shutil.copy2(LOG_FILE, snapshot)
    return {"saved_path": str(snapshot)}


# ── Routes: Workflow assets ───────────────────────────────────────────────────

@app.get("/api/workflow-assets/{name}/{filename:path}")
async def api_workflow_asset(name: str, filename: str):
    a    = _app()
    base = Path(a.WORKFLOWS_DIR).resolve()
    path = (base / name / filename).resolve()
    if not path.is_relative_to(base):
        raise HTTPException(400, "Invalid path")
    if not path.is_file():
        raise HTTPException(404, "Asset not found")
    return FileResponse(str(path))


# ── Routes: Batch generation ──

@app.post("/api/batch/generate")
async def api_batch_generate(req: BatchGenerateRequest):
    """
    Batch img2img: process each image in input_folder using current params.
    Streams SSE events: batch_progress, progress, image, video, error, done.
    """
    if _mgr().is_busy:
        raise HTTPException(423, "A generation is already running")

    async def event_stream():
        from PIL import Image as PILImage
        mgr = _mgr()

        # Validate and list input images
        folder = Path(req.input_folder)
        if not folder.is_dir():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Input folder not found: {req.input_folder}'})}\n\n"
            return

        exts = {'.jpg', '.jpeg', '.png', '.webp'}
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
        if not images:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No images found in folder'})}\n\n"
            return

        base_dir = Path(req.output_dir or _output_dir())
        total = len(images)
        processed = 0
        mgr.is_batch_running = True
        try:
            for i, img_path in enumerate(images):
                if mgr.stop_requested:
                    break

                yield f"data: {json.dumps({'type': 'batch_progress', 'current': i + 1, 'total': total, 'filename': img_path.name})}\n\n"

                # Load image
                try:
                    pil_image = PILImage.open(img_path).convert("RGB")
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Skipping {img_path.name}: {e}'})}\n\n"
                    continue

                params = req.model_dump()
                params["input_images"] = [pil_image]
                params["mask_image"]   = None
                try:
                    run_dir = run_store.create_run(base_dir, params, [
                        {"image_path": img_path, "mask_path": None, "strength": req.img_strength}])
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Could not create run folder for {img_path.name}: {e}'})}\n\n"
                    continue

                try:
                    async for event in _run_events(run_dir, params, req.seed):
                        yield f"data: {json.dumps(event)}\n\n"
                    processed += 1
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Error on {img_path.name}: {e}'})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'processed': processed})}\n\n"
        finally:
            mgr.is_batch_running = False

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging (params dump, branch tracing, LoRA state)")
    args = parser.parse_args()

    if args.debug:
        import pipeline as _pl
        _pl.DEBUG = True
        print("[DEBUG] Debug mode enabled")

    if args.no_auto_shutdown:
        os.environ["IMAGEGEN_NO_AUTO_SHUTDOWN"] = "1"

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
