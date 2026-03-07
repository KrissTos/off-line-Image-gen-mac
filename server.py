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
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

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

# ── Lazy imports (avoid loading torch at import time) ─────────────────────────

def _app():
    import app as _a
    return _a

def _mgr():
    from pipeline import manager
    return manager

def _wu():
    from workflow_utils import load_any_workflow, get_locally_available_models
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
    return getattr(a, "DEFAULT_OUTPUT_DIR",
                   os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen"))


# ── Routes: Status / devices / models ────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    return _mgr().current_status()


@app.get("/api/devices")
async def api_devices():
    return {"devices": _app().get_available_devices()}


@app.get("/api/models")
async def api_models():
    from workflow_utils import get_locally_available_models
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

    params = req.model_dump()
    params["input_images"] = input_images
    params["mask_image"]   = mask_image
    params["output_dir"]   = req.output_dir or _output_dir()

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

    from workflow_utils import get_locally_available_models as glam
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
    return {"models": models, "summary": summary}


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
    args = parser.parse_args()

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
