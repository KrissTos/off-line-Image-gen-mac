"""
PipelineManager — async bridge between FastAPI and the generation logic in app.py.

Runs generate_image() in a thread executor (it's a sync generator) and converts
each (image, video, status) yield into a dict event that FastAPI streams as SSE.

Usage:
    from pipeline import manager
    async for event in manager.generate(params):
        yield f"data: {json.dumps(event)}\n\n"
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import json
import os
import time
from typing import Any, AsyncGenerator

# ── No-op progress shim ───────────────────────────────────────────────────────
# generate_image() expects a gr.Progress-compatible object as its last arg.
# This drop-in does nothing, so we avoid importing Gradio just for the type.

class _NoOpProgress:
    """Dummy gr.Progress that silently ignores all calls."""
    def __init__(self, track_tqdm: bool = False):
        pass
    def __call__(self, value=None, desc=None, total=None, unit=None, **kw):
        pass
    def tqdm(self, iterable, *args, **kw):
        return iterable


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_data_url(img) -> str:
    """Convert a PIL Image to a base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _save_output_image(img, output_dir: str, prompt: str, seed: int) -> str:
    """Save PIL image to output_dir, return the file path."""
    import re, datetime
    os.makedirs(output_dir, exist_ok=True)
    slug = re.sub(r"[^\w\s-]", "", (prompt or "image"))[:40].strip().replace(" ", "_")
    # Use millisecond precision so repeat-count images within the same second get unique names
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]   # …_mmm
    filename = f"{ts}_{seed}_{slug}.png"
    path = os.path.join(output_dir, filename)
    img.save(path, "PNG")
    return path


# ── PipelineManager ───────────────────────────────────────────────────────────

class PipelineManager:
    """
    Singleton that serialises access to the GPU pipeline.

    All heavy work runs in a single-thread ThreadPoolExecutor so asyncio
    is never blocked, while ensuring only one generation runs at a time.
    """

    def __init__(self):
        self._lock    = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                                thread_name_prefix="pipeline")
        self.is_busy   = False

    # ── status ────────────────────────────────────────────────────────────────

    def current_status(self) -> dict:
        """Return a dict describing the currently loaded model / device."""
        try:
            import app as _app
            return {
                "model":   getattr(_app, "current_model",  None),
                "device":  getattr(_app, "current_device", None),
                "loaded":  getattr(_app, "pipe", None) is not None,
                "busy":    self.is_busy,
                "vram_gb": self._get_vram(),
            }
        except Exception:
            return {"model": None, "device": None, "loaded": False,
                    "busy": self.is_busy, "vram_gb": 0.0}

    def _get_vram(self) -> float:
        try:
            import app as _app
            return _app.get_memory_usage()
        except Exception:
            return 0.0

    # ── model loading ─────────────────────────────────────────────────────────

    async def load_model(self, model_choice: str, device: str) -> str:
        """Load a model pipeline.  Returns a status string."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._load_model_sync(model_choice, device),
        )

    def _load_model_sync(self, model_choice: str, device: str) -> str:
        import app as _app
        try:
            _app.load_pipeline(model_choice, device)
            return f"✓ Loaded {model_choice} on {device}"
        except Exception as e:
            return f"✗ Failed to load {model_choice}: {e}"

    # ── LoRA ──────────────────────────────────────────────────────────────────

    async def load_lora(self, lora_path: str, strength: float, device: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._load_lora_sync(lora_path, strength, device),
        )

    def _load_lora_sync(self, lora_path: str, strength: float, device: str) -> str:
        import app as _app
        return _app.load_lora(lora_path, strength, device)

    def clear_lora(self) -> str:
        import app as _app
        try:
            return _app.clear_lora()[1]   # clear_lora() returns (None, status_msg)
        except Exception as e:
            return f"Error: {e}"

    # ── generation ────────────────────────────────────────────────────────────

    async def generate(self, params: dict) -> AsyncGenerator[dict, None]:
        """
        Async generator that runs generate_image() in a thread and yields
        SSE-ready event dicts.

        params keys mirror generate_image() positional arguments plus:
            output_dir  (str)  — where to save images
        """
        if self._lock.locked():
            yield {"type": "error", "message": "A generation is already running."}
            return

        async with self._lock:
            self.is_busy = True
            try:
                async for event in self._run_generate(params):
                    yield event
            finally:
                self.is_busy = False

    async def _run_generate(self, params: dict) -> AsyncGenerator[dict, None]:
        loop    = asyncio.get_event_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def _thread():
            """Runs in the thread executor.  Pushes events onto the asyncio Queue."""
            import app as _app

            # Build generate_image() kwargs
            kwargs = dict(
                prompt            = params.get("prompt", ""),
                height            = int(params.get("height", 512)),
                width             = int(params.get("width", 512)),
                steps             = int(params.get("steps", 20)),
                seed              = int(params.get("seed", -1)),
                guidance          = float(params.get("guidance", 7.0)),
                device            = params.get("device", "mps"),
                model_choice      = params.get("model_choice", _app.MODEL_CHOICES[0]),
                model_source_choice = params.get("model_source", "Local"),
                input_images      = params.get("input_images"),     # list[PIL] or None
                lora_files        = params.get("lora_files") or [],
                lora_file         = params.get("lora_file"),
                lora_strength     = float(params.get("lora_strength", 1.0)),
                img_strength      = float(params.get("img_strength", 1.0)),
                repeat_count      = int(params.get("repeat_count", 1)),
                auto_save         = False,   # pipeline handles saving + sidecar; prevents double-save
                output_dir        = params.get("output_dir", _app.DEFAULT_OUTPUT_DIR),
                upscale_enabled   = bool(params.get("upscale_enabled", False)),
                upscale_model_path= params.get("upscale_model_path", ""),
                num_frames        = int(params.get("num_frames", 25)),
                fps_val           = int(params.get("fps", 24)),
                mask_image        = params.get("mask_image"),       # PIL or None
                mask_mode         = params.get("mask_mode", "Crop & Composite (Fast)"),
                outpaint_align    = params.get("outpaint_align", "center"),
                progress          = _NoOpProgress(track_tqdm=True),
                step_callback     = lambda s, t: asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "progress", "message": f"Step {s}/{t}", "step": s, "total": t}),
                    loop,
                ),
            )

            try:
                for image, video, status in _app.generate_image(**kwargs):
                    if image is not None:
                        # Save to disk; derive URL from filename.
                        # app.py embeds the actual per-iteration seed in status ("Seed: 123456 | …"),
                        # so parse it out for the filename instead of the original seed kwarg.
                        import re as _re
                        out_dir  = kwargs["output_dir"]
                        _sm = _re.search(r"Seed:\s*(\d+)", status or "")
                        seed_val = int(_sm.group(1)) if _sm else kwargs["seed"]
                        fpath    = _save_output_image(
                            image, out_dir, kwargs["prompt"], seed_val
                        )
                        fname    = os.path.basename(fpath)
                        event = {
                            "type": "image",
                            "url":  f"/api/output/{fname}",
                            "path": fpath,
                            "info": status or "",
                        }
                    elif video is not None:
                        fname = os.path.basename(video)
                        event = {
                            "type": "video",
                            "url":  f"/api/output/{fname}",
                            "path": video,
                        }
                    else:
                        event = {"type": "progress", "message": status or ""}

                    asyncio.run_coroutine_threadsafe(queue.put(event), loop)

                # sentinel — generation finished
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            except MemoryError:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error",
                               "message": "Out of memory — try a smaller resolution."}),
                    loop,
                )
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "message": str(exc)}),
                    loop,
                )
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        future = loop.run_in_executor(self._executor, _thread)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        await asyncio.wrap_future(future)   # propagate any unhandled exception


# ── Module-level singleton ────────────────────────────────────────────────────

manager = PipelineManager()
