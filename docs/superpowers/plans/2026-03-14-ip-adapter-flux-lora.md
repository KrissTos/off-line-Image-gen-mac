# IP-Adapter + FLUX LoRA Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add InstantX IP-Adapter support (up to 3 reference images) and extend LoRA loading to work with FLUX.2-klein models including CivitAI-trained LoRA compatibility.

**Architecture:** New `core/ip_adapter_flux.py` handles all IP-Adapter weight management and pipeline injection. New `core/lora_flux2.py` handles key remapping for FLUX.2-klein LoRA compatibility. Both plug into `app.py`'s existing `generate_image()` generator via new optional params. Backend exposes 3 new API endpoints. Frontend adds a new IP-Adapter accordion in `Sidebar.tsx`, extends the LoRA accordion to cover FLUX models, and wires new state/actions through `store.ts`.

**Tech Stack:** Python/FastAPI backend, diffusers `IPAdapterMixin` (`pipe.load_ip_adapter` / `pipe.set_ip_adapter_scale`), HuggingFace Hub download, React/TypeScript/Tailwind frontend, existing SSE pattern for download progress.

**Design doc:** `docs/plans/2026-03-14-ip-adapter-lora-flux-ui-design.md`

---

## Chunk 1: Backend — IP-Adapter core module + API endpoints

### Task 1: Create `core/ip_adapter_flux.py`

**Files:**
- Create: `core/ip_adapter_flux.py`

- [ ] **Step 1: Create the module**

```python
# core/ip_adapter_flux.py
"""
IP-Adapter management for FLUX.2-klein pipelines.
Uses InstantX/FLUX.1-dev-IP-Adapter weights via native diffusers IPAdapterMixin.

Weight layout on disk:
  ./models/ip_adapter/instantx/ip_adapter.bin       (~5.3 GB)
  (SigLIP encoder is downloaded automatically by HF Hub into HF_HUB_CACHE)
"""
from __future__ import annotations
import os
from pathlib import Path

# Where we store the adapter weights (not in HF_HUB_CACHE — user-visible)
IP_ADAPTER_DIR  = Path("./models/ip_adapter/instantx")
IP_ADAPTER_FILE = IP_ADAPTER_DIR / "ip_adapter.bin"
REPO_ID         = "InstantX/FLUX.1-dev-IP-Adapter"
WEIGHT_NAME     = "ip_adapter.bin"


def is_downloaded() -> bool:
    """Return True if adapter weights file exists locally."""
    return IP_ADAPTER_FILE.exists()


def load_ip_adapter(pipe) -> None:
    """
    Inject IP-Adapter into an already-loaded Flux2KleinPipeline.
    Must be called after the model is loaded, before generation.
    Idempotent — calling twice is safe (diffusers handles it).
    """
    if not is_downloaded():
        raise RuntimeError("IP-Adapter weights not downloaded. Call download first.")
    pipe.load_ip_adapter(
        str(IP_ADAPTER_DIR),
        weight_name=WEIGHT_NAME,
        image_encoder_pretrained_model_name_or_path="google/siglip-so400m-patch14-384",
    )


def unload_ip_adapter(pipe) -> None:
    """Remove IP-Adapter from pipeline. Safe to call even if not loaded."""
    try:
        pipe.unload_ip_adapter()
    except Exception:
        pass


def set_scale(pipe, scales: list[float]) -> None:
    """
    Set per-image IP-Adapter scale(s).
    Pass a list even for a single image — diffusers handles both forms.
    """
    pipe.set_ip_adapter_scale(scales if len(scales) > 1 else scales[0])


def download(progress_cb=None) -> None:
    """
    Download ip_adapter.bin from HF Hub into IP_ADAPTER_DIR.
    progress_cb(downloaded_bytes, total_bytes) called during download.
    """
    from huggingface_hub import hf_hub_download
    IP_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=WEIGHT_NAME,
        local_dir=str(IP_ADAPTER_DIR),
        # HF Hub streams automatically; progress via tqdm callback
    )
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac
source venv/bin/activate
python -c "from core.ip_adapter_flux import is_downloaded, IP_ADAPTER_FILE; print('OK', is_downloaded())"
```
Expected: `OK False`

- [ ] **Step 3: Commit**

```bash
git add core/ip_adapter_flux.py
git commit -m "feat: add core/ip_adapter_flux.py — IP-Adapter weight management"
```

---

### Task 2: Add download progress streaming to `core/ip_adapter_flux.py`

The download needs to stream progress via SSE (same pattern as batch upscale). HF Hub's `hf_hub_download` doesn't expose byte-level progress natively, but we can use `huggingface_hub.file_download` internals or a simple polling approach. Best approach: use `requests` to stream the file with a chunk loop.

**Files:**
- Modify: `core/ip_adapter_flux.py`

- [ ] **Step 1: Replace `download()` with streaming version**

```python
def download(progress_cb=None) -> None:
    """
    Download ip_adapter.bin with byte-level progress reporting.
    progress_cb(downloaded: int, total: int) — both in bytes.
    total may be 0 if server doesn't send Content-Length.
    """
    import requests
    from huggingface_hub import HfApi

    IP_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve actual download URL via HF Hub API
    api  = HfApi()
    url  = api.hf_hub_url(repo_id=REPO_ID, filename=WEIGHT_NAME)
    headers = {}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest = IP_ADAPTER_FILE
    tmp  = dest.with_suffix(".tmp")

    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        total      = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)

    tmp.rename(dest)


def _hf_token() -> str | None:
    """Read HF token from local file (same location app.py uses)."""
    token_path = Path("huggingface/token")
    if token_path.exists():
        return token_path.read_text().strip() or None
    return None
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "from core.ip_adapter_flux import download; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add core/ip_adapter_flux.py
git commit -m "feat: ip_adapter_flux — streaming download with byte-level progress"
```

---

### Task 3: Add IP-Adapter endpoints to `server.py`

**Files:**
- Modify: `server.py`

- [ ] **Step 1: Add import at top of server.py** (find the section with other core imports)

```python
from core.ip_adapter_flux import (
    is_downloaded        as ipa_is_downloaded,
    load_ip_adapter      as ipa_load,
    unload_ip_adapter    as ipa_unload,
    set_scale            as ipa_set_scale,
    download             as ipa_download,
)
```

- [ ] **Step 2: Add state flag to track if IP-Adapter is currently loaded in pipeline**

Find where `pipeline_manager` is constructed in `server.py` and add after it:
```python
_ipa_loaded: bool = False   # tracks whether IP-Adapter is injected into current pipe
```

- [ ] **Step 3: Add the 3 new endpoints** — add after the LoRA endpoints

```python
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
    global _ipa_loaded

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

        while True:
            evt = await queue.get()
            yield f"data: {json.dumps(evt)}\n\n"
            if evt["type"] in ("done", "error"):
                break

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.delete("/api/ip-adapter")
async def ipa_delete():
    """Unload from pipeline and optionally delete weights."""
    global _ipa_loaded
    from core.ip_adapter_flux import IP_ADAPTER_FILE
    if _ipa_loaded and pipeline_manager.pipe is not None:
        ipa_unload(pipeline_manager.pipe)
        _ipa_loaded = False
    if IP_ADAPTER_FILE.exists():
        IP_ADAPTER_FILE.unlink()
    return {"status": "deleted"}
```

- [ ] **Step 4: Extend `GenerateRequest` with IP-Adapter fields**

Find `class GenerateRequest(BaseModel)` and add:
```python
ip_adapter_image_ids: list[str] = Field(default_factory=list)
ip_adapter_scales:    list[float] = Field(default_factory=list)
ip_adapter_enabled:   bool = False
```

- [ ] **Step 5: In the `/api/generate` handler, load IP-Adapter images and pass them**

Find where `input_images` is loaded from `req.input_image_ids` in the generate handler, and add after it:

```python
# IP-Adapter reference images
ip_adapter_images = None
if req.ip_adapter_enabled and req.ip_adapter_image_ids:
    ip_adapter_images = [_load_pil(fid) for fid in req.ip_adapter_image_ids]
    # Load IP-Adapter into pipeline if not already loaded
    global _ipa_loaded
    if not _ipa_loaded and pipeline_manager.pipe is not None:
        ipa_load(pipeline_manager.pipe)
        _ipa_loaded = True

params["ip_adapter_images"]  = ip_adapter_images
params["ip_adapter_scales"]  = req.ip_adapter_scales or [0.6] * len(req.ip_adapter_image_ids)
params["ip_adapter_enabled"] = req.ip_adapter_enabled
```

- [ ] **Step 6: Restart server and verify new endpoints respond**

```bash
curl http://localhost:7860/api/ip-adapter/status
```
Expected: `{"downloaded": false, "loaded": false}`

- [ ] **Step 7: Commit**

```bash
git add server.py
git commit -m "feat: server.py — IP-Adapter status/download/delete endpoints + GenerateRequest fields"
```

---

### Task 4: Wire IP-Adapter into `app.generate_image()`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add new params to `generate_image()` signature**

Find the `def generate_image(` signature and add to the end of params:
```python
ip_adapter_images: list | None = None,
ip_adapter_scales: list | None = None,
ip_adapter_enabled: bool = False,
```

- [ ] **Step 2: Before FLUX pipeline calls, inject IP-Adapter scale**

Find the section in `generate_image()` just before the FLUX `pipe(...)` call (around line 1120) and add:

```python
# IP-Adapter: set scale before each call, clear after
_ipa_active = ip_adapter_enabled and ip_adapter_images
if _ipa_active:
    from core.ip_adapter_flux import set_scale as _ipa_set_scale
    scales = ip_adapter_scales or [0.6] * len(ip_adapter_images)
    _ipa_set_scale(pipe, scales)
```

- [ ] **Step 3: Add `ip_adapter_image` kwarg to FLUX text-to-image call**

In the main FLUX `pipe(...)` call, add:
```python
**({"ip_adapter_image": ip_adapter_images} if _ipa_active else {}),
```

- [ ] **Step 4: Do the same for FLUX inpainting call**

The `FluxInpaintPipeline.from_pipe(pipe)(...)` call should also get the kwarg when `_ipa_active`.

- [ ] **Step 5: Also pass through from `pipeline.py`**

In `pipeline.py`, find where `app.generate_image(**kwargs)` is called and ensure the new params flow through — they come from `params` dict so should pass automatically as `**kwargs`. Verify `params` includes these keys from `server.py`.

- [ ] **Step 6: Smoke test with curl (no real weights yet)**

```bash
curl -X POST http://localhost:7860/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","ip_adapter_enabled":false,"ip_adapter_image_ids":[],"ip_adapter_scales":[],"height":512,"width":512,"steps":4,"seed":42,"guidance":0,"device":"mps","model_choice":"Z-Image Turbo (Quantized)","model_source":"Local","input_image_ids":[],"mask_image_id":null,"lora_file":null,"lora_strength":1.0,"img_strength":0.75,"repeat_count":1,"auto_save":false,"output_dir":"","upscale_enabled":false,"upscale_model_path":"","num_frames":25,"fps":24,"mask_mode":"Crop & Composite (Fast)","outpaint_align":"center"}'
```
Expected: SSE stream with progress events, no errors.

- [ ] **Step 7: Commit**

```bash
git add app.py pipeline.py
git commit -m "feat: app.py — wire ip_adapter_images/scales/enabled into generate_image()"
```

---

## Chunk 2: Backend — FLUX LoRA with CivitAI key remapping

### Task 5: Upgrade diffusers + create `core/lora_flux2.py`

**Research findings:** The installed diffusers (`f112eab`, 2026-01-15) has 4 bugs in `Flux2KleinPipeline` LoRA loading, all fixed in PRs merged Feb 2026. Upgrading diffusers is the cleanest solution. `core/lora_flux2.py` then wraps the upgraded loader with PEFT-format pre-processing.

**4 bugs fixed by upgrading:**
1. Hardcoded block counts (48 single blocks vs 4B model's actual 20) → `KeyError`
2. `lora_down`/`lora_up` keys not normalized to `lora_A`/`lora_B`
3. No guard checks → fails for partial LoRAs (only some layers trained)
4. `base_model.model.` prefix (PEFT/fal format) not recognized

**Files:**
- Create: `core/lora_flux2.py`

- [ ] **Step 1: Upgrade diffusers to main (gets all 4 fixes)**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac
source venv/bin/activate
uv pip install git+https://github.com/huggingface/diffusers.git
```

Verify upgrade succeeded:
```bash
python -c "import diffusers; print(diffusers.__version__)"
```
Expected: version string newer than `0.32.x` (post-Feb 2026 main).

- [ ] **Step 2: Verify existing functionality still works**

```bash
python -c "from app import load_pipeline; print('import OK')"
```
Expected: `import OK` (no ImportError or breaking changes).

- [ ] **Step 3: Create `core/lora_flux2.py`**

```python
# core/lora_flux2.py
"""
LoRA loading for Flux2KleinPipeline.

The upgraded diffusers (post-Feb 2026) handles ai-toolkit and diffusers-native
LoRA formats automatically. This module adds:
  - PEFT/fal format pre-processing (base_model.model. prefix strip)
  - Friendly error messages when LoRA is truly incompatible
  - Unload helper
"""
from __future__ import annotations


def load_lora(pipe, lora_path: str, strength: float) -> str:
    """
    Load a LoRA into Flux2KleinPipeline, handling all known key formats:
      - diffusers-native  (transformer. prefix)
      - ai-toolkit        (diffusion_model. prefix + lora_A/B or lora_down/up)
      - PEFT/fal trainer  (base_model.model.diffusion_model. prefix)
      - CivitAI           (any of the above depending on trainer used)

    Returns a status string for display in the UI.
    Raises RuntimeError with a user-friendly message if incompatible.
    """
    from safetensors.torch import load_file

    state_dict = load_file(lora_path)

    # Pre-process PEFT/fal format: strip base_model.model. prefix
    # (diffusers upgraded main handles the rest automatically)
    if any(k.startswith("base_model.model.") for k in state_dict):
        state_dict = {
            k.replace("base_model.model.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    # Unload any existing LoRA first
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    try:
        pipe.load_lora_weights(state_dict, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[strength])
    except Exception as e:
        err_str = str(e)
        if "No LoRA keys" in err_str or "KeyError" in err_str:
            raise RuntimeError(
                "LoRA not compatible with FLUX.2-klein. "
                "Try a LoRA trained for FLUX.2-klein or standard FLUX.1. "
                f"(Detail: {err_str[:120]})"
            )
        raise

    lora_name = lora_path.split("/")[-1]
    return f"Loaded LoRA: {lora_name} (strength {strength:.2f})"


def unload_lora(pipe) -> str:
    """Unload LoRA from pipeline. Safe to call even if none loaded."""
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    return "LoRA unloaded"
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "from core.lora_flux2 import load_lora, unload_lora; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add core/lora_flux2.py
git commit -m "feat: upgrade diffusers to main + core/lora_flux2.py — FLUX.2-klein LoRA loader (fixes 4 bugs)"
```

---

### Task 6: Extend `app.py` LoRA loading to support FLUX models

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Locate `load_lora()` function in app.py** (~line 686)

Currently it returns early for non-Z-Image-Full models:
```python
if "Z-Image" not in str(current_model) or "Full" not in str(current_model):
    return "LoRA only supported with Z-Image Full model"
```

- [ ] **Step 2: Replace that guard with model-aware dispatch**

```python
def load_lora(lora_file, lora_strength: float, device: str):
    """Load or update LoRA adapter. Supports Z-Image Full and all FLUX.2-klein models."""
    global current_lora_path, pipe

    is_flux   = current_model is not None and "FLUX" in str(current_model)
    is_zimage = current_model is not None and "Full" in str(current_model) and "Z-Image" in str(current_model)

    if not is_flux and not is_zimage:
        return "LoRA requires FLUX.2-klein or Z-Image Full model"

    if lora_file is None or lora_file == "":
        if current_lora_path is not None:
            if is_flux:
                from core.lora_flux2 import unload_lora as _flux_unload
                _flux_unload(pipe)
            else:
                pipe.unload_lora_weights()
            current_lora_path = None
        return "No LoRA loaded"

    lora_path = lora_file if isinstance(lora_file, str) else lora_file.name

    if not os.path.exists(lora_path):
        return f"LoRA file not found: {lora_path}"

    if not lora_path.endswith('.safetensors'):
        return "Only .safetensors LoRA files are supported"

    if is_flux:
        from core.lora_flux2 import load_lora as _flux_load_lora
        try:
            status = _flux_load_lora(pipe, lora_path, lora_strength)
            current_lora_path = lora_path
            return status
        except RuntimeError as e:
            return f"LoRA error: {e}"
    else:
        # Original Z-Image path
        if current_lora_path == lora_path:
            pipe.set_adapters(["default"], adapter_weights=[lora_strength])
            return f"Updated LoRA strength to {lora_strength}"
        if current_lora_path is not None:
            pipe.unload_lora_weights()
        lora_name = os.path.basename(lora_path)
        pipe.load_lora_weights(lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        current_lora_path = lora_path
        return f"Loaded LoRA: {lora_name} (strength {lora_strength})"
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import app; print('OK')"
```
Expected: `OK` (may print model loading messages, no ImportError/SyntaxError)

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: app.py — extend load_lora() to support FLUX.2-klein models via core/lora_flux2"
```

---

## Chunk 3: Frontend — Types, Store, API

### Task 7: Extend `types.ts` with IP-Adapter types

**Files:**
- Modify: `frontend/src/types.ts`

- [ ] **Step 1: Add `IpAdapterSlot` and `IpAdapterStatus`, extend `GenerateParams`**

Add at the end of `types.ts`:
```typescript
/** One IP-Adapter reference image slot */
export interface IpAdapterSlot {
  slotId:   number       // 1-based
  imageId:  string       // temp upload id
  imageUrl: string       // preview URL
  scale:    number       // per-image strength (0–1)
}

/** Status of IP-Adapter weights on disk / in pipeline */
export interface IpAdapterStatus {
  downloaded:  boolean
  loaded:      boolean
}
```

Add to `GenerateParams` interface:
```typescript
ip_adapter_image_ids: string[]
ip_adapter_scales:    number[]
ip_adapter_enabled:   boolean
```

- [ ] **Step 2: Add IP-Adapter fields to `DEFAULT_PARAMS` in `store.ts`**

```typescript
ip_adapter_image_ids: [],
ip_adapter_scales:    [],
ip_adapter_enabled:   false,
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend
npm run build 2>&1 | tail -5
```
Expected: build succeeds (or only pre-existing errors, no new ones).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/src/store.ts
git commit -m "feat: types/store — add IpAdapterSlot, IpAdapterStatus, GenerateParams IP-Adapter fields"
```

---

### Task 8: Add IP-Adapter state and actions to `store.ts`

**Files:**
- Modify: `frontend/src/store.ts`

- [ ] **Step 1: Add IP-Adapter state to `State` interface**

```typescript
// IP-Adapter
ipAdapterSlots:  IpAdapterSlot[]
ipAdapterEnabled: boolean
ipAdapterStatus: IpAdapterStatus | null
```

- [ ] **Step 2: Add to `initialState`**

```typescript
ipAdapterSlots:   [],
ipAdapterEnabled: false,
ipAdapterStatus:  null,
```

- [ ] **Step 3: Add actions to the `Action` union**

```typescript
| { type: 'ADD_IPA_SLOT';    imageId: string; imageUrl: string }
| { type: 'REMOVE_IPA_SLOT'; slotId: number }
| { type: 'UPDATE_IPA_SCALE'; slotId: number; scale: number }
| { type: 'CLEAR_IPA_SLOTS' }
| { type: 'TOGGLE_IPA' }
| { type: 'SET_IPA_STATUS';  status: IpAdapterStatus }
```

- [ ] **Step 4: Add helper to sync IPA slots → params**

```typescript
function ipaSlotsToParams(slots: IpAdapterSlot[], enabled: boolean):
  Pick<GenerateParams, 'ip_adapter_image_ids' | 'ip_adapter_scales' | 'ip_adapter_enabled'> {
  return {
    ip_adapter_image_ids: slots.map(s => s.imageId),
    ip_adapter_scales:    slots.map(s => s.scale),
    ip_adapter_enabled:   enabled && slots.length > 0,
  }
}
```

- [ ] **Step 5: Add cases to reducer**

```typescript
case 'ADD_IPA_SLOT': {
  if (state.ipAdapterSlots.length >= 3) return state  // max 3
  const slot: IpAdapterSlot = {
    slotId:   state.ipAdapterSlots.length + 1,
    imageId:  action.imageId,
    imageUrl: action.imageUrl,
    scale:    0.6,
  }
  const slots = [...state.ipAdapterSlots, slot]
  return { ...state, ipAdapterSlots: slots,
    params: { ...state.params, ...ipaSlotsToParams(slots, state.ipAdapterEnabled) } }
}

case 'REMOVE_IPA_SLOT': {
  const slots = state.ipAdapterSlots
    .filter(s => s.slotId !== action.slotId)
    .map((s, i) => ({ ...s, slotId: i + 1 }))
  return { ...state, ipAdapterSlots: slots,
    params: { ...state.params, ...ipaSlotsToParams(slots, state.ipAdapterEnabled) } }
}

case 'UPDATE_IPA_SCALE': {
  const slots = state.ipAdapterSlots.map(s =>
    s.slotId === action.slotId ? { ...s, scale: action.scale } : s)
  return { ...state, ipAdapterSlots: slots,
    params: { ...state.params, ...ipaSlotsToParams(slots, state.ipAdapterEnabled) } }
}

case 'CLEAR_IPA_SLOTS':
  return { ...state, ipAdapterSlots: [],
    params: { ...state.params, ...ipaSlotsToParams([], state.ipAdapterEnabled) } }

case 'TOGGLE_IPA': {
  const enabled = !state.ipAdapterEnabled
  return { ...state, ipAdapterEnabled: enabled,
    params: { ...state.params, ...ipaSlotsToParams(state.ipAdapterSlots, enabled) } }
}

case 'SET_IPA_STATUS':
  return { ...state, ipAdapterStatus: action.status }
```

- [ ] **Step 6: Build and verify**

```bash
npm run build 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/store.ts
git commit -m "feat: store — IP-Adapter state, actions, and reducer cases"
```

---

### Task 9: Add IP-Adapter API calls to `api.ts`

**Files:**
- Modify: `frontend/src/api.ts`

- [ ] **Step 1: Add the three IP-Adapter API functions**

```typescript
// ── IP-Adapter ────────────────────────────────────────────────────────────────

export const fetchIpAdapterStatus = () =>
  get<{ downloaded: boolean; loaded: boolean }>('/api/ip-adapter/status')

export const deleteIpAdapter = () =>
  del<{ status: string }>('/api/ip-adapter')

/**
 * Stream IP-Adapter weight download progress.
 * Calls onEvent for each SSE event; resolves when done or rejects on error.
 */
export async function streamIpAdapterDownload(
  onEvent: (e: { type: string; pct?: number; message?: string }) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch('/api/ip-adapter/download', { method: 'POST', signal })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail ?? `Download failed: ${r.status}`)
  }
  const reader  = r.body!.getReader()
  const decoder = new TextDecoder()
  let   buf     = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const ev = JSON.parse(line.slice(6))
          onEvent(ev)
          if (ev.type === 'done' || ev.type === 'error') return
        } catch { /* ignore malformed */ }
      }
    }
  }
}
```

- [ ] **Step 2: Build and verify**

```bash
npm run build 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat: api.ts — IP-Adapter status, download stream, delete endpoints"
```

---

## Chunk 4: Frontend — IP-Adapter UI Panel + LoRA Accordion Extension

### Task 10: Create `IpAdapterPanel.tsx`

**Files:**
- Create: `frontend/src/components/IpAdapterPanel.tsx`

The panel has 3 states: not-downloaded, downloading, ready.
Max 3 image slots. Each slot: thumbnail + scale slider + remove button.
Drop zone accepts both file-picker and drag-and-drop (from gallery, Task 15).

- [ ] **Step 1: Create the component**

```tsx
// frontend/src/components/IpAdapterPanel.tsx
import { useRef, useState, useCallback } from 'react'
import type { IpAdapterSlot, IpAdapterStatus } from '../types'
import type { Action } from '../store'
import { uploadImage, streamIpAdapterDownload, fetchIpAdapterStatus } from '../api'
import HelpTip from './HelpTip'

interface Props {
  slots:    IpAdapterSlot[]
  enabled:  boolean
  status:   IpAdapterStatus | null
  dispatch: (a: Action) => void
}

export default function IpAdapterPanel({ slots, enabled, status, dispatch }: Props) {
  const fileRef   = useRef<HTMLInputElement>(null)
  const [downloading, setDownloading] = useState(false)
  const [dlPct,       setDlPct]       = useState(0)
  const [dlMsg,       setDlMsg]       = useState('')
  const [dragOver,    setDragOver]    = useState<number | null>(null)  // slotId or -1 for new slot

  const downloaded = status?.downloaded ?? false

  // ── Download weights ───────────────────────────────────────────────────────
  async function handleDownload() {
    setDownloading(true)
    setDlPct(0)
    setDlMsg('Starting download…')
    try {
      await streamIpAdapterDownload(ev => {
        if (ev.type === 'progress') { setDlPct(ev.pct ?? 0); setDlMsg(ev.message ?? '') }
        if (ev.type === 'done')     { setDlMsg('Download complete!') }
        if (ev.type === 'error')    { setDlMsg(`Error: ${ev.message}`) }
      })
      // Refresh status
      const s = await fetchIpAdapterStatus()
      dispatch({ type: 'SET_IPA_STATUS', status: s })
    } catch (e: unknown) {
      setDlMsg(`Error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setDownloading(false)
    }
  }

  // ── Upload image to a slot ─────────────────────────────────────────────────
  const handleFile = useCallback(async (file: File, targetSlotId?: number) => {
    if (slots.length >= 3 && targetSlotId === undefined) return
    try {
      const { id, url } = await uploadImage(file)
      if (targetSlotId !== undefined) {
        // Replace existing slot (not implemented yet — for now just add)
      }
      dispatch({ type: 'ADD_IPA_SLOT', imageId: id, imageUrl: url })
    } catch (e) {
      console.error('IPA upload failed', e)
    }
  }, [slots.length, dispatch])

  // ── Drag and drop ──────────────────────────────────────────────────────────
  function onDragOver(e: React.DragEvent, slotId: number) {
    e.preventDefault()
    setDragOver(slotId)
  }
  function onDragLeave() { setDragOver(null) }
  async function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(null)
    // Accept file drop
    const file = e.dataTransfer.files[0]
    if (file) { await handleFile(file); return }
    // Accept gallery drag (URL transfer)
    const url = e.dataTransfer.getData('text/plain')
    if (url) {
      const r    = await fetch(url)
      const blob = await r.blob()
      await handleFile(new File([blob], 'ref.png', { type: blob.type || 'image/png' }))
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-3">

      {/* Not downloaded state */}
      {!downloaded && !downloading && (
        <div className="space-y-2">
          <p className="text-xs text-[var(--color-muted)]">
            Weights not installed (~7 GB total including image encoder)
          </p>
          <button
            onClick={handleDownload}
            className="w-full py-2 rounded bg-[var(--color-accent)] text-white text-sm font-medium hover:opacity-90"
          >
            Download IP-Adapter weights
          </button>
          <HelpTip text="IP-Adapter lets you guide generation using a reference photo — useful for matching a face, artistic style, color mood, or object appearance. Downloaded once, stored in ./models/ip_adapter/." />
        </div>
      )}

      {/* Downloading state */}
      {downloading && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-[var(--color-muted)]">
            <span>{dlMsg}</span>
            <span>{dlPct}%</span>
          </div>
          <div className="h-1 bg-[var(--color-border)] rounded-full overflow-hidden">
            <div className="h-full bg-[var(--color-accent)] transition-all" style={{ width: `${dlPct}%` }} />
          </div>
        </div>
      )}

      {/* Ready state */}
      {downloaded && !downloading && (
        <div className="space-y-3">
          {/* Enable toggle */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-[var(--color-label)]">Enable</span>
              <HelpTip text="When enabled, IP-Adapter injects visual features from your reference images into every generation. Disable to generate without it." />
            </div>
            <button
              onClick={() => dispatch({ type: 'TOGGLE_IPA' })}
              className={`relative w-10 h-5 rounded-full transition-colors ${enabled ? 'bg-[var(--color-accent)]' : 'bg-[var(--color-border)]'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${enabled ? 'translate-x-5' : ''}`} />
            </button>
          </div>

          {/* Reference image slots */}
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <span className="text-xs text-[var(--color-label)]">Reference images</span>
              <HelpTip text="Up to 3 reference images. Their visual features are blended together. Higher scale = stronger influence from that image. 0.4–0.7 recommended." />
            </div>
            <div className="flex gap-2 flex-wrap">
              {slots.map(slot => (
                <div key={slot.slotId} className="flex flex-col items-center gap-1">
                  <div
                    className={`relative w-16 h-16 rounded border-2 ${dragOver === slot.slotId ? 'border-[var(--color-accent)]' : 'border-[var(--color-border)]'} overflow-hidden bg-[var(--color-surface)]`}
                    onDragOver={e => onDragOver(e, slot.slotId)}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                  >
                    <img src={slot.imageUrl} className="w-full h-full object-cover" alt="" />
                    <button
                      onClick={() => dispatch({ type: 'REMOVE_IPA_SLOT', slotId: slot.slotId })}
                      className="absolute top-0.5 right-0.5 w-4 h-4 rounded-full bg-black/60 text-white text-[10px] flex items-center justify-center hover:bg-black"
                    >×</button>
                  </div>
                  {/* Per-slot scale slider */}
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={slot.scale}
                    onChange={e => dispatch({ type: 'UPDATE_IPA_SCALE', slotId: slot.slotId, scale: parseFloat(e.target.value) })}
                    className="w-16 accent-[var(--color-accent)]"
                  />
                  <span className="text-[10px] text-[var(--color-muted)]">{slot.scale.toFixed(2)}</span>
                </div>
              ))}

              {/* Add slot button (max 3) */}
              {slots.length < 3 && (
                <div
                  className={`w-16 h-16 rounded border-2 border-dashed ${dragOver === -1 ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border)]'} flex items-center justify-center cursor-pointer hover:border-[var(--color-accent)] transition-colors`}
                  onClick={() => fileRef.current?.click()}
                  onDragOver={e => onDragOver(e, -1)}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                >
                  <span className="text-[var(--color-muted)] text-xl">+</span>
                </div>
              )}
            </div>
          </div>

          {slots.length > 0 && (
            <button
              onClick={() => dispatch({ type: 'CLEAR_IPA_SLOTS' })}
              className="text-xs text-[var(--color-muted)] hover:text-white transition-colors"
            >
              Clear all
            </button>
          )}
        </div>
      )}

      <input
        ref={fileRef} type="file" accept="image/*" className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = '' }}
      />
    </div>
  )
}
```

- [ ] **Step 2: Build and verify**

```bash
npm run build 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/IpAdapterPanel.tsx
git commit -m "feat: IpAdapterPanel — download, enable toggle, 3-slot reference image UI"
```

---

### Task 11: Wire `IpAdapterPanel` into `Sidebar.tsx` and extend LoRA accordion

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Import `IpAdapterPanel` and `fetchIpAdapterStatus` in Sidebar**

```tsx
import IpAdapterPanel from './IpAdapterPanel'
import { fetchIpAdapterStatus } from '../api'
```

- [ ] **Step 2: Add IP-Adapter status poll on mount**

Inside the Sidebar component, add a `useEffect` to fetch IP-Adapter status on mount:
```tsx
useEffect(() => {
  fetchIpAdapterStatus()
    .then(s => dispatch({ type: 'SET_IPA_STATUS', status: s }))
    .catch(() => {})
}, [])
```

- [ ] **Step 3: Add IP-Adapter accordion section**

Add a new accordion item (following the existing pattern for LoRA/Upscale/Video accordions):
```tsx
{/* IP-Adapter */}
<AccordionItem title="IP-Adapter" defaultOpen={false}>
  <IpAdapterPanel
    slots={state.ipAdapterSlots}
    enabled={state.ipAdapterEnabled}
    status={state.ipAdapterStatus}
    dispatch={dispatch}
  />
</AccordionItem>
```
Place it after the LoRA accordion and before Upscale.

- [ ] **Step 4: Extend LoRA accordion to show for FLUX models**

Find the condition that hides the LoRA accordion for non-Z-Image-Full models. It currently reads something like:
```tsx
{isZImageFull && <AccordionItem title="LoRA">…</AccordionItem>}
```
Change to:
```tsx
{(isZImageFull || isFlux) && <AccordionItem title="LoRA">…</AccordionItem>}
```
Where `isFlux` is `state.params.model_choice.includes('FLUX')`.

- [ ] **Step 5: Add LoRA compatibility warning for FLUX**

Inside the LoRA accordion, after the upload button, add when `isFlux && !isZImageFull`:
```tsx
{isFlux && (
  <HelpTip text="LoRAs trained for standard FLUX.1 may not be compatible with FLUX.2-klein. If loading fails, try a LoRA specifically trained for FLUX.2-klein. Find compatible ones at huggingface.co/linoyts/Flux2-Klein-Delight-LoRA" />
)}
```

- [ ] **Step 6: Build, start dev server, verify accordion appears**

```bash
npm run build && echo "Build OK"
```
Then open the app in browser: IP-Adapter accordion should appear in sidebar, with "Download" button since weights aren't installed.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: Sidebar — add IpAdapterPanel accordion, extend LoRA to FLUX models"
```

---

### Task 12: Bootstrap IP-Adapter status in `App.tsx`

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: On bootstrap / model load, re-fetch IP-Adapter status**

Find the bootstrap `useEffect` in App.tsx and add alongside other status fetches:
```tsx
fetchIpAdapterStatus()
  .then(s => dispatch({ type: 'SET_IPA_STATUS', status: s }))
  .catch(() => {})
```

- [ ] **Step 2: Build final verification**

```bash
npm run build 2>&1 | grep -E "error|warning|built" | tail -10
```

- [ ] **Step 3: End-to-end manual test**

1. Open app at `http://localhost:7860`
2. Open IP-Adapter accordion — should show "Download" button
3. Click Download — should stream progress (will actually download ~7 GB on real hardware)
4. After download, toggle Enable — should activate
5. Drop an image into a slot — thumbnail appears, scale slider visible
6. Generate — request should include `ip_adapter_enabled: true`, `ip_adapter_image_ids: [...]`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: App.tsx — bootstrap IP-Adapter status on startup"
```

---

## Chunk 5: Integration + LoRA research incorporation

### Task 13: Verify LoRA loading with real CivitAI LoRA (if available)

Research is complete and incorporated into Task 5 (diffusers upgrade + `core/lora_flux2.py`).
Supported formats confirmed:

| Source | Key prefix | lora key names | Status after upgrade |
|--------|-----------|----------------|---------------------|
| ai-toolkit (new) | `diffusion_model.` | `lora_A/B` | ✅ Fixed by PR #13030 + #13119 |
| ai-toolkit (old) | `diffusion_model.` | `lora_down/up` | ✅ Fixed by PR #13119 |
| PEFT/fal trainer | `base_model.model.diffusion_model.` | either | ✅ Fixed by PR #13169 + our pre-process |
| CivitAI (varies) | any of above | either | ✅ All covered |
| diffusers DreamBooth | `transformer.` | `lora_A/B` | ✅ Always worked |

- [ ] **Step 1: If a CivitAI FLUX LoRA is available, test loading it**

```bash
python -c "
import app
# Load a FLUX model first (must be loaded before testing LoRA)
# Then test:
from core.lora_flux2 import load_lora
# status = load_lora(app.pipe, '/path/to/civitai.safetensors', 0.8)
# print(status)
print('lora_flux2 import OK')
"
```

- [ ] **Step 2: Verify the error message when an incompatible LoRA is passed**

The `RuntimeError` in `load_lora()` should surface a user-friendly message in the UI info bar.

---

### Task 14: Final build + manual smoke test

- [ ] **Step 1: Build frontend**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npm run build
```

- [ ] **Step 2: Start server**

```bash
source venv/bin/activate && python server.py --port 7860 --no-auto-shutdown
```

- [ ] **Step 3: Verify all new endpoints**

```bash
curl http://localhost:7860/api/ip-adapter/status
# Expected: {"downloaded": false, "loaded": false}
```

- [ ] **Step 4: Verify LoRA accordion shows for FLUX models**

Load a FLUX model in the UI, check sidebar — LoRA accordion should appear.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: IP-Adapter + FLUX LoRA — final integration verified"
```
