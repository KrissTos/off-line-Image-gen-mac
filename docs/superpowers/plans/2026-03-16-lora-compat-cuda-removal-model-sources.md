# LoRA Compat Check, CUDA Removal, Model Sources Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate LoRA files on upload, strip all CUDA code paths, and add a curated Model Sources registry to the Settings drawer.

**Architecture:** Three independent tasks. Task 1 adds a header-only safetensors key check before saving LoRA files and surfaces errors through the existing 422 → UI error flow. Task 2 is a mechanical search-and-remove of all `torch.cuda.*` branches in Python files. Task 3 adds a new `model_sources.json` file backed by two REST endpoints and a new Settings section in the React frontend.

**Tech Stack:** Python 3 / FastAPI / safetensors · React + TypeScript + Tailwind CSS v3 · Lucide React icons

**Spec:** `docs/superpowers/specs/2026-03-16-lora-compat-cuda-removal-model-sources.md`

---

## Chunk 1: LoRA Compatibility Check on Upload

### Task 1: Add `check_lora_compatibility()` to `core/lora_flux2.py`

**Files:**
- Modify: `core/lora_flux2.py`

FLUX.2-klein has 20 single blocks (indices 0–19) and 19 double blocks (indices 0–18). A LoRA trained for full FLUX will have keys like `single_blocks.24.…` — those must be rejected at upload time.

The check reads only the safetensors **header** (JSON metadata at file start) via `safetensors.safe_open` — no tensor data is loaded into memory.

- [ ] **Add the function** at the top of `core/lora_flux2.py`, before `load_lora`:

```python
def check_lora_compatibility(path: str) -> None:
    """
    Validate that a LoRA file is compatible with FLUX.2-klein before saving.
    Reads only the safetensors header (no tensor data loaded).

    Raises RuntimeError with a user-facing message if incompatible.
    """
    from safetensors import safe_open
    import re

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception as e:
        raise RuntimeError(f"Could not read LoRA file: {e}")

    if not keys:
        raise RuntimeError("LoRA file appears to be empty.")

    # Normalise all key prefixes to bare block names for uniform checking
    # Handles: diffusion_model.*, base_model.model.diffusion_model.*, transformer.*
    def _normalise(k: str) -> str:
        k = re.sub(r'^base_model\.model\.', '', k)
        k = re.sub(r'^diffusion_model\.', '', k)
        k = re.sub(r'^transformer\.single_transformer_blocks\.', 'single_blocks.', k)
        k = re.sub(r'^transformer\.transformer_blocks\.', 'double_blocks.', k)
        return k

    normalised = [_normalise(k) for k in keys]

    # Detect FLUX keys at all
    flux_keys = [k for k in normalised if k.startswith(('single_blocks.', 'double_blocks.'))]
    if not flux_keys:
        raise RuntimeError(
            "No FLUX LoRA keys found — this may be a Stable Diffusion or other format LoRA."
        )

    # Check block index bounds for FLUX.2-klein
    single_re = re.compile(r'^single_blocks\.(\d+)\.')
    double_re = re.compile(r'^double_blocks\.(\d+)\.')

    for k in normalised:
        m = single_re.match(k)
        if m and int(m.group(1)) >= 20:
            raise RuntimeError(
                f"LoRA not compatible with FLUX.2-klein — trained for a larger model "
                f"(found single_blocks.{m.group(1)}, klein has 20). "
                f"Use a LoRA trained for FLUX.2-klein 4B or 9B."
            )
        m = double_re.match(k)
        if m and int(m.group(1)) >= 19:
            raise RuntimeError(
                f"LoRA not compatible with FLUX.2-klein — trained for a larger model "
                f"(found double_blocks.{m.group(1)}, klein has 19). "
                f"Use a LoRA trained for FLUX.2-klein 4B or 9B."
            )
```

- [ ] **Commit**

```bash
git add core/lora_flux2.py
git commit -m "feat: add check_lora_compatibility() — header-only FLUX.2-klein block check"
```

---

### Task 2: Call the check in `server.py` upload endpoint

**Files:**
- Modify: `server.py` — `api_upload_lora` function (~line 904)

Current flow saves the file then returns. New flow: save to a temp path → check → rename to final path (or delete on failure → 422).

- [ ] **Replace `api_upload_lora`** (find it by the `@app.post("/api/lora/upload")` decorator):

```python
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
```

- [ ] **Verify manually** — start the server and try uploading an incompatible LoRA (one trained for FLUX.1 full, not klein):

```bash
curl -s -X POST http://localhost:7860/api/lora/upload \
  -F "file=@/path/to/incompatible_lora.safetensors" | python3 -m json.tool
# Expected: {"detail": "LoRA not compatible with FLUX.2-klein ..."}
# HTTP status should be 422
```

For a compatible klein LoRA, it should return `{"path": "...", "name": "..."}`.

- [ ] **Commit**

```bash
git add server.py
git commit -m "feat: reject incompatible LoRAs at upload with 422 and user-friendly message"
```

---

### Task 3: Surface the 422 detail in the frontend

**Files:**
- Modify: `frontend/src/api.ts` — `uploadLora` function (~line 107)

Currently `if (!r.ok) throw new Error(\`LoRA upload failed: ${r.status}\`)` discards the `detail` field from the 422 JSON body. The user sees `"LoRA upload failed: 422"` instead of the actual message.

- [ ] **Replace `uploadLora`**:

```typescript
export async function uploadLora(file: File): Promise<{ path: string; name: string }> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/api/lora/upload', { method: 'POST', body: fd })
  if (!r.ok) {
    let detail = `LoRA upload failed: ${r.status}`
    try {
      const body = await r.json()
      if (body?.detail) detail = body.detail
    } catch { /* ignore parse errors */ }
    throw new Error(detail)
  }
  return r.json()
}
```

- [ ] **Build the frontend**

```bash
cd frontend && npm run build
```

Expected: no TypeScript errors, build succeeds.

- [ ] **Manual smoke test** — open the app in the browser, try uploading an incompatible LoRA via the LoRA panel. The error toast should now show the actual incompatibility message instead of `"LoRA upload failed: 422"`.

- [ ] **Commit**

```bash
git add frontend/src/api.ts frontend/dist
git commit -m "fix: surface LoRA upload 422 detail message in UI instead of raw status code"
```

---

## Chunk 2: CUDA Removal

### Task 4: Remove CUDA branches from `app.py`

**Files:**
- Modify: `app.py`

No test framework exists in this project. Verify by grepping for remaining cuda references after edits.

Work through each location in order. All changes are find-and-replace; line numbers are approximate — search by content.

- [ ] **Module docstring** — find `"Fast image generation on Apple Silicon and CUDA."` (near line 4), change to:
  ```python
  "Offline AI image generation for Mac Silicon (MPS)."
  ```

- [ ] **Device detection** — find the block:
  ```python
  if torch.cuda.is_available():
      devices.append("cuda")
  ```
  Delete it entirely. The `devices` list should only contain `"mps"` (if available) and `"cpu"`.

- [ ] **FLUX/LTX dtype** — find all occurrences of:
  ```python
  dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
  ```
  Replace each with:
  ```python
  dtype = torch.bfloat16 if device == "mps" else torch.float32
  ```
  There are two: one in the FLUX.2-klein load path (~line 414) and one in the LTX-Video load path (~line 741).

- [ ] **Z-Image quantized dtype** — find:
  ```python
  dtype = torch.float16 if device == "cuda" else torch.float32
  ```
  Replace with:
  ```python
  dtype = torch.float32
  ```

- [ ] **VRAM reporting** — find the block that calls `torch.cuda.memory_allocated()` inside a `torch.cuda.is_available()` guard. Delete the entire `if torch.cuda.is_available(): ... torch.cuda.memory_allocated()` branch. The MPS VRAM path (using `psutil` or `subprocess`) should remain.

- [ ] **`torch.cuda.empty_cache()` and `torch.cuda.synchronize()` calls** — search for all occurrences:
  ```bash
  grep -n "torch.cuda.empty_cache\|torch.cuda.synchronize" app.py
  ```
  There are approximately 5 locations. For each, delete the **entire enclosing `if torch.cuda.is_available():` block** — not just the inner call. For example the block around lines 1257–1259 likely looks like:
  ```python
  elif torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.synchronize()
  ```
  Delete the entire `elif` (or `if`) block including its body in one step — do not delete lines individually or you'll leave an empty `elif:` header.

- [ ] **Generator device** — find the entire block (approximately lines 1053–1058):
  ```python
  if device == "cuda":
      generator = torch.Generator("cuda").manual_seed(current_seed)
  elif device == "mps":
      generator = torch.Generator("mps").manual_seed(current_seed)
  else:
      generator = torch.Generator().manual_seed(current_seed)
  ```
  Delete the `if device == "cuda":` branch entirely and convert the `elif` to `if`:
  ```python
  if device == "mps":
      generator = torch.Generator("mps").manual_seed(current_seed)
  else:
      generator = torch.Generator().manual_seed(current_seed)
  ```
  Note: `torch.Generator("cpu")` (the `else` branch default) is correct for MPS seeding too.

- [ ] **Upscale device fallback** — find:
  ```python
  "cuda" if torch.cuda.is_available() else "cpu"
  ```
  Replace with:
  ```python
  "cpu"
  ```
  (spandrel's MPS fallback is already implemented separately — this is just the initial device hint)

- [ ] **Verify no cuda references remain**:
  ```bash
  grep -n "cuda" app.py
  ```
  Expected output: zero matches. If any remain, fix them.

- [ ] **Commit**

```bash
git add app.py
git commit -m "refactor: remove all CUDA code paths from app.py — Mac Silicon (MPS) only"
```

---

### Task 5: Remove CUDA branches from `generate.py`

**Files:**
- Modify: `generate.py`

- [ ] **dtype** — find:
  ```python
  dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
  ```
  Replace with:
  ```python
  dtype = torch.bfloat16 if device == "mps" else torch.float32
  ```

- [ ] **Generator** — find the block around lines 79–80. It may look like:
  ```python
  if device == "cuda":
      generator = torch.Generator("cuda").manual_seed(seed)
  ```
  Delete the entire `if device == "cuda":` block. If an `elif device == "mps":` follows, convert it to `if`. The correct result is that a CPU or MPS generator is used and no cuda branch remains.

- [ ] **argparse help** — find `"mps, cuda, cpu"` in the `--device` argument help string, change to `"mps, cpu"`.

- [ ] **Auto-detection** — find:
  ```python
  elif torch.cuda.is_available():
      device = "cuda"
  ```
  Delete this `elif` block. The auto-detection chain should be: MPS available → `"mps"`, otherwise → `"cpu"`.

- [ ] **Validation** — find:
  ```python
  elif device == "cuda" and not torch.cuda.is_available():
      print("WARNING: CUDA not available")
  ```
  Delete this block. Preserve the final `else: print("Invalid device …"); sys.exit(1)` branch — it still correctly rejects any unknown device string.

- [ ] **Verify no cuda references remain**:
  ```bash
  grep -n "cuda" generate.py
  ```
  Expected: zero matches.

- [ ] **Commit**

```bash
git add generate.py
git commit -m "refactor: remove CUDA code paths from generate.py CLI — Mac Silicon only"
```

---

### Task 6: Update `pyproject.toml` description

**Files:**
- Modify: `pyproject.toml`

- [ ] Find the `description` field and replace the current value with:
  ```toml
  description = "Offline AI image generation for Mac Silicon (MPS) — FLUX.2-klein, Z-Image Turbo, LTX-Video, LoRA support"
  ```

- [ ] **Commit**

```bash
git add pyproject.toml
git commit -m "docs: update pyproject description — Mac Silicon only, remove CUDA mention"
```

---

## Chunk 3: Model Sources Registry

### Task 7: Add backend endpoints to `server.py`

**Files:**
- Modify: `server.py`
- Create: `model_sources.json` (gitignored, seeded at runtime — do NOT create manually)
- Modify: `.gitignore`

- [ ] **Add `model_sources.json` to `.gitignore`** — append to `.gitignore`:
  ```
  model_sources.json
  ```

- [ ] **Add `DEFAULT_SOURCES` constant** to `server.py`. Find a good location near the top of the file (after imports, before route definitions). Add:

```python
# ── Model Sources Registry ─────────────────────────────────────────────────
MODEL_SOURCES_FILE = ROOT / "model_sources.json"

DEFAULT_SOURCES: list[dict] = [
    # Base models — name must match KNOWN_MODELS display name in app.py for Download to work
    {"id": "src-001", "name": "FLUX.2-klein-4B (4bit SDNQ)",  "url": "https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",           "type": "base",     "description": "~8 GB VRAM · MPS optimized · recommended for 16 GB Mac"},
    {"id": "src-002", "name": "FLUX.2-klein-9B (4bit SDNQ)",  "url": "https://huggingface.co/Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",   "type": "base",     "description": "~12 GB VRAM · high quality"},
    {"id": "src-003", "name": "FLUX.2-klein-4B (Int8)",        "url": "https://huggingface.co/aydin99/FLUX.2-klein-4B-int8",                        "type": "base",     "description": "~16 GB VRAM · MPS explicit · 4B int8"},
    {"id": "src-004", "name": "FLUX.2-klein-9B FP8",           "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8",               "type": "base",     "description": "Official BFL FP8 · not yet loadable in-app"},
    {"id": "src-005", "name": "Z-Image Turbo (Full)",           "url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",                           "type": "base",     "description": "~24 GB VRAM · LoRA support"},
    {"id": "src-006", "name": "Z-Image Turbo (Quantized)",      "url": "https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",            "type": "base",     "description": "~6 GB VRAM · fast"},
    {"id": "src-007", "name": "LTX-Video",                      "url": "https://huggingface.co/Lightricks/LTX-Video",                               "type": "base",     "description": "Official video generation model"},
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
```

- [ ] **Add the two endpoints** to `server.py`. Place them near the other settings endpoints (`/api/settings`). Note: use `def` (not `async def`) since these do synchronous file I/O:

```python
@app.get("/api/model-sources")
def api_get_model_sources():
    """Return the model sources list. Seeds from DEFAULT_SOURCES if file missing."""
    if MODEL_SOURCES_FILE.exists():
        try:
            data = json.loads(MODEL_SOURCES_FILE.read_text())
            return {"sources": data.get("sources", DEFAULT_SOURCES)}
        except Exception:
            pass
    return {"sources": DEFAULT_SOURCES}


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
    MODEL_SOURCES_FILE.write_text(json.dumps({"version": 1, "sources": sources}, indent=2))
    return {"ok": True}
```

- [ ] **Verify `json` is imported** at the top of `server.py`. It almost certainly is; if not, add `import json`.

- [ ] **Add `Body` to the FastAPI import** at the top of `server.py`. Find the line `from fastapi import FastAPI, File, HTTPException, Request, UploadFile` and add `Body` to it:
  ```python
  from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
  ```

- [ ] **Test endpoints**:

```bash
# Start server, then:
curl -s http://localhost:7860/api/model-sources | python3 -m json.tool | head -20
# Expected: {"sources": [...31 entries...]}

# Test save:
curl -s -X POST http://localhost:7860/api/model-sources \
  -H "Content-Type: application/json" \
  -d '{"sources":[{"id":"x","name":"Test","url":"https://example.com","type":"base","description":"test"}]}' \
  | python3 -m json.tool
# Expected: {"ok": true}

# Test validation:
curl -s -X POST http://localhost:7860/api/model-sources \
  -H "Content-Type: application/json" \
  -d '{"sources":[{"id":"x","name":"","url":"https://example.com","type":"base","description":""}]}' \
  | python3 -m json.tool
# Expected: {"detail": "Each source must have a non-empty name and url."}
```

- [ ] **Commit**

```bash
git add server.py .gitignore
git commit -m "feat: add GET/POST /api/model-sources endpoints with DEFAULT_SOURCES prefill"
```

---

### Task 8: Add types and API helpers to the frontend

**Files:**
- Modify: `frontend/src/api.ts`

- [ ] **Add `ModelSource` type** near the top of `api.ts` (with other interface definitions):

```typescript
export interface ModelSource {
  id:          string
  name:        string
  url:         string
  type:        'base' | 'lora' | 'upscaler'
  description: string
}
```

- [ ] **Add two API functions** at the bottom of `api.ts`:

```typescript
export async function fetchModelSources(): Promise<ModelSource[]> {
  const r = await fetch('/api/model-sources')
  if (!r.ok) throw new Error(`Failed to fetch model sources: ${r.status}`)
  const data = await r.json()
  return data.sources as ModelSource[]
}

export async function saveModelSources(sources: ModelSource[]): Promise<void> {
  const r = await fetch('/api/model-sources', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sources }),
  })
  if (!r.ok) {
    let detail = `Failed to save model sources: ${r.status}`
    try { const b = await r.json(); if (b?.detail) detail = b.detail } catch {}
    throw new Error(detail)
  }
}
```

- [ ] **Commit** (no build yet — building together with the UI changes in Task 9):

```bash
git add frontend/src/api.ts
git commit -m "feat: add fetchModelSources / saveModelSources API helpers and ModelSource type"
```

---

### Task 9: Add Model Sources section to `SettingsDrawer.tsx`

**Files:**
- Modify: `frontend/src/components/SettingsDrawer.tsx`

The new section goes between the **Storage** section and the **Server Log** section (around line 621 — after `</section>` for Storage, before the `{/* ── Server Log ── */}` comment).

- [ ] **Add imports** at the top of `SettingsDrawer.tsx`. In the existing lucide-react import, add `ExternalLink`, `Globe`. In the existing api import, add `fetchModelSources`, `saveModelSources`, `ModelSource`, `updateModel`:

```typescript
// lucide-react — add ONLY these to the existing import (Trash2, RefreshCw, Download are already there):
ExternalLink, Globe, Plus,
// api — add to existing import (updateModel is already imported — do NOT add it again):
fetchModelSources, saveModelSources, type ModelSource,
```

- [ ] **Add state variables** inside `SettingsDrawer` component, near the other state declarations:

```typescript
// Model Sources
const [sources, setSources]         = useState<ModelSource[]>([])
const [sourcesLoaded, setSourcesLoaded] = useState(false)
const [addingSource, setAddingSource]   = useState(false)
const [newSource, setNewSource]     = useState<Omit<ModelSource, 'id'>>({
  name: '', url: '', type: 'base', description: ''
})
const [downloadingSource, setDownloadingSource] = useState<string | null>(null)
```

- [ ] **Load sources on drawer open** — find the existing `useEffect` that runs when `open` changes (it fetches storage, models, etc.). Add a `fetchModelSources` call inside it:

```typescript
fetchModelSources().then(setSources).catch(() => {}).finally(() => setSourcesLoaded(true))
```

- [ ] **Add helper functions** inside the component (near `handleSaveLog` or similar):

```typescript
async function handleDeleteSource(id: string) {
  const next = sources.filter(s => s.id !== id)
  setSources(next)
  await saveModelSources(next).catch(() => {})
}

async function handleAddSource() {
  if (!newSource.name.trim() || !newSource.url.trim()) return
  const entry: ModelSource = {
    ...newSource,
    id: crypto.randomUUID(),
    name: newSource.name.trim(),
    url: newSource.url.trim(),
    description: newSource.description.trim(),
  }
  const next = [...sources, entry]
  setSources(next)
  await saveModelSources(next).catch(() => {})
  setNewSource({ name: '', url: '', type: 'base', description: '' })
  setAddingSource(false)
}

async function handleDownloadSource(source: ModelSource) {
  if (source.type !== 'base') return
  setDownloadingSource(source.id)
  try {
    await updateModel(source.name)
    setStatusMsg(`Downloading ${source.name}…`)
  } catch (e: any) {
    setStatusMsg(e.message || 'Download failed')
  } finally {
    setDownloadingSource(null)
  }
}
```

- [ ] **Add the JSX section** — insert between the Storage `</section>` and the `{/* ── Server Log ── */}` comment:

```tsx
{/* ── Model Sources ── */}
<section>
  <h3 className="text-xs font-semibold uppercase tracking-wider text-label mb-3 flex items-center gap-1.5">
    <Globe size={13} /> Model Sources
  </h3>
  <p className="text-[10px] text-muted/70 mb-3">
    Curated Mac Silicon models. User-editable — add your own sources.
  </p>

  {!sourcesLoaded ? (
    <p className="text-xs text-muted">Loading…</p>
  ) : (
    <div className="space-y-2">
      {sources.map(src => {
        const typeBadgeClass =
          src.type === 'base'     ? 'bg-blue-900/50 text-blue-300 border-blue-700/50' :
          src.type === 'lora'     ? 'bg-accent/20 text-accent border-accent/30' :
                                    'bg-teal-900/50 text-teal-300 border-teal-700/50'
        const canDownload = src.type === 'base'
        const downloadTooltip =
          src.type === 'lora'     ? 'Download from HuggingFace, then upload via the LoRA panel' :
          src.type === 'upscaler' ? 'Download from HuggingFace, then upload via the Upscale panel' :
          !KNOWN_MODELS_NAMES.has(src.name) ? 'Open HuggingFace page to download manually' : ''

        return (
          <div key={src.id} className="flex items-start gap-2 p-2 rounded-lg bg-card border border-border group">
            <span className={`shrink-0 mt-0.5 text-[9px] font-semibold px-1.5 py-0.5 rounded border uppercase tracking-wide ${typeBadgeClass}`}>
              {src.type}
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-xs text-white truncate">{src.name}</div>
              {src.description && (
                <div className="text-[10px] text-muted/70 truncate mt-0.5">{src.description}</div>
              )}
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <button
                onClick={() => window.open(src.url, '_blank')}
                title="Open HuggingFace page"
                className="p-1 rounded text-muted hover:text-white hover:bg-white/10 transition-colors"
              >
                <ExternalLink size={12} />
              </button>
              <button
                onClick={() => canDownload && KNOWN_MODELS_NAMES.has(src.name) && handleDownloadSource(src)}
                disabled={!canDownload || !KNOWN_MODELS_NAMES.has(src.name) || downloadingSource === src.id}
                title={downloadTooltip || `Download ${src.name}`}
                className={`p-1 rounded transition-colors ${
                  canDownload && KNOWN_MODELS_NAMES.has(src.name)
                    ? 'text-muted hover:text-white hover:bg-white/10'
                    : 'text-muted/30 cursor-not-allowed'
                }`}
              >
                {downloadingSource === src.id
                  ? <RefreshCw size={12} className="animate-spin" />
                  : <Download size={12} />
                }
              </button>
              <button
                onClick={() => handleDeleteSource(src.id)}
                title="Remove from list"
                className="p-1 rounded text-muted/40 hover:text-red-400 hover:bg-red-900/20 transition-colors opacity-0 group-hover:opacity-100"
              >
                <Trash2 size={11} />
              </button>
            </div>
          </div>
        )
      })}

      {/* Add new source */}
      {addingSource ? (
        <div className="p-2 rounded-lg bg-card border border-accent/40 space-y-2">
          <input
            type="text"
            placeholder="Name"
            value={newSource.name}
            onChange={e => setNewSource(s => ({ ...s, name: e.target.value }))}
            className="w-full bg-surface border border-border rounded px-2 py-1 text-xs text-white placeholder-muted focus:outline-none focus:border-accent"
          />
          <input
            type="url"
            placeholder="https://huggingface.co/..."
            value={newSource.url}
            onChange={e => setNewSource(s => ({ ...s, url: e.target.value }))}
            className="w-full bg-surface border border-border rounded px-2 py-1 text-xs text-white placeholder-muted focus:outline-none focus:border-accent"
          />
          <input
            type="text"
            placeholder="Description (optional)"
            value={newSource.description}
            onChange={e => setNewSource(s => ({ ...s, description: e.target.value }))}
            className="w-full bg-surface border border-border rounded px-2 py-1 text-xs text-white placeholder-muted focus:outline-none focus:border-accent"
          />
          <select
            value={newSource.type}
            onChange={e => setNewSource(s => ({ ...s, type: e.target.value as ModelSource['type'] }))}
            className="w-full bg-surface border border-border rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-accent"
          >
            <option value="base">Base model</option>
            <option value="lora">LoRA</option>
            <option value="upscaler">Upscaler</option>
          </select>
          <div className="flex gap-2">
            <button
              onClick={handleAddSource}
              className="flex-1 py-1 rounded text-xs font-medium bg-accent/80 hover:bg-accent text-white transition-colors"
            >
              Add
            </button>
            <button
              onClick={() => { setAddingSource(false); setNewSource({ name: '', url: '', type: 'base', description: '' }) }}
              className="flex-1 py-1 rounded text-xs font-medium bg-card border border-border text-muted hover:text-white transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setAddingSource(true)}
          className="w-full py-1.5 rounded-lg text-xs text-muted hover:text-white border border-dashed border-border hover:border-accent/50 transition-colors flex items-center justify-center gap-1.5"
        >
          <Plus size={11} /> Add source
        </button>
      )}
    </div>
  )}
</section>
```

- [ ] **Add `KNOWN_MODELS_NAMES` constant** — this is the set of model names that `POST /api/models/update` can handle. Add it as a module-level constant near the top of `SettingsDrawer.tsx`, after imports:

```typescript
// Names that match KNOWN_MODELS in app.py — these get an active Download button
const KNOWN_MODELS_NAMES = new Set([
  'FLUX.2-klein-4B (4bit SDNQ)',
  'FLUX.2-klein-9B (4bit SDNQ)',
  'FLUX.2-klein-4B (Int8)',
  'Z-Image Turbo (Full)',
  'Z-Image Turbo (Quantized)',
  'LTX-Video',
])
```

- [ ] **Verify lucide imports** — make sure `ExternalLink`, `Globe`, `Plus`, `Download` are all in the lucide-react import line. `Trash2`, `RefreshCw`, `Download` are already imported — check before adding to avoid duplicates.

- [ ] **Build the frontend**

```bash
cd frontend && npm run build
```

Fix any TypeScript errors before proceeding.

- [ ] **Manual smoke test**:
  1. Open the app → click the gear icon → Settings drawer opens
  2. Scroll to **Model Sources** section — should see ~31 entries grouped with type badges
  3. Click `↗` (ExternalLink) on any entry — opens HuggingFace in a new tab ✓
  4. For a `lora` entry, hover the Download button — tooltip should say "Download from HuggingFace…" and button should be grayed-out ✓
  5. For `FLUX.2-klein-4B (4bit SDNQ)`, the Download button should be active ✓
  6. Click **+ Add source**, fill in a test name/URL/type → click Add → entry appears in list ✓
  7. Hover a row → delete (×) button appears → click it → entry removed ✓
  8. Close and reopen Settings → custom entry still there (persisted to `model_sources.json`) ✓

- [ ] **Commit**

```bash
git add frontend/src/components/SettingsDrawer.tsx frontend/src/api.ts frontend/dist
git commit -m "feat: Model Sources registry in Settings — curated Mac Silicon model list with open/download/add/delete"
```

---

## Final Verification

- [ ] **Confirm no cuda remains in Python**:

```bash
grep -rn "cuda" app.py generate.py core/
# Expected: zero matches
```

- [ ] **Confirm LoRA compat check is wired**:

```bash
grep -n "check_lora_compatibility" server.py
# Expected: one match in api_upload_lora
```

- [ ] **Confirm model_sources.json is gitignored**:

```bash
echo "test" > model_sources.json
git status model_sources.json
# Expected: nothing (file is ignored)
rm model_sources.json
```
