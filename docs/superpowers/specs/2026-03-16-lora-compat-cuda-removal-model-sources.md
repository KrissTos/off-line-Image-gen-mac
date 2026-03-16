# Spec: LoRA Compatibility Check, CUDA Removal, Model Sources Registry

**Date:** 2026-03-16
**Status:** Approved

---

## Overview

Three independent improvements to the off-line-Image-gen-mac project:

1. **LoRA compatibility check on upload** — reject incompatible LoRA files at upload time with a clear error message, before wasting disk space
2. **CUDA removal** — strip all CUDA/nvidia code paths from `app.py` and `generate.py`; project is Mac Silicon (MPS) only
3. **Model Sources registry** — new Settings panel section listing curated HuggingFace model URLs (base models, LoRAs, upscalers) that users can browse, open, download, and edit

---

## Task 1 — LoRA Compatibility Check on Upload

### Problem

When a LoRA trained for full FLUX (25+ single blocks) is uploaded, it passes upload silently. The `KeyError` only surfaces at generation time with an unintelligible message like `'single_blocks.24.linear1.lora_A.weight'`.

Additionally, the existing error handler in `core/lora_flux2.py` checked `"KeyError" in str(e)` — which never matches because Python's `str(KeyError("key"))` returns `"'key'"` not `"KeyError: 'key'"`. This was also fixed (already applied as a hotfix before this spec).

### Design

**`core/lora_flux2.py`** — add `check_lora_compatibility(path: str) -> None`:

- Read only the safetensors **header** using `safetensors.safe_open(path, framework="pt")` + `.keys()` — this reads only the JSON metadata at the start of the file, NOT tensor data. Do NOT use `load_file()` (loads all tensors). This is fast (< 1ms for header) and low memory.
- Check for FLUX.2-klein block count violations across all known key prefixes:
  - Raw: `single_blocks.N` / `double_blocks.N`
  - ai-toolkit: `diffusion_model.single_blocks.N` / `diffusion_model.double_blocks.N`
  - PEFT/fal: `base_model.model.diffusion_model.single_blocks.N`
  - diffusers: `transformer.single_transformer_blocks.N` / `transformer.transformer_blocks.N`
  - Violation: `single_blocks.N` with N ≥ 20, or `double_blocks.N` with N ≥ 19
- If incompatible: raise `RuntimeError` with message: `"LoRA not compatible with FLUX.2-klein (trained for a larger model — found single_blocks.{N}). Use a LoRA trained for FLUX.2-klein 4B or 9B."`
- If no FLUX keys found at all: raise `RuntimeError("No FLUX LoRA keys found — this may be an SD or other format LoRA.")`
- This function can be called synchronously inside `async def api_upload_lora` — header parsing is fast enough to not need `run_in_executor`.

**`server.py`** — `api_upload_lora` endpoint:

```
1. Save file to temp path (lora_uploads/.tmp_<filename>)
2. Call check_lora_compatibility(temp_path)
3. On RuntimeError: delete temp file, return HTTP 422 {"detail": error_message}
4. On success: rename temp → final path, return {"path": ..., "name": ...}
```

This ensures no incompatible file remains on disk.

### Error UX

`api.ts` `uploadLora()` currently throws `"LoRA upload failed: 422"` without reading the body. It must be updated to extract `detail` from the 422 JSON response body and include it in the thrown error message. `LoraPanel` already displays thrown error messages — no other frontend changes needed.

---

## Task 2 — CUDA Removal

### Scope

Files modified: `app.py`, `generate.py`, `pyproject.toml`
Files NOT modified: `uv.lock` (NVIDIA packages come from torch's platform-agnostic wheel — cannot and should not be removed without switching to a Mac-specific torch index)

### Changes in `app.py`

| Location | Current | Replacement |
|----------|---------|-------------|
| Module docstring (line 4) | "Apple Silicon and CUDA" | "Apple Silicon (MPS)" |
| Device detection (lines 78–79) | `if torch.cuda.is_available(): devices.append("cuda")` | Remove entirely |
| dtype FLUX/LTX (lines 414, 741) | `if device in ["mps", "cuda"]` | `if device == "mps"` |
| dtype Z-Image quantized (line 423) | `torch.float16 if device == "cuda" else torch.float32` | `torch.float32` (remove conditional entirely) |
| VRAM reporting (lines 452–453) | `torch.cuda.memory_allocated()` | Remove CUDA branch; MPS path already exists |
| `torch.cuda.empty_cache()` (lines 650, 1229, 1257, 1659, 1689) | Called unconditionally or in cuda branch | Remove all calls (MPS has no equivalent; `gc.collect()` already present) |
| `torch.cuda.synchronize()` (line 1258) | Called after generation | Remove |
| Generator (lines 1053–1055) | `torch.Generator("cuda").manual_seed(seed)` | `torch.Generator("cpu").manual_seed(seed)` (works correctly for MPS seeding) |
| Upscale device fallback (lines 1409–1410) | `"cuda" if torch.cuda.is_available() else "cpu"` | `"cpu"` (spandrel MPS fallback already in place) |

### Changes in `generate.py`

| Location | Current | Replacement |
|----------|---------|-------------|
| dtype (line 32) | `if device in ["mps", "cuda"]` | `if device == "mps"` |
| Generator (lines 79–80) | `torch.Generator("cuda").manual_seed(seed)` | `torch.Generator("cpu").manual_seed(seed)` |
| argparse help (line 131) | `"mps, cuda, cpu"` | `"mps, cpu"` |
| Auto-detection (lines 178–180) | `elif torch.cuda.is_available(): device = "cuda"` | Remove |
| Validation (lines 185–187) | `elif device == "cuda" and not torch.cuda.is_available()` | Remove — preserve the final `else: print("Invalid device"); sys.exit(1)` branch |

### Changes in `pyproject.toml`

- Description: `"Offline AI image generation for Mac Silicon and CUDA"` → `"Offline AI image generation for Mac Silicon (MPS) — FLUX.2-klein, Z-Image Turbo, LTX-Video, LoRA support"`

---

## Task 3 — Model Sources Registry

### Data

**`model_sources.json`** at project root (gitignored). Created on first `GET /api/model-sources` if missing, seeded from `DEFAULT_SOURCES` constant in `server.py`.

```json
{
  "version": 1,
  "sources": [
    {
      "id": "<uuid4>",
      "name": "FLUX.2-klein-4B (4bit SDNQ)",
      "url": "https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
      "type": "base",
      "description": "~8 GB VRAM · MPS optimized · recommended for 16 GB Mac"
    }
  ]
}
```

`type` values: `"base"` | `"lora"` | `"upscaler"`

### Prefilled Sources (31 entries)

**Base models (7)**

`name` values must exactly match `KNOWN_MODELS` display names in `app.py` so the Download button can call `POST /api/models/update {model_choice: entry.name}`. Entries not in `KNOWN_MODELS` get a grayed-out Download button.

| Name (= KNOWN_MODELS key) | HuggingFace path | Description | In KNOWN_MODELS |
|---------------------------|-----------------|-------------|-----------------|
| FLUX.2-klein-4B (4bit SDNQ) | Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic | ~8 GB · fast · 16 GB Mac | ✓ |
| FLUX.2-klein-9B (4bit SDNQ) | Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32 | ~12 GB · high quality | ✓ |
| FLUX.2-klein-4B (Int8) | aydin99/FLUX.2-klein-4B-int8 | ~16 GB · MPS explicit | ✓ |
| FLUX.2-klein-9B FP8 | black-forest-labs/FLUX.2-klein-9b-fp8 | Official BFL FP8 | ✗ (open only) |
| Z-Image Turbo (Full) | Tongyi-MAI/Z-Image-Turbo | ~24 GB · LoRA support | ✓ |
| Z-Image Turbo (Quantized) | Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32 | ~6 GB · fast | ✓ |
| LTX-Video | Lightricks/LTX-Video | Official video model | ✓ |

**LoRAs (14)**

| Name | HuggingFace path | Description |
|------|-----------------|-------------|
| Outpaint (klein 4B) | fal/flux-2-klein-4B-outpaint-lora | Add green border region to fill |
| Zoom (klein 4B) | fal/flux-2-klein-4B-zoom-lora | Red region → enlarged detail |
| Spritesheet (klein 4B) | fal/flux-2-klein-4b-spritesheet-lora | Single object → 2×2 sprite sheet |
| Virtual Try-on (klein 9B) | fal/flux-klein-9b-virtual-tryon-lora | Clothing swap with references |
| 360 Outpaint (klein 4B) | nomadoor/flux-2-klein-4B-360-erp-outpaint-lora | Equirectangular outpaint |
| 360 Outpaint (klein 9B) | nomadoor/flux-2-klein-9B-360-erp-outpaint-lora | Equirectangular outpaint 9B |
| Style Pack (klein 9B) | DeverStyle/Flux.2-Klein-Loras | Arcane, DMC, flat-vector styles |
| Anime→Real (klein) | WarmBloodAban/Flux2_Klein_Anything_to_Real_Characters | Anime to photorealistic |
| Relight (klein 9B) | linoyts/Flux2-Klein-Delight-LoRA | Remove and replace lighting |
| Consistency (klein 9B) | dx8152/Flux2-Klein-9B-Consistency | Improve edit coherence |
| Enhanced Details (klein 9B) | dx8152/Flux2-Klein-9B-Enhanced-Details | Realism and texture boost |
| Distillation LoRA (klein 9B) | vafipas663/flux2-klein-base-9b-distill-lora | Better CFG + fine detail |
| AC Style (klein) | valiantcat/FLUX.2-klein-AC-Style-LORA | Comics and cyber neon style |
| Unified Reward (klein 9B) | CodeGoat24/FLUX.2-klein-base-9B-UnifiedReward-Flex-lora | Quality preference alignment |

**Upscalers (10)**

| Name | HuggingFace path | Description |
|------|-----------------|-------------|
| Real-ESRGAN x4 | Comfy-Org/Real-ESRGAN_repackaged | Safetensors · general purpose |
| 4xNomosWebPhoto RealPLKSR | Phips/4xNomosWebPhoto_RealPLKSR | Best for web/JPEG photos |
| 4xNomosWebPhoto ATD | Phips/4xNomosWebPhoto_atd | ATD architecture variant |
| 4xRealWebPhoto v4 DRCT-L | Phips/4xRealWebPhoto_v4_drct-l | Latest DRCT, real photos |
| 4x-UltraSharp | Kim2091/UltraSharp | JPEG artifact recovery |
| 4x-Remacri | OzzyGT/4xRemacri | Safetensors, general use |
| gyre upscalers (SwinIR + HAT) | halffried/gyre_upscalers | SwinIR + HAT safetensors |
| SwinIR safetensors | GraydientPlatformAPI/safetensor-upscalers | Clean safetensors collection |
| uwg upscaler collection | uwg/upscaler | Large multi-architecture collection |
| OpenModelDB | https://openmodeldb.info | Browse all community upscalers |

### Backend

Two new sync endpoints in `server.py` (sync because they do file I/O):

```python
GET  /api/model-sources   → reads model_sources.json; seeds from DEFAULT_SOURCES if missing
POST /api/model-sources   → body: {sources: [...]}; writes model_sources.json
```

`DEFAULT_SOURCES` is a constant list defined in `server.py` (not imported from a separate file — YAGNI).

### Frontend — `SettingsDrawer.tsx`

New collapsible section **"Model Sources"** placed between the upscale models section and the Server Log section.

**List view** (read mode):

```
[base]  FLUX.2-klein-4B SDNQ 4bit                    [↗ Open] [↓ Download]
        ~8 GB VRAM · MPS optimized · 16 GB Mac

[lora]  Outpaint LoRA (klein 4B)                      [↗ Open] [↓ Download]
        Add green border region to fill

[+ Add source]
```

- Type badge color: `base` = blue · `lora` = purple (accent) · `upscaler` = teal
- `↗ Open` → `window.open(entry.url, '_blank')` — opens the model card/page
- `↓ Download`:
  - **`base` type, in KNOWN_MODELS**: active button — calls `POST /api/models/update {model_choice: entry.name}`. The request body uses `LoadModelRequest {model_choice, device}` — no `url` field.
  - **`base` type, NOT in KNOWN_MODELS**: grayed-out with tooltip "Open HuggingFace page to download manually"
  - **`lora` type**: grayed-out with tooltip "Download from HuggingFace, then upload via the LoRA panel"
  - **`upscaler` type**: grayed-out with tooltip "Download from HuggingFace, then upload via the Upscale panel"
- Each row has a delete (×) button
- `[+ Add source]` opens an inline form: name, URL, type dropdown, description. New entries get a UUID via `crypto.randomUUID()` client-side.

### State

Local React state in `SettingsDrawer` — `sources` array loaded from `GET /api/model-sources` on drawer open (same timing as storage/model refresh). Local state is the source of truth after load — mutations update local state immediately and fire `POST /api/model-sources` (no re-fetch after save). Basic server-side validation on POST: `type` must be one of `base|lora|upscaler`, `url` and `name` must be non-empty strings.

---

## File Change Summary

| File | Change |
|------|--------|
| `core/lora_flux2.py` | Add `check_lora_compatibility()` function |
| `server.py` | Update `api_upload_lora` to call compat check; add `GET/POST /api/model-sources`; add `DEFAULT_SOURCES` constant |
| `app.py` | Remove all CUDA branches (see table above) |
| `generate.py` | Remove all CUDA branches (see table above) |
| `pyproject.toml` | Update description |
| `model_sources.json` | New file (gitignored, seeded on first read) |
| `frontend/src/api.ts` | Update `uploadLora()` to read `detail` from 422 JSON body |
| `frontend/src/components/SettingsDrawer.tsx` | Add Model Sources section |
| `.gitignore` | Add `model_sources.json` |

---

## Out of Scope

- GGUF model support (requires different inference stack)
- MLX model support (requires MLX framework)
- Core ML model support (requires coremltools)
- In-app download of LoRA/upscaler files directly from URL (requires streaming download endpoint)
- Civitai integration (requires API key flow)
