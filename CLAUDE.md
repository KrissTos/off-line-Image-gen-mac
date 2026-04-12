# off-line-Image-gen-mac — Project Notes

<!-- TOC: maintaining · what-is · entry-points · run · architecture · api-table · sse-events · state-slots · iterate-masks · models · deps · gitignore · known-issues · features -->

## Maintaining this file
- **Single file only** — never split into multiple docs; one file is easier for AI sessions to load and reason about
- **Keep it short** — target ≤ 200 lines; when adding new content, trim or compress something else
- **Update after every session** — reflect actual code state; stale docs are worse than no docs
- **TOC comment** — keep the `<!-- TOC: … -->` line at the top updated with section anchors
- **Dense format** — prefer tables, inline code, and one-liner bullets over prose paragraphs; avoid restating things the code makes obvious
- **Proactive `/ukn` prompts** — after fixing a non-obvious bug, adding a significant architectural piece, or resolving a tricky gotcha, proactively suggest "worth running `/ukn` to save this" before context fills up

## When to use superpowers vs direct implementation
- **`quick:` prefix** — user signals direct implementation; skip brainstorm/plan/subagents entirely, just write the code
- **1–2 files, clear requirements** → implement directly, no brainstorm/plan needed
- **3+ files, or design is unclear** → full superpowers flow (brainstorm → spec → plan → subagent-driven-development)

## What this project is
Fully offline AI image generation for Mac Silicon (MPS). No cloud, no subscriptions. Supports FLUX.2 and Z-Image Turbo models with 4-bit/int8 quantization. Features: text-to-image, image-to-image editing, multi-slot reference images with per-slot rectangle-mask drawing, iterative multi-mask inpainting, multi-LoRA stacking (up to 5, per-slot strength), gallery drag-and-drop into ref slots, inline HelpTip ⓘ tooltips, upscaling (single + batch folder), video generation (LTX-Video), **batch img2img**, **depth map generation** (DA3 16-bit PNG), **watermark removal** (FFT heuristic detect + LaMa inpainting, `core/erase.py`). **Gradio fully removed** — `app.py` is pure backend logic only.

**Renamed from** `ultra-fast-image-gen` → `off-line-Image-gen-mac`. Brand name in UI: **"Local AI Image Gen"** (TopBar + browser tab title).

## Entry points
| File | Purpose |
|------|---------|
| `server.py` | **FastAPI backend** — production API + static file server |
| `app.py` | Pure backend logic — Gradio UI fully removed |
| `generate.py` | CLI — **Z-Image Turbo only**, does NOT support FLUX/LTX |
| `Launch.command` | Double-click Mac launcher (3 modes below) |

## How to run
```bash
./Launch.command              # production: builds frontend/dist, serves :7860
./Launch.command --dev        # dev: FastAPI :7861 + Vite HMR at :5173

# Manual
source venv/bin/activate
python server.py --port 7860 --no-auto-shutdown
cd frontend && npm run build  # rebuild after any frontend change
```

**Browser heartbeat / auto-shutdown**: server shuts down 60 s after last ping. Frontend sends `POST /api/ping` every 5s AND fires `navigator.sendBeacon('/api/shutdown')` on `beforeunload`. `api_shutdown` starts a **4 s cancellable countdown** (`_shutdown_task`); the next `/api/ping` cancels it — so a page **refresh survives** (reconnects in <1s), while a real tab close shuts down after 4s. Watcher skips shutdown if `manager.is_busy`. Disable: `--no-auto-shutdown`.

## HuggingFace token
Stored in `huggingface/token` (gitignored). Type: **Read** (fine-grained, gated repos). Login via Settings drawer in UI, or `python -c "from huggingface_hub import login; login()"`. Must also accept terms on each gated model page.

## Architecture

### Backend key files
**`pipeline.py`** — `PipelineManager` singleton wrapping `app.generate_image()` in `asyncio.Lock` + `ThreadPoolExecutor(1)`. Yields SSE event dicts. `auto_save=False` prevents double-saving. Stop: `threading.Event` + `_GenerationStopped` raised from step callback; `finally` always runs `gc.collect()` + `torch.mps.empty_cache()`. `is_batch_running: bool` flag; `stop_requested` property (public accessor for `_stop_event.is_set()`).

**`server.py`** — FastAPI. Serves `frontend/dist/`. All routes `/api/*`. SSE via `StreamingResponse`. HTTP 423 when pipeline busy. Writes `.json` sidecar alongside each output image. Suppresses resource_tracker semaphore warning at import time via `warnings.filterwarnings`.

**`app.py`** — pure backend logic (Gradio fully removed). `generate_image()` initialises `image = None` and `video_frames = None` before each repeat-loop iteration. `lora_files: list[dict]` replaces `lora_file/lora_strength`; legacy single-LoRA args still accepted and merged at call time. `current_lora_paths: list` replaces `current_lora_path`. FLUX LoRA now loaded during generation (was previously skipped — pre-existing bug).

Output filename: `{YYYYMMDD}_{slug}.png` (date + slug, no seed/time; collision → `_2`, `_3` suffix). Sidecar `…{slug}.json` has ALL params. Companion folder `{slug}/` holds `params.json` + `ref_slot_N.png` + `mask.png` when refs/mask exist. Saved to `~/Pictures/ultra-fast-image-gen/` (`app.DEFAULT_OUTPUT_DIR`).

### Frontend (`frontend/`)
Vite + React + TypeScript + Tailwind CSS v3 → `frontend/dist/`. Browser tab title: `Local AI Image Gen` (`index.html`).

Key source files:
| File | Role |
|------|------|
| `src/App.tsx` | Root: bootstrap, 4 s status poll, 5 s heartbeat, SSE handler, ref-slot handlers, iterate loop |
| `src/store.ts` | `useReducer` global state; `useAppState()` → `{ state, dispatch }` |
| `src/types.ts` | `AppStatus`, `GenerateParams`, `SSEEvent`, `OutputItem`, `RefImageSlot`, `Workflow` |
| `src/api.ts` | Typed fetch helpers: `streamGenerate`, `streamBatchGenerate`, `uploadImage`, `uploadFromUrl`, `streamBatchUpscale`, `eraseDetect`, `eraseRemove`, … |

3-row center layout: Canvas (flex 5) / RefImagesRow (flex 4) / Gallery (flex 1) → 50/40/10 % via `style={{ flex: 'N 0 0%' }}`.

### Component details

**`Sidebar.tsx`** (`w-[576px]`) — Accordions: Model, Parameters, Size, LoRA (Z-Image Full + FLUX.2), Upscale (single + batch), **Batch Img2Img**, **Depth Map**, **Watermark Remover**, Video (LTX only), Workflows.

**`RefImagesRow.tsx`** — Horizontal strip (flex 4). Each slot: 80×80 thumbnail with role badge, 56×56 mask target (pencil → `MaskEditorModal`), per-slot strength slider.

**`MaskEditorModal`** — rectangle drag-select canvas. Always shows slot #1's image. Window-level `mousemove`/`mouseup` listeners. Escape/Enter shortcuts.

**`EraseEditorModal.tsx`** — full-screen canvas editor for watermark mask. Two-canvas: detached offscreen `maskRef` (full-res, natural image dims) + `displayRef` (scaled to ≤760×560 for display). Rectangle tool + brush tool (Shift=erase). 45% red tint overlay. `handleConfirm` → `canvas.toBlob` → `POST /api/upload` → `onConfirm(maskId, maskUrl)`. Error shown inline if upload fails.

**`HelpTip.tsx`** — inline ⓘ icon with hover tooltip. Uses `position:fixed` + `getBoundingClientRect()`. `pointer-events-none`, `z-50`.

**`Canvas.tsx`** — flex 5. Result image/video + generating overlay with spinner + progress %.

**`Gallery.tsx`** — flex 1, horizontal scroll. Thumbnails: `draggable` for gallery→ref slot drag. Hover: Info, Load Params, Upscale ×4, Delete. Do NOT use `title` on outer div — causes native browser tooltip.

**`SettingsDrawer.tsx`** — `w-96`. Output folder, Default Model, HF login, model list, upscale model list, storage summary, Server Log, Model Sources.

**`TopBar.tsx`** — "Local AI Image Gen" brand, model, device, VRAM, "generating…" pulse, settings gear.

### State — ref image slots
```typescript
interface RefImageSlot {
  slotId: number; imageId: string; imageUrl: string
  maskId: string | null; maskUrl: string | null
  strength: number; w?: number; h?: number
}
```
Actions: `ADD_REF_SLOT` · `REMOVE_REF_SLOT` · `SET_SLOT_MASK` · `CLEAR_SLOT_MASK` · `CLEAR_ALL_SLOTS` · `UPDATE_SLOT_STRENGTH` · `SET_SLOT_DIMS`

### Iterative multi-mask inpainting
`handleIterateGenerate` in `App.tsx` chains one `/api/generate` call per masked slot. Pass N: `inputs=[prev_out, slotN.image], mask=slotN.maskId, strength=slotN.strength`. `uploadFromUrl(url)` re-uploads between passes.

### API endpoints
| Method | Path | Notes |
|--------|------|-------|
| POST | `/api/ping` | Heartbeat |
| POST | `/api/stop` | Signal stop at next step boundary |
| POST | `/api/shutdown` | Immediate shutdown (sendBeacon on tab close); 4s delay; no-op if `--no-auto-shutdown` |
| GET | `/api/status` | `{model, device, loaded, busy, is_batch_running, vram_gb}` |
| GET/POST | `/api/models` / `/api/models/load` | List / load model |
| DELETE | `/api/models/{name}` | Delete cached model |
| GET | `/api/models/check-updates` | Query HF Hub for latest hashes |
| POST | `/api/generate` | SSE stream |
| POST | `/api/batch/generate` | SSE stream — folder of images; yields `batch_progress` |
| POST | `/api/upload` | Upload temp image → `{id, url}` |
| GET | `/api/temp/{id}` | Serve temp file |
| GET | `/api/outputs` | Recent outputs (with sidecar data) |
| GET/POST | `/api/workflows` / `/api/workflows/{name}` / `/api/workflows/save` / `/api/workflows/import` | Workflow CRUD + ComfyUI import |
| GET | `/api/lora/list` | `{files:[{name,path,model_type}]}` — `model_type`: `"flux"|"zimage"|"unknown"` |
| POST/POST/POST | `/api/upscale/upload` · `/api/upscale/batch` · `/api/upscale/single` | Upscale |
| GET | `/api/open-file-dialog` | macOS image picker → `{path, cancelled}` |
| GET | `/api/open-folder-dialog` | macOS folder picker → `{path, cancelled}` |
| POST | `/api/logs/save` | Snapshot `logs/server.log` → timestamped file |
| GET/POST | `/api/settings` | App settings |
| GET | `/api/storage` | Directory sizes |
| GET/POST/POST | `/api/hf/status` · `/api/hf/login` · `/api/hf/logout` | HF auth |
| POST | `/api/depth-map` | DA3/DA2 depth map; `{filename, model_repo}`; runs in `ThreadPoolExecutor(1)` |
| POST | `/api/erase/detect` | FFT watermark heuristic → `{image_id, image_url, mask_id, mask_url}`; `ThreadPoolExecutor` |
| POST | `/api/erase` | LaMa inpainting fill → `{url, filename}`; `mask_id` path-traversal guarded; `_erased_2.png` collision suffix |

### SSE event format
```json
{"type":"progress","message":"Step 5/20","step":5,"total":20}
{"type":"image","url":"/api/output/foo.png","info":"512×512 · seed 42"}
{"type":"done"}  {"type":"error","message":"…"}
{"type":"batch_progress","current":3,"total":12,"filename":"photo_003.jpg"}
```

## Key Python modules
- `pipeline.py` — `PipelineManager`, SSE generation loop
- `core/depth_map.py` — DA3/DA2 depth estimation; `generate_depth_map(path, repo_id, invert=True)` → 16-bit PNG bytes; white=near; module-level model cache
- `core/erase.py` — `detect_watermark(path) → bytes` (Laplacian+brightness anomaly, `np.bincount` for CC sizing); `remove_watermark(path, mask_bytes) → bytes` (LaMa, `_lama_cache`); mask resize uses `Image.NEAREST` (binary mask — LANCZOS would anti-alias edges)
- `core/lora_zimage.py` — LoRA injection for Linear/Conv2d; `load_lora_for_pipeline()` — NEVER use `pipe.load_lora_weights()` for Z-Image
- `core/lora_flux2.py` — LoRA for FLUX.2-klein: PEFT/fal prefix remap; requires diffusers git main
- `core/quantized_flux2.py` — 4-bit SDNQ + int8 quantization utilities
- `core/workflow_utils.py` — workflow parse/save/load, ComfyUI importer

## Models (cached in `./models/`)
| Model | VRAM | Notes |
|-------|------|-------|
| FLUX.2-klein-4B (4bit SDNQ) | <8 GB @ 512px | Fast |
| FLUX.2-klein-9B (4bit SDNQ) | ~12 GB @ 512px | Higher quality |
| FLUX.2-klein-4B (Int8) | ~16 GB | |
| Z-Image Turbo (Quantized) | ~8 GB | Fastest |
| Z-Image Turbo (Full) | ~24 GB | LoRA support |
| LTX-Video | — | txt2video / img2video |

## Environment / deps / tooling
Package manager: **`uv`** (not pip). Lock: `uv.lock`. Metadata: `pyproject.toml`. **Always sync with `UV_PROJECT_ENVIRONMENT=venv uv sync`** — `Launch.command` sets `UV_PROJECT_ENVIRONMENT=venv`; plain `uv sync` targets `.venv/` (wrong env).

Key deps: `torch`, `transformers`, `diffusers` (git), `sdnq` (git), `peft>=0.17`, `optimum-quanto>=0.2.7`, `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart>=0.0.12`, `aiofiles>=24.0`, `spandrel>=0.4.0` (upscaler), `simple-lama-inpainting>=0.1.1` (watermark removal)

Tailwind tokens: `bg:#0a0a0a` · `surface:#141414` · `card:#1c1c1c` · `border:#2a2a2a` · `accent:#7c3aed` · `muted:#6b7280` · `label:#6b7280`

## .gitignore key exclusions
`models/` · `venv/` · `huggingface/` · `lora_uploads/` · `upscale_models/` · `__pycache__/` · `*.safetensors *.bin *.gguf *.pt *.pth` · `*.env .env*` · `.DS_Store`

## Known issues / TODOs
- **`generate.py`**: Z-Image Turbo only; does not work with FLUX or LTX-Video
- **`slotsToParams()`**: single-pass only sends slot #1's mask; use Iterate Masks (Pipeline mode) for per-slot masks
- **No CLIP loader** — text encoders are bundled per model, loaded at model-load time

## Guidance scale + Steps per model
| Model | guidance | steps | Reason |
|-------|----------|-------|--------|
| Z-Image Turbo (any) | **0** | **4** | Step-wise distilled |
| FLUX.2 (all variants) | **0** | **20** | Step-wise distilled |
| LTX-Video | **3.0** | **25** | |
`guidanceForModel(model)` and `stepsForModel(model)` in `App.tsx`. Guidance slider **hidden** in UI; unhide when adding full-precision non-distilled models.

## Implementation notes
- **FLUX LoRA detection**: `current_model` (internal key) uses `startswith("flux2")`; `model_choice` (display name) uses `startsWith('FLUX')` — never mix
- **diffusers git main required** for FLUX.2-klein LoRA — stable release had hardcoded block count (48 vs actual 20)
- **FluxInpaintPipeline incompatible with Flux2Klein** — `app.py` skips it; falls to img2img for masked FLUX.2-klein generations
- **FastAPI route order**: SPA wildcard `/{path:path}` must be LAST — routes after it are unreachable
- **HF auth endpoints must be sync `def`**: blocking calls (`whoami()`, `os.walk`) — `async def` blocks event loop
- **Multi-LoRA stacking**: `lora_files: LoraSlot[]` (up to 5); named adapters in `load_loras()`; `if not lora_files` (not `is None`) for legacy fallback
- **Batch img2img stop**: `stop_requested` is the public property — never access `_stop_event` directly from `server.py`
- **Default model**: stored as `default_model` in `app_settings.json`; bootstrap in `App.tsx` reads it after `fetchSettings()`
- **DA3 depth map**: `depth-anything/DA3MONO-LARGE` (official); DA3 = invert, DA2 = no invert; output LANCZOS-resized to source resolution; GS/3D export deps mocked via `sys.modules`
- **xformers on Apple Silicon**: not installable, not needed — PyTorch MPS has built-in SDPA
- **`TEMP_DIR` path guard**: all endpoints that accept temp file IDs must check `path.resolve().is_relative_to(TEMP_DIR.resolve())` — see `/api/erase` and `/api/workflow-assets`
