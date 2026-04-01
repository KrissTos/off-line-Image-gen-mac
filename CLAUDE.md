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
Fully offline AI image generation for Mac Silicon (MPS). No cloud, no subscriptions. Supports FLUX.2 and Z-Image Turbo models with 4-bit/int8 quantization. Features: text-to-image, image-to-image editing, multi-slot reference images with per-slot rectangle-mask drawing, iterative multi-mask inpainting, multi-LoRA stacking (up to 5, per-slot strength), gallery drag-and-drop into ref slots, inline HelpTip ⓘ tooltips, upscaling (single + batch folder), video generation (LTX-Video), **batch img2img** (process a folder of images), **depth map generation** (DA3 16-bit PNG, Gallery button). **Gradio fully removed** — `app.py` is pure backend logic only.

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
| `src/api.ts` | Typed fetch helpers: `streamGenerate`, `streamBatchGenerate`, `uploadImage`, `uploadFromUrl`, `streamBatchUpscale`, `pingServer`, … |

3-row center layout: Canvas (flex 5) / RefImagesRow (flex 4) / Gallery (flex 1) → 50/40/10 % via `style={{ flex: 'N 0 0%' }}`.

### Component details

**`Sidebar.tsx`** (`w-[576px]`) — Accordions: Model, Parameters, Size, LoRA (Z-Image Full + FLUX.2), Upscale (single + batch), **Batch Img2Img**, Video (LTX only), Workflows. HelpTip ⓘ tooltips on: Steps, Guidance, Seed, Repeat count, LoRA strength, mask mode, outpaint align, model. Single adaptive bottom button:
- **Generate** — default (single-pass)
- **Iterate Masks** — shown instead of Generate when mask mode = "Inpainting Pipeline (Quality)" AND ≥1 slot has a mask; calls `handleIterateGenerate`
- **Stop** — replaces the button while generating

**`RefImagesRow.tsx`** — Horizontal strip (flex 4). Each slot shows:
- 80×80 thumbnail with role badge: teal **"base"** for slot #1, purple **"ref N"** for slots #2+
- 56×56 mask target (upload or draw via pencil icon → `MaskEditorModal`)
- Per-slot strength slider (0–1) below each card pair
- Mask-mode dropdown when ≥1 mask exists

**`MaskEditorModal`** — rectangle drag-select canvas. Always shows slot #1's image (all masks define regions on the base image); slot #2+ shows amber info bar. Window-level `mousemove`/`mouseup` listeners keep drag alive past canvas edge. Escape/Enter shortcuts. Outputs full-res B/W PNG mask.

**`HelpTip.tsx`** — inline ⓘ icon with hover tooltip (positions: top/bottom/left/right, default top). Uses `position:fixed` + `getBoundingClientRect()` for viewport-aware placement. `pointer-events-none` on tooltip, `z-50` stacking.

**`Canvas.tsx`** — flex 5. Shows result image/video + generating overlay. Drag-and-drop ref image support. Generating overlay: 56px spinner with `pct`% number overlaid at center (when step/total available), progress message text, thin progress bar.

**`Gallery.tsx`** — flex 1, horizontal scroll strip. `onSelect` injects prompt + model into sidebar. Thumbnails: `draggable` + `onDragStart` sets `dataTransfer('text/plain', item.url)` for gallery→ref slot drag. Hover shows 4 icon buttons: Info (`W × H px`), Load Params (`RotateCcw` green, restores all params + ref slots), Upscale ×4, Delete. Info overlay is `position:absolute inset-0` inside the thumbnail. Do NOT use `title` on outer div — causes native browser tooltip.

**`SettingsDrawer.tsx`** — `w-96` slide-in. Sections: Output folder, **Default Model** (dropdown → `default_model` in `app_settings.json`, pre-selects on bootstrap), HF login, model list (✓ cached / ↓ not downloaded + size + delete + update), upscale model list, storage summary, Server Log (Save → timestamped snapshot in `logs/`), Model Sources. Refresh button reloads all data.

**`TopBar.tsx`** — "Local AI Image Gen" brand, model, device, VRAM, "generating…" pulse, settings gear.

### State — ref image slots
```typescript
interface RefImageSlot {
  slotId:   number         // 1-based; slot #1 = base image, #2+ = style references
  imageId:  string         // temp upload id
  imageUrl: string         // preview URL (/api/temp/…)
  maskId:   string | null  // mask temp upload id
  maskUrl:  string | null  // mask preview URL
  strength: number         // per-slot inpaint strength; slot #1 also drives params.img_strength
  w?:       number         // natural image width (populated on thumbnail onLoad)
  h?:       number         // natural image height
}
```
Actions: `ADD_REF_SLOT` · `REMOVE_REF_SLOT` · `SET_SLOT_MASK` · `CLEAR_SLOT_MASK` · `CLEAR_ALL_SLOTS` · `UPDATE_SLOT_STRENGTH` · `SET_SLOT_DIMS`

### Iterative multi-mask inpainting
`handleIterateGenerate` in `App.tsx` chains one `/api/generate` call per masked slot. Pass N: `inputs=[prev_out, slotN.image], mask=slotN.maskId, strength=slotN.strength`. `uploadFromUrl(url)` re-uploads between passes. Progress: "Pass N/M — applying mask on slot #K…". Loop stops early on failure or Stop.

### API endpoints
| Method | Path | Notes |
|--------|------|-------|
| POST | `/api/ping` | Heartbeat |
| POST | `/api/stop` | Signal generation thread to stop at next step boundary |
| POST | `/api/shutdown` | Immediate shutdown (called by `sendBeacon` on tab close); 1.5s delay; no-op if `--no-auto-shutdown` |
| GET | `/api/status` | `{model, device, loaded, busy, is_batch_running, vram_gb}` |
| GET/POST | `/api/models` / `/api/models/load` | List / load model |
| DELETE | `/api/models/{name}` | Delete cached model |
| GET | `/api/models/check-updates` | Query HF Hub for latest hashes; returns `{results:[{choice,status,local_hash,online_hash}]}` |
| POST | `/api/models/update` | Download/update a model from HF Hub |
| GET | `/api/open-folder-dialog` | Open native macOS folder picker (osascript); returns `{path, cancelled}` |
| GET | `/api/open-workflow-folder-dialog` | Same but opens at WORKFLOWS_DIR via osascript `default location` |
| GET | `/api/devices` | Available compute devices |
| POST | `/api/generate` | SSE stream |
| POST | `/api/batch/generate` | SSE stream — process folder of images; yields `batch_progress` + normal generation events |
| POST | `/api/upload` | Upload temp image → `{id, url}` |
| GET | `/api/temp/{id}` | Serve temp file |
| GET | `/api/outputs` | Recent outputs (with sidecar data) |
| GET | `/api/output/{file}` | Serve output file |
| GET/POST | `/api/workflows` / `/api/workflows/{name}` / `/api/workflows/save` / `/api/workflows/import` | Workflow CRUD + ComfyUI import |
| GET | `/api/workflow-assets/{name}/{filename}` | Serve saved ref slot images/masks; `path.is_relative_to()` traversal guard |
| GET | `/api/lora/list` | Scan `lora_uploads/` sorted by mtime desc → `{files:[{name,path,model_type}]}` (`model_type`: `"flux"\|"zimage"\|"unknown"`) |
| POST/DELETE | `/api/lora/upload` · `/api/lora/load` · `/api/lora` | LoRA management |
| POST/POST/POST | `/api/upscale/upload` · `/api/upscale/batch` · `/api/upscale/single` | Upscale: model upload, batch SSE, single image |
| GET | `/api/open-file-dialog` | macOS image file picker (osascript) → `{path, cancelled}` |
| POST | `/api/logs/save` | Copy `logs/server.log` → timestamped snapshot in `logs/`; returns `{saved_path}` |
| GET/POST | `/api/settings` | App settings |
| GET | `/api/storage` | Directory sizes |
| GET/POST/POST | `/api/hf/status` · `/api/hf/login` · `/api/hf/logout` | HF auth |
| GET | `/api/models/extras` | `{upscale_models:[{name,size}]}` |
| DELETE | `/api/upscale/{filename}` | Delete upscale model file |
| POST | `/api/depth-map` | Generate 16-bit depth map for an output image; `{filename, model_repo}`; runs in `ThreadPoolExecutor(1)` |

### SSE event format
```json
{"type":"progress","message":"Step 5/20","step":5,"total":20}
{"type":"image","url":"/api/output/foo.png","info":"512×512 · seed 42"}
{"type":"video","url":"/api/output/bar.mp4"}
{"type":"error","message":"…"}
{"type":"done"}
{"type":"batch_progress","current":3,"total":12,"filename":"photo_003.jpg"}
```

## Key Python modules
- `pipeline.py` — `PipelineManager`, SSE generation loop
- `core/depth_map.py` — DA3/DA2 depth estimation; `generate_depth_map(path, repo_id, invert=True)` → 16-bit PNG bytes; white=near; output always resized back to source resolution with LANCZOS; module-level model cache; DA3 path uses `depth_anything_3.api.DepthAnything3`, DA2 path uses `transformers.pipeline`; detects DA3 via `'DA3'/'da3' in repo_id`
- `core/lora_zimage.py` — LoRA injection for Linear/Conv2d layers
- `core/lora_flux2.py` — LoRA for FLUX.2-klein: PEFT/fal prefix remap; requires diffusers git main
- `core/ip_adapter_flux.py` — IP-Adapter code (kept on disk but NOT imported; Flux2KleinPipeline has no IPA support)
- `core/quantized_flux2.py` — 4-bit SDNQ + int8 quantization utilities
- `core/workflow_utils.py` — workflow parse/save/load, ComfyUI importer, `get_locally_available_models()`

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
```python
# Set by app.py:
PYTORCH_MPS_FAST_MATH = "1"
PYTORCH_MPS_HIGH_WATERMARK_RATIO = "0.0"
HF_HUB_CACHE = "./models"
```
Package manager: **`uv`** (not pip). Lock: `uv.lock`. Metadata: `pyproject.toml`.
- **`[tool.uv] dev-dependencies`** deprecated — use `[dependency-groups] dev = []` instead (fixed in session 6)
Key deps: `torch`, `transformers`, `diffusers` (git), `sdnq` (git), `peft>=0.17`, `optimum-quanto>=0.2.7`, `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart>=0.0.12`, `aiofiles>=24.0`, `spandrel>=0.4.0` (upscaler)

Tailwind tokens: `bg:#0a0a0a` · `surface:#141414` · `card:#1c1c1c` · `border:#2a2a2a` · `accent:#7c3aed` · `muted:#6b7280` · `label:#6b7280`

## .gitignore key exclusions
`models/` · `venv/` · `huggingface/` · `lora_uploads/` · `upscale_models/` · `__pycache__/` · `*.safetensors *.bin *.gguf *.pt *.pth` · `*.env .env*` · `.DS_Store`

## Known issues / TODOs
- **`generate.py`**: Z-Image Turbo only; does not work with FLUX or LTX-Video
- **`slotsToParams()`**: single-pass only sends slot #1's mask; use Iterate Masks (Pipeline mode) for per-slot masks
- **No CLIP loader** — text encoders are bundled per model, loaded at model-load time, not configurable per generation

## Guidance scale + Steps per model
| Model | guidance | steps | Reason |
|-------|----------|-------|--------|
| Z-Image Turbo (any) | **0** | **4** | Step-wise distilled — CFG ignored; more steps hurt quality |
| FLUX.2 (all variants) | **0** | **20** | Step-wise distilled — diffusers warns + ignores any non-zero value |
| LTX-Video | **3.0** | **25** | |
`guidanceForModel(model)` and `stepsForModel(model)` in `App.tsx`. Guidance slider **hidden** in UI (value still sent); unhide when adding full-precision non-distilled models.

## Implementation notes
- `DELETE /api/output/{filename:path}` — deletes file + `.json` sidecar; `upscale_image()` falls back to CPU if MPS unsupported by spandrel
- `upscale_model_path` saved to `app_settings.json` on upload/clear; restored on bootstrap; handles old key `upscaler_model_path`
- Auto-outpaint: `_prepare_outpaint(ref, w, h, align)` fires when ref dims ≠ output size and no explicit mask; 9-pos `outpaint_align`; 3×3 picker in SizePanel
- **FLUX LoRA detection**: `current_model` (internal key) uses `startswith("flux2")`; `model_choice` (display name, e.g. `"FLUX.2-klein-4B…"`) uses `startsWith('FLUX')` — never mix
- **diffusers git main required** for FLUX.2-klein LoRA — stable release had hardcoded block count (48 vs actual 20)
- **FluxInpaintPipeline incompatible with Flux2Klein** — `app.py` skips it; falls to img2img for masked FLUX.2-klein generations
- **FLUX inpainting `mode` crash** — fixed by simplifying to `if preprocessed_flux_refs is not None / else` (was double-guarded if/elif, `mode` never assigned → `UnboundLocalError`)
- **FastAPI route order**: SPA wildcard `/{path:path}` must be LAST — routes after it are unreachable
- **HF auth endpoints must be sync `def`**: blocking calls (`whoami()`, `os.walk`) — `async def` blocks event loop
- **Multi-LoRA stacking**: `lora_files: LoraSlot[]` (up to 5); named adapters in `load_loras()`; partial-load cleanup via `unload_lora_weights()` before re-raise; `if not lora_files` (not `is None`) for legacy fallback; stable slot keys via `useRef` counter
- **LoRA dropdown filtered by model**: `GET /api/lora/list` returns `model_type: "flux"|"zimage"|"unknown"` (safetensors header check); `LoraPanel` shows compatible + `unknown` entries (hide only positively incompatible); `_detect_lora_type()` in `server.py`
- **Model Sources local highlight**: `isLocal` uses `src.model_choice` (not `src.name`) for base type — `DEFAULT_SOURCES` names are shorter than `MODEL_CHOICES` strings; `model_choice` field added to each base entry; GET endpoint merges it by ID for existing `model_sources.json` on disk
- **Z-Image LoRA loader**: uses `load_lora_for_pipeline()` from `core/lora_zimage.py` (forward-patch, not PEFT) — `pipe.load_lora_weights()` fails with "target modules not found" because PEFT can't match Z-Image module names; networks tracked in `_zimage_lora_networks: list`; `_unload_zimage_loras()` calls `net.remove()` on each
- **`mode` UnboundLocalError**: `mode = "txt2img"` initialised alongside `image = None` before try block; Z-Image inpainting fallback flag `_zimage_inpaint_failed` ensures img2img runs when `ZImageInpaintPipeline` throws
- **`--debug` flag**: `server.py` sets `pipeline.DEBUG = True`; `pipeline.py` dumps full generation params + LoRA state before each run; `app.py` `_dbg()` prints branch-taken lines; `Launch-Debug.command` double-click shortcut
- **LoRA compat check on upload**: `check_lora_compatibility()` rejects single_blocks≥20 or double_blocks≥19; `api_upload_lora` uses `.tmp_<name>` temp → HTTP 422 on failure; `uploadLora()` in `api.ts` surfaces the `detail` message
- **`SaveWorkflowRequest.lora_file`**: `str | None = None` — frontend sends `null`; plain `str` causes 422
- **Workflow ref persistence**: `api_save_workflow` copies slot images/masks as `slot_N_image/mask.png`; `handleWorkflowLoad` is async + sequential — `ADD_REF_SLOT` assigns slotId at dispatch time so order matters
- **Workflow folder naming**: `yy-mm-dd_Name`; `GET /api/workflows` returns `[:15]` newest only
- **Server log**: `_LogTee` tees stdout+stderr to `logs/server.log`; ANSI stripped; truncated on server start
- **CUDA fully removed**: MPS-only; `torch.Generator("cpu").manual_seed(seed)`; Z-Image dtype always `torch.float32`
- **Model Sources registry**: `GET/POST /api/model-sources` → `model_sources.json`; 31 DEFAULT_SOURCES; `Body(...)` required for dict POST
- **Sleep/wake auto-shutdown**: `_heartbeat_watcher` tracks wall-clock between ticks; jump > 60 s = Mac sleep → resets `_last_ping` instead of shutting down
- **Safari Gallery padding**: `viewport-fit=cover` in `index.html`; `paddingBottom: 'max(8px, env(safe-area-inset-bottom))'` in Gallery
- **MPS memory flush after generation**: `pipeline.py` `_thread()` has a `finally` block with `gc.collect()` + `torch.mps.empty_cache()` — releases pooled MPS memory back to OS after every generation (success or error). Root cause: `app.py` has per-branch cache calls but none guaranteed at top-level exit.
- **Terminal auto-close**: `Launch.command` runs `osascript` in background (`&`) with `delay 0.3` then `exit` — shell exits BEFORE osascript fires, preventing macOS "running process" dialog. If osascript runs while shell is still alive (foreground), Terminal shows a confirmation dialog.
- **Stop generation**: `POST /api/stop` sets `manager._stop_event`; step callback raises `_GenerationStopped`; caught cleanly (no error SSE); `finally` always frees MPS memory. Frontend `handleStop` calls `stopGeneration()` then aborts SSE.
- **Default model**: stored as `default_model` in `app_settings.json`; bootstrap in `App.tsx` reads it after `fetchSettings()` and dispatches `SET_PARAM model_choice` + guidance + steps overrides.
- **Guidance slider hidden**: removed from `Sidebar.tsx` UI; value still in state, correct default set by `guidanceForModel()`. ALL current models are step-wise distilled → guidance=0 always sent.
- **Debug dump cleanup**: `lora_file`/`lora_strength` excluded from `[DEBUG]` params when `lora_files` is populated (legacy fields, always `None`/`1.0` from frontend).
- **Batch img2img**: `POST /api/batch/generate` uses `BatchGenerateRequest(GenerateRequest)` + `input_folder`. Endpoint sets `manager.is_batch_running`, streams `batch_progress` + generation events per image; single-image failures log and continue. Frontend stop calls `stopGeneration()` (mid-image) + `abort()` (between images). `stop_requested` is the public property; never access `_stop_event` directly from server.py.
- **Depth map**: `POST /api/depth-map` → `core/depth_map.py`; accepts `file_path` (absolute) or `filename` (output-dir); DA3 = invert, DA2 = no invert; output always LANCZOS-resized to source resolution; saved as `{stem}_depth.png`; `depth_model_repo` in `GenerateParams` + `app_settings.json`; model selector in **Sidebar DEPTH MAP accordion**; `from pipeline import manager` lazy import in endpoint.
- **DA3 install**: `depth_anything_3` installed with `--no-deps` + `omegaconf addict moviepy<2 opencv-python-headless`; GS/3D export deps (`gsplat`, `plyfile`, `evo`) mocked at import time via `sys.modules` — never needed for inference. Official model: `depth-anything/DA3MONO-LARGE`; local weights → `models/da3mono-large/`; `from_pretrained` uses local dir if `model.safetensors` exists, else falls back to HF Hub.
- **DA3 model IDs**: `depth-anything/DA3MONO-LARGE` (official, verified); `depth-anything/Depth-Anything-V2-Large-hf` (DA2 fallback); `istiakiat/DA3-BASE` does NOT exist (404). Sidebar dropdown has 2 options only.
- **xformers on Apple Silicon**: not installable, not needed — PyTorch MPS has built-in SDPA. Never attempt to install xformers on Mac.
