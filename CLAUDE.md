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
Fully offline AI image generation for Mac Silicon (MPS). No cloud, no subscriptions. Supports FLUX.2 and Z-Image Turbo models with 4-bit/int8 quantization. Features: text-to-image, image-to-image editing, multi-slot reference images with per-slot rectangle-mask drawing, iterative multi-mask inpainting, LoRA support (Z-Image Full + FLUX.2-klein), gallery drag-and-drop into ref slots, inline HelpTip ⓘ tooltips, upscaling (single + batch folder), video generation (LTX-Video).

**Renamed from** `ultra-fast-image-gen` → `off-line-Image-gen-mac`. Brand name in UI: **"Local AI Image Gen"** (TopBar + browser tab title).

## Entry points
| File | Purpose |
|------|---------|
| `server.py` | **FastAPI backend** — production API + static file server |
| `app.py` | Legacy Gradio UI (fallback) |
| `generate.py` | CLI — **Z-Image Turbo only**, does NOT support FLUX/LTX |
| `Launch.command` | Double-click Mac launcher (3 modes below) |

## How to run
```bash
./Launch.command              # production: builds frontend/dist, serves :7860
./Launch.command --dev        # dev: FastAPI :7861 + Vite HMR at :5173
./Launch.command --gradio     # legacy Gradio :7860

# Manual
source venv/bin/activate
python server.py --port 7860 --no-auto-shutdown
cd frontend && npm run build  # rebuild after any frontend change
```

**Browser heartbeat / auto-shutdown**: server shuts down 60 s after the last ping (raised from 15 s; background tabs throttle `setInterval`). Watcher skips shutdown if `manager.is_busy` is True. Frontend sends `POST /api/ping` every 5 s. 20 s grace on first launch. Disable: `--no-auto-shutdown`.

## HuggingFace token
Stored in `huggingface/token` (gitignored). Type: **Read** (fine-grained, gated repos). Login via Settings drawer in UI, or `python -c "from huggingface_hub import login; login()"`. Must also accept terms on each gated model page.

## Architecture

### Backend key files
**`pipeline.py`** — `PipelineManager` singleton wrapping `app.generate_image()` in `asyncio.Lock` + `ThreadPoolExecutor(1)`. Yields SSE event dicts. `auto_save=False` prevents double-saving.

**`server.py`** — FastAPI. Serves `frontend/dist/`. All routes `/api/*`. SSE via `StreamingResponse`. HTTP 423 when pipeline busy. Writes `.json` sidecar alongside each output image. Suppresses resource_tracker semaphore warning at import time via `warnings.filterwarnings`.

**`app.py`** — original Gradio app. `generate_image()` initialises `image = None` and `video_frames = None` before each repeat-loop iteration to guard against `UnboundLocalError` when conditional branches are skipped.

Output filename: `{YYYYMMDD_HHMMSS}_{seed}_{slug}.png` → sidecar `…{slug}.json`. Saved to `~/Pictures/ultra-fast-image-gen/` (`app.DEFAULT_OUTPUT_DIR`).

### Frontend (`frontend/`)
Vite + React + TypeScript + Tailwind CSS v3 → `frontend/dist/`. Browser tab title: `Local AI Image Gen` (`index.html`).

Key source files:
| File | Role |
|------|------|
| `src/App.tsx` | Root: bootstrap, 4 s status poll, 5 s heartbeat, SSE handler, ref-slot handlers, iterate loop |
| `src/store.ts` | `useReducer` global state; `useAppState()` → `{ state, dispatch }` |
| `src/types.ts` | `AppStatus`, `GenerateParams`, `SSEEvent`, `OutputItem`, `RefImageSlot`, `Workflow` |
| `src/api.ts` | Typed fetch helpers: `streamGenerate`, `uploadImage`, `uploadFromUrl`, `streamBatchUpscale`, `pingServer`, … |

3-row center layout: Canvas (flex 5) / RefImagesRow (flex 4) / Gallery (flex 1) → 50/40/10 % via `style={{ flex: 'N 0 0%' }}`.

### Component details

**`Sidebar.tsx`** (`w-[576px]`) — Accordions: Model, Parameters, Size, LoRA (Z-Image Full + FLUX.2), Upscale (single + batch), Video (LTX only), Workflows. HelpTip ⓘ tooltips on: Steps, Guidance, Seed, Repeat count, LoRA strength, mask mode, outpaint align, model. Single adaptive bottom button:
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

**`Gallery.tsx`** — flex 1, horizontal scroll strip. `onSelect` injects prompt + model into sidebar. Thumbnails: `draggable` + `onDragStart` sets `dataTransfer('text/plain', item.url)` for gallery→ref slot drag. Hover shows 3 icon buttons (Info overlay with `W × H px`, Upscale ×4, Delete). Dimensions read from `img.naturalWidth/naturalHeight` on load into `imgDims` state. Info overlay is `position:absolute inset-0` inside the thumbnail (avoids `overflow-hidden` clipping). Do NOT use `title` attribute on the outer div — causes native browser tooltip showing the prompt everywhere.

**`SettingsDrawer.tsx`** — `w-96` slide-in. Output folder (editable, saved via `POST /api/settings`), HF login, model list (✓ cached / ↓ not downloaded + per-model size + delete with confirm), upscale model list (delete), storage summary, **Server Log** section (Save Log → `POST /api/logs/save` → timestamped snapshot in `logs/`). Refresh button reloads all data.

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
| GET | `/api/status` | `{model, device, loaded, busy, vram_gb}` |
| GET/POST | `/api/models` / `/api/models/load` | List / load model |
| DELETE | `/api/models/{name}` | Delete cached model |
| GET | `/api/models/check-updates` | Query HF Hub for latest hashes; returns `{results:[{choice,status,local_hash,online_hash}]}` |
| POST | `/api/models/update` | Download/update a model from HF Hub |
| GET | `/api/open-folder-dialog` | Open native macOS folder picker (osascript); returns `{path, cancelled}` |
| GET | `/api/open-workflow-folder-dialog` | Same but opens at WORKFLOWS_DIR via osascript `default location` |
| GET | `/api/devices` | Available compute devices |
| POST | `/api/generate` | SSE stream |
| POST | `/api/upload` | Upload temp image → `{id, url}` |
| GET | `/api/temp/{id}` | Serve temp file |
| GET | `/api/outputs` | Recent outputs (with sidecar data) |
| GET | `/api/output/{file}` | Serve output file |
| GET/POST | `/api/workflows` / `/api/workflows/{name}` / `/api/workflows/save` / `/api/workflows/import` | Workflow CRUD + ComfyUI import |
| GET | `/api/workflow-assets/{name}/{filename}` | Serve saved ref slot images/masks; `path.is_relative_to()` traversal guard |
| POST/DELETE | `/api/lora/upload` · `/api/lora/load` · `/api/lora` | LoRA management |
| POST/POST/POST | `/api/upscale/upload` · `/api/upscale/batch` · `/api/upscale/single` | Upscale: model upload, batch SSE, single image |
| GET | `/api/open-file-dialog` | macOS image file picker (osascript) → `{path, cancelled}` |
| POST | `/api/logs/save` | Copy `logs/server.log` → timestamped snapshot in `logs/`; returns `{saved_path}` |
| GET/POST | `/api/settings` | App settings |
| GET | `/api/storage` | Directory sizes |
| GET/POST/POST | `/api/hf/status` · `/api/hf/login` · `/api/hf/logout` | HF auth |
| GET | `/api/models/extras` | `{upscale_models:[{name,size}]}` |
| DELETE | `/api/upscale/{filename}` | Delete upscale model file |

### SSE event format
```json
{"type":"progress","message":"Step 5/20","step":5,"total":20}
{"type":"image","url":"/api/output/foo.png","info":"512×512 · seed 42"}
{"type":"video","url":"/api/output/bar.mp4"}
{"type":"error","message":"…"}
{"type":"done"}
```

## Key Python modules
- `pipeline.py` — `PipelineManager`, SSE generation loop
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
Key deps: `torch`, `transformers`, `diffusers` (git), `sdnq` (git), `gradio>=6.0`, `peft>=0.17`, `optimum-quanto>=0.2.7`, `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart>=0.0.12`, `aiofiles>=24.0`, `spandrel>=0.4.0` (upscaler)

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
| Z-Image Turbo (any) | **0** | **4** | Distilled — CFG not used; more steps hurt quality |
| LTX-Video | **3.0** | **25** | |
| FLUX.2 (all) | **3.5** | **20** | |
`guidanceForModel(model)` and `stepsForModel(model)` helpers in `App.tsx` auto-set on model change and bootstrap.

## Implementation notes
- `_save_output_image` uses `%H%M%S_%f` (ms precision) — prevents repeat-count filename collisions
- `DELETE /api/output/{filename:path}` — deletes file + `.json` sidecar
- `GET /api/open-output-folder` — reveals output folder in Finder via `open <path>`
- `GET /api/open-folder-dialog` — macOS folder picker via `asyncio.create_subprocess_exec` + osascript (async, not blocking)
- `upscale_model_path` saved to `app_settings.json` on upload/clear; restored into params on bootstrap; handles old key `upscaler_model_path` too
- `upscale_image()` tries requested device, falls back to CPU if MPS unsupported by spandrel; errors surfaced in info bar
- **Upscale broken root cause**: `spandrel` was missing from `pyproject.toml` → `ModuleNotFoundError` silently caught; fixed by adding `spandrel>=0.4.0` dep
- **GitHub Actions**: `.github/workflows/claude.yml` installed — `@claude` mentions in PRs/issues trigger Claude Code bot
- Step progress: `generate_image()` `step_callback(step,total)` → diffusers `callback_on_step_end` on all 8 pipe calls → SSE `{step,total}` → Canvas `%` + bar
- Auto-outpaint: `_prepare_outpaint(ref, w, h, align)` fires when ref dims ≠ output size and no explicit mask; `outpaint_align` param (9-pos); 3×3 picker in SizePanel when `hasRefImage`
- Model-aware size presets: `presetsForModel()` in Sidebar; FLUX=6, Z-Image=3, LTX=4; 3-col grid with label+dims
- **FLUX LoRA detection**: `current_model` stores internal strings (`"flux2-klein-int8"`, `"flux2-klein-sdnq"`, `"zimage-full"`), NOT display names. Use `current_model.startswith("flux2")` — never check for `"FLUX"` in the string.
- **diffusers git main required** for FLUX.2-klein LoRA — stable release had hardcoded block count (48 vs actual 20)
- `requests>=2.32` added to `pyproject.toml` (needed for potential future downloads)
- **FluxInpaintPipeline incompatible with Flux2Klein** — uses different transformer/vae types; `app.py` now skips the attempt and falls straight to img2img for masked FLUX.2-klein generations
- **"↕ ref size" button** in SizePanel — appears when slot #1 has dims; reads `refSlots[0].{w,h}` (populated via thumbnail `onLoad`), snaps to nearest 64, sets output width/height
- **FLUX inpainting warnings** — amber note shown in RefImagesRow (below mask mode dropdown) AND Sidebar (above Generate button) when `model_choice.startsWith('FLUX')` + mask mode = "Inpainting Pipeline (Quality)"
- LoRA files go in `lora_uploads/` (gitignored, create manually if missing)
- **Workflow ref persistence**: `api_save_workflow` bypasses `a.save_workflow()` — owns full folder creation, copies slot images/masks as `slot_N_image.png`/`slot_N_mask.png`; `api_load_workflow` returns `ref_slots:[{imageUrl,maskUrl,strength}]`; `handleWorkflowLoad` is async + sequential (not `Promise.all`) — `ADD_REF_SLOT` assigns slotId from `state.refSlots.length+1` at dispatch time so order matters; `isRestoringWorkflow` useRef guards against double-click race
- **`isFlux` in Sidebar**: use `params.model_choice.startsWith('flux2')` — `.toLowerCase().includes('flux')` was wrong (fixed 2026-03-15)
- **Server log capture**: `server.py` tees stdout+stderr to `logs/server.log` via `_LogTee` class; ANSI escape codes stripped with `_ANSI_RE` before writing to file; log truncated on each server start
- **HF auth endpoints must be sync `def`**: `api_hf_status/login/logout` and `api_storage` use blocking calls (`whoami()`, `os.walk`); must be `def` so FastAPI runs them in thread pool — `async def` blocks event loop causing all other requests to queue
- **FastAPI route order**: SPA wildcard `/{path:path}` must be LAST — any route registered after it is unreachable (matched by wildcard first)
- **FLUX inpainting `mode` crash**: FLUX + "Inpainting Pipeline (Quality)" + mask — the `if/elif` chain at the img2img path both guarded with `not (has_mask and "Inpainting" in mask_mode)`, so both branches skipped → `mode` never assigned → `UnboundLocalError`. Fixed: simplified to `if preprocessed_flux_refs is not None / else`
- **`POST /api/upscale/single`**: resolves `source:'gallery'` → `output_dir/filename`, `source:'path'` → absolute path; saves as `stem_WxH.ext` next to original; scale ×2/×3 = run ×4 then resize down
- **`SaveWorkflowRequest.lora_file`**: must be `str | None = None` — frontend sends `null` when no LoRA selected; plain `str` causes 422
- **Workflow folder naming**: `yy-mm-dd_Name` (e.g. `26-03-15_My_Portrait`); `GET /api/workflows` returns `[:15]` newest only
- **Workflow Open button**: `handleOpenFolder` in `WorkflowPanel` calls `openWorkflowFolderDialog()` → extracts `basename` → calls existing `loadWorkflow(name)`
