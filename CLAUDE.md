# off-line-Image-gen-mac — Project Notes

<!-- TOC: maintaining · what-is · entry-points · run · architecture · api-table · sse-events · state-slots · iterate-masks · models · deps · gitignore · known-issues · features -->

## Maintaining this file
- **Single file only** — never split into multiple docs; one file is easier for AI sessions to load and reason about
- **Keep it short** — target ≤ 200 lines; when adding new content, trim or compress something else
- **Update after every session** — reflect actual code state; stale docs are worse than no docs
- **TOC comment** — keep the `<!-- TOC: … -->` line at the top updated with section anchors
- **Dense format** — prefer tables, inline code, and one-liner bullets over prose paragraphs; avoid restating things the code makes obvious

## What this project is
Fully offline AI image generation for Mac Silicon (MPS) and NVIDIA (CUDA). No cloud, no subscriptions. Supports FLUX.2 and Z-Image Turbo models with 4-bit/int8 quantization. Features: text-to-image, image-to-image editing, multi-slot reference images with per-slot rectangle-mask drawing, iterative multi-mask inpainting, LoRA support, upscaling (single + batch folder), video generation (LTX-Video).

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

**Browser heartbeat / auto-shutdown**: server shuts down 15 s after the last ping. Frontend sends `POST /api/ping` every 5 s. 20 s grace on first launch. Disable: `--no-auto-shutdown`.

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

**`Sidebar.tsx`** (`w-[576px]`) — Accordions: Model, Parameters, Size, LoRA, Upscale (single + batch), Video (LTX only), Workflows. Single adaptive bottom button:
- **Generate** — default (single-pass)
- **Iterate Masks** — shown instead of Generate when mask mode = "Inpainting Pipeline (Quality)" AND ≥1 slot has a mask; calls `handleIterateGenerate`
- **Stop** — replaces the button while generating

**`RefImagesRow.tsx`** — Horizontal strip (flex 4). Each slot shows:
- 80×80 thumbnail with role badge: teal **"base"** for slot #1, purple **"ref N"** for slots #2+
- 56×56 mask target (upload or draw via pencil icon → `MaskEditorModal`)
- Per-slot strength slider (0–1) below each card pair
- Mask-mode dropdown when ≥1 mask exists

**`MaskEditorModal`** — canvas rectangle drag-select. **Always shows slot #1's image** as the drawing canvas (all masks define regions on the base image). For slots #2+ shows an amber info bar. Canvas and `<img>` are co-located inside a shared inner `relative` wrapper sized to `displaySize`, so `absolute top:0 left:0` always aligns perfectly. Window-level `mousemove`/`mouseup` listeners (attached on mount, coords clamped to canvas bounds) keep drag alive past the canvas edge. Escape/Enter shortcuts. Outputs a full-resolution black/white PNG mask.

**`Canvas.tsx`** — flex 5. Shows result image/video + generating overlay. Drag-and-drop ref image support. Generating overlay: 56px spinner with `pct`% number overlaid at center (when step/total available), progress message text, thin progress bar.

**`Gallery.tsx`** — flex 1, horizontal scroll strip. `onSelect` injects prompt + model into sidebar. Hover shows prompt tooltip.

**`SettingsDrawer.tsx`** — `w-96` slide-in. Output folder (editable, saved via `POST /api/settings`), HF login, model list (✓ cached / ↓ not downloaded + per-model size + delete with confirm), storage summary. Refresh button reloads all data.

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
}
```
Actions: `ADD_REF_SLOT` · `REMOVE_REF_SLOT` · `SET_SLOT_MASK` · `CLEAR_SLOT_MASK` · `CLEAR_ALL_SLOTS` · `UPDATE_SLOT_STRENGTH`

### Iterative multi-mask inpainting
`handleIterateGenerate` in `App.tsx` chains one `/api/generate` call per masked slot:
```
Pass 1  inputs=[slot1.imageId]              mask=slot1.maskId  strength=slot1.strength → out_A
Pass 2  re-upload out_A → tmp_A
        inputs=[tmp_A, slot2.imageId]       mask=slot2.maskId  strength=slot2.strength → out_B
Pass N  re-upload out_N-1 → tmp
        inputs=[tmp, slotN.imageId]         mask=slotN.maskId  strength=slotN.strength → out_N
```
`uploadFromUrl(url)` re-uploads the previous output as a new temp file. Progress: "Pass N/M — applying mask on slot #K…". Loop stops early on failure or Stop.

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
| GET | `/api/devices` | Available compute devices |
| POST | `/api/generate` | SSE stream |
| POST | `/api/upload` | Upload temp image → `{id, url}` |
| GET | `/api/temp/{id}` | Serve temp file |
| GET | `/api/outputs` | Recent outputs (with sidecar data) |
| GET | `/api/output/{file}` | Serve output file |
| GET/POST | `/api/workflows` / `/api/workflows/{name}` / `/api/workflows/save` / `/api/workflows/import` | Workflow CRUD + ComfyUI import |
| POST/DELETE | `/api/lora/upload` · `/api/lora/load` · `/api/lora` | LoRA management |
| POST/POST | `/api/upscale/upload` · `/api/upscale/batch` | Upscale model + batch SSE |
| GET/POST | `/api/settings` | App settings |
| GET | `/api/storage` | Directory sizes |
| GET/POST/POST | `/api/hf/status` · `/api/hf/login` · `/api/hf/logout` | HF auth |

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
Key deps: `torch`, `transformers`, `diffusers` (git), `sdnq` (git), `gradio>=6.0`, `peft>=0.17`, `optimum-quanto>=0.2.7`, `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart>=0.0.12`, `aiofiles>=24.0`

Tailwind tokens: `bg:#0a0a0a` · `surface:#141414` · `card:#1c1c1c` · `border:#2a2a2a` · `accent:#7c3aed` · `muted:#6b7280` · `label:#6b7280`

## .gitignore key exclusions
`models/` · `venv/` · `huggingface/` · `lora_uploads/` · `upscale_models/` · `__pycache__/` · `*.safetensors *.bin *.gguf *.pt *.pth` · `*.env .env*` · `.DS_Store`

## Known issues / TODOs
- **`generate.py`**: Z-Image Turbo only; does not work with FLUX or LTX-Video
- **`slotsToParams()`**: single-pass only sends slot #1's mask; use Iterate Masks (Pipeline mode) for per-slot masks
- **No CLIP loader** — text encoders are bundled per model, loaded at model-load time, not configurable per generation

## Guidance scale per model
| Model | guidance | Reason |
|-------|----------|--------|
| Z-Image Turbo (any) | **0** | Distilled — CFG not used; >0 hurts quality |
| LTX-Video | **3.0** | |
| FLUX.2 (all) | **3.5** | |
`guidanceForModel(model)` helper in `App.tsx` auto-sets on model change and bootstrap.

## Implementation notes
- `_save_output_image` uses `%H%M%S_%f` (ms precision) — prevents repeat-count filename collisions
- `DELETE /api/output/{filename:path}` — deletes file + `.json` sidecar
- `GET /api/open-output-folder` — reveals output folder in Finder via `open <path>`
- `GET /api/open-folder-dialog` — macOS folder picker via `asyncio.create_subprocess_exec` + osascript (async, not blocking)
- `upscale_model_path` saved to `app_settings.json` on upload/clear; restored into params on bootstrap; handles old key `upscaler_model_path` too
- `upscale_image()` tries requested device, falls back to CPU if MPS unsupported by spandrel; errors surfaced in info bar
- Step progress: `generate_image()` `step_callback(step,total)` → diffusers `callback_on_step_end` on all 8 pipe calls → SSE `{step,total}` → Canvas `%` + bar
- Auto-outpaint: `_prepare_outpaint(ref, w, h, align)` fires when ref dims ≠ output size and no explicit mask; `outpaint_align` param (9-pos); 3×3 picker in SizePanel when `hasRefImage`
- Model-aware size presets: `presetsForModel()` in Sidebar; FLUX=6, Z-Image=3, LTX=4; 3-col grid with label+dims
