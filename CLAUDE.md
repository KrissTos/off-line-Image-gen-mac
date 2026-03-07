# off-line-Image-gen-mac — Project Notes

## What this project is
Fully offline AI image generation for Mac Silicon (MPS) and NVIDIA (CUDA). No cloud, no subscriptions. Supports FLUX.2 and Z-Image Turbo models with 4-bit/int8 quantization for low memory usage. Includes text-to-image, image-to-image editing, multi-slot reference images, LoRA support, and video generation (LTX-Video).

**Renamed from** `ultra-fast-image-gen` → `off-line-Image-gen-mac`. Fresh git history started (single initial commit on `main`). GitHub repo: create at github.com/new with name `off-line-Image-gen-mac`.

## HuggingFace token
- Token stored in `huggingface/token` (local folder, **gitignored** — never commits)
- The `huggingface/` folder is the HF cache dir moved inside the project
- Token type needed: **Read** token (fine-grained, with "Read access to gated repos")
- Must also manually accept terms on each gated model page (e.g. black-forest-labs/FLUX.1-dev)
- Login via Settings drawer in UI, or: `python -c "from huggingface_hub import login; login()"`
- Standard save location on Mac: `~/.cache/huggingface/token`

## Entry points
- `app.py` — legacy Gradio web UI at `http://localhost:7860`
- `server.py` — **FastAPI backend** at configurable port (default 7860 production / 7861 dev)
- `generate.py` — CLI for batch generation: `python generate.py "your prompt"`
- `Launch.command` — double-click Mac launcher (three modes, see below)

## How to run

### Production (React UI — default)
```bash
./Launch.command
# Builds frontend/dist/ if needed, serves everything at http://localhost:7860
```

### Development (FastAPI + Vite HMR)
```bash
./Launch.command --dev
# FastAPI on :7861, Vite dev server on :5173 — open http://localhost:5173
```

### Legacy Gradio UI
```bash
./Launch.command --gradio
# Original Gradio app on :7860
```

### Manual
```bash
source venv/bin/activate
python server.py --port 7860       # production (needs frontend/dist/)
python app.py                       # legacy Gradio
cd frontend && npm run dev          # frontend dev server only
```

### Frontend rebuild (after any frontend change)
```bash
rm -rf frontend/dist && cd frontend && npm run build
# Note: may need mcp__cowork__allow_cowork_file_delete if EPERM on dist/
```

## Architecture

### Backend
- `pipeline.py` — `PipelineManager` singleton: wraps `app.generate_image()` in `asyncio.Lock` + `ThreadPoolExecutor(max_workers=1)`. Provides `async generate()` generator yielding SSE event dicts. Uses `_NoOpProgress` shim instead of `gr.Progress`.
- `server.py` — FastAPI app. Serves `frontend/dist/` as static files. All API routes under `/api/`. SSE streaming via `StreamingResponse` with `text/event-stream`. Returns HTTP 423 if pipeline busy. Saves **sidecar JSON** alongside each generated image (same name, `.json` ext) with prompt/model/params for gallery recall.
- `app.py` — original Gradio app, kept intact as fallback.

### Frontend (`frontend/`)
Vite + React + TypeScript + Tailwind CSS v3. Built to `frontend/dist/` (served by FastAPI in production).

Key files:
- `src/App.tsx` — root component: bootstrap, status polling (4s), SSE generation handler, upload handlers, workflow loader. **3-row center layout**: row1=Canvas, row2=RefImagesRow, row3=Gallery.
- `src/store.ts` — `useReducer`-based global state; `useAppState()` returns `{ state, dispatch }`. Uses **slot-based ref images** (`refSlots: RefImageSlot[]`).
- `src/types.ts` — `AppStatus`, `GenerateParams`, `SSEEvent` union, `OutputItem` (with `prompt?`, `model_choice?`), `Workflow`, `RefImageSlot`
- `src/api.ts` — typed fetch helpers; `streamGenerate()` uses manual ReadableStream SSE (not EventSource)
- `src/components/` — `TopBar`, `Canvas`, `Gallery`, `Sidebar`, `SettingsDrawer`, `RefImagesRow`

### Component details

**Sidebar.tsx** — `w-[576px]` (doubled from original `w-72`). Contains: model/device selectors, prompt textarea (`rows={8}`), generation params, LoRA panel, Upscale panel (with file browser button for `.pth/.pt/.onnx/.safetensors/.bin`), seed controls. No longer contains Img2Img panel (moved to RefImagesRow).

**RefImagesRow.tsx** — Horizontal strip below Canvas. Each slot: 80×80 image thumbnail with `#N` purple badge + adjacent 56×56 mask upload target. Hover-reveal remove/clear buttons. Strength slider + mask-mode dropdown when slots exist. Maps to `input_image_ids` / `mask_image_id` at generation time via `slotsToParams()`.

**Gallery.tsx** — `onSelect` passes full `OutputItem` (not just URL). Hover shows prompt text strip. Clicking a gallery image injects prompt + model into the sidebar fields and shows the image in Canvas.

**SettingsDrawer.tsx** — `w-96`. Shows HF login status (green badge if logged in, username). "Models" section lists all model choices with ✓ cached / ↓ not downloaded indicators. Note about HF login needed for gated models. Enter key on token input triggers login.

### State management — ref image slots
```typescript
// RefImageSlot shape:
interface RefImageSlot {
  slotId:   number      // 1-based label (#1, #2, ...)
  imageId:  string      // upload ID
  imageUrl: string      // preview URL
  maskId:   string | null
  maskUrl:  string | null
}

// Actions:
ADD_REF_SLOT    // appends slot
REMOVE_REF_SLOT // removes + re-numbers
SET_SLOT_MASK / CLEAR_SLOT_MASK
CLEAR_ALL_SLOTS

// Maps to API params:
slotsToParams(slots) → { input_image_ids: [...], mask_image_id: slots[0]?.maskId }
```

### Sidecar JSON pattern (gallery prompt recall)
When `server.py` yields an `image` or `video` SSE event, it saves a `.json` file alongside the output:
```python
# e.g. 20260301_133342_foo.png → 20260301_133342_foo.json
{ "prompt": "...", "model_choice": "...", "width": 512, "height": 512, "steps": 20, ... }
```
`/api/outputs` reads these via `_read_sidecar()` and includes `prompt` + `model_choice` in each `OutputItem`. Gallery `onSelect` dispatches `SET_RESULT_URL` + `SET_PARAM` for prompt/model.

### API endpoints (server.py)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Pipeline status (model, device, busy, vram_gb) |
| GET | `/api/models` | Model choices + locally-available list |
| GET | `/api/devices` | Available compute devices |
| POST | `/api/generate` | SSE stream: progress/image/video/done/error events |
| POST | `/api/upload` | Upload temp image → returns `{id, url}` |
| GET | `/api/outputs` | Recent output file list (includes prompt, model_choice from sidecar) |
| GET | `/api/output/{file}` | Serve output image/video (also .webm) |
| GET/POST/DELETE | `/api/workflows/*` | Workflow CRUD |
| POST | `/api/workflows/import-comfyui` | Import ComfyUI JSON workflow |
| GET/POST/DELETE | `/api/lora/*` | LoRA management |
| GET/PUT | `/api/settings` | App settings |
| GET | `/api/storage` | Directory sizes |
| GET/POST | `/api/hf/*` | HuggingFace login/status |

### SSE event format
```json
{"type": "progress", "message": "Step 5/20", "step": 5, "total": 20}
{"type": "image",    "url": "/api/output/foo.png", "info": "512×512 · seed 42"}
{"type": "video",    "url": "/api/output/bar.mp4"}
{"type": "error",    "message": "..."}
{"type": "done"}
```

## Key modules
- `lora_zimage.py` — LoRA support for Linear/Conv2d layers.
- `quantized_flux2.py` — 4-bit SDNQ and int8 quantization utilities.
- `workflow_utils.py` — parse/save/load workflows; ComfyUI JSON importer (`parse_comfyui_workflow()`); `get_locally_available_models()` scans `./models/` HF cache dirs; `_KNOWN_CORE_NODES` frozenset (~110 standard ComfyUI node types) for custom-node detection.

## ComfyUI workflow import
`workflow_utils.parse_comfyui_workflow()` maps checkpoint filenames to local MODEL_CHOICES via `_APP_MODEL_REPOS` dict. Unknown/custom nodes are collected in `result["_unknown_nodes"]` and surfaced in the UI as warnings. `result["_comfyui_ckpt_name"]` preserves the raw checkpoint name for display.

## Models (cached in `./models/`)
| Model | VRAM | Notes |
|-------|------|-------|
| FLUX.2-klein-4B (4bit SDNQ) | <8GB @ 512px | Fast, text+image editing |
| FLUX.2-klein-9B (4bit SDNQ) | ~12GB @ 512px | Higher quality |
| FLUX.2-klein-4B (Int8) | ~16GB | |
| Z-Image Turbo (Quantized) | ~8GB | Fastest |
| Z-Image Turbo (Full) | ~24GB | LoRA support |
| LTX-Video | — | txt2video, img2video |

## Output
Images saved to `~/Pictures/ultra-fast-image-gen/` with timestamp-based filenames. Each image has a sidecar `.json` with generation params.

## Package manager
Uses `uv` (not pip directly). Lock file: `uv.lock`. Project metadata in `pyproject.toml`.

## Key dependencies
`torch`, `transformers`, `diffusers` (git), `sdnq` (git), `gradio>=6.0`, `peft>=0.17`, `optimum-quanto>=0.2.7`, `pillow`, `scipy`, `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart>=0.0.12`, `aiofiles>=24.0`

## Environment vars set by app.py
```python
PYTORCH_MPS_FAST_MATH = "1"
PYTORCH_MPS_HIGH_WATERMARK_RATIO = "0.0"   # use all unified memory
HF_HUB_CACHE = "./models"                   # local cache location
```

## Tailwind custom tokens (frontend)
```
bg: #0a0a0a  |  surface: #141414  |  card: #1c1c1c
border: #2a2a2a  |  accent: #7c3aed  |  muted: #6b7280
```

## .gitignore — key exclusions
```
models/         # downloaded model weights
venv/           # Python virtualenv
huggingface/    # HF token + cache (NEVER commit)
__pycache__/
*.safetensors *.bin *.gguf *.pt *.pth
*.env  .env*
.DS_Store
```

## Git history
- Fresh single initial commit on `main` branch (old ultra-fast-image-gen history discarded)
- No remote set yet — user will push manually after renaming folder to `off-line-Image-gen-mac`
- Push command: `gh repo create off-line-Image-gen-mac --public --source=. --remote=origin --push`
  or: `git remote add origin https://github.com/USERNAME/off-line-Image-gen-mac.git && git push -u origin main`
