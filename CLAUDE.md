# ultra-fast-image-gen — Project Notes

## What this project is
AI image generation tool for Mac Silicon (MPS) and NVIDIA (CUDA). Supports multiple FLUX.2 and Z-Image Turbo models with 4-bit/int8 quantization for low memory usage. Includes text-to-image, image-to-image editing, LoRA support, and video generation (LTX-Video).

## Entry points
- `app.py` — legacy Gradio web UI at `http://localhost:7860`
- `server.py` — **new FastAPI backend** at configurable port (default 7860 production / 7861 dev)
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

## Architecture

### Backend
- `pipeline.py` — `PipelineManager` singleton: wraps `app.generate_image()` in `asyncio.Lock` + `ThreadPoolExecutor(max_workers=1)`. Provides `async generate()` generator yielding SSE event dicts. Uses `_NoOpProgress` shim instead of `gr.Progress`.
- `server.py` — FastAPI app. Serves `frontend/dist/` as static files. All API routes under `/api/`. SSE streaming via `StreamingResponse` with `text/event-stream`. Returns HTTP 423 if pipeline busy.
- `app.py` — original Gradio app, kept intact as fallback.

### Frontend (`frontend/`)
Vite + React + TypeScript + Tailwind CSS v3. Built to `frontend/dist/` (served by FastAPI in production).

Key files:
- `src/App.tsx` — root component: bootstrap, status polling (4s), SSE generation handler, upload handlers, workflow loader
- `src/store.ts` — `useReducer`-based global state; `useAppState()` returns `{ state, dispatch }`
- `src/types.ts` — `AppStatus`, `GenerateParams`, `SSEEvent` union, `OutputItem`, `Workflow`
- `src/api.ts` — typed fetch helpers; `streamGenerate()` uses manual ReadableStream SSE (not EventSource)
- `src/components/` — `TopBar`, `Canvas`, `Gallery`, `Sidebar`, `SettingsDrawer`

### API endpoints (server.py)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Pipeline status (model, device, busy, vram_gb) |
| GET | `/api/models` | Model choices + locally-available list |
| GET | `/api/devices` | Available compute devices |
| POST | `/api/generate` | SSE stream: progress/image/video/done/error events |
| POST | `/api/upload` | Upload temp image → returns `{id, url}` |
| GET | `/api/outputs` | Recent output file list |
| GET | `/api/output/{file}` | Serve output image/video |
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
Images saved to `~/Pictures/ultra-fast-image-gen/` with timestamp-based filenames.

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

## Files removed / cleaned up
- `Launch.command alias` — macOS Alias file (dead shortcut), deleted.
- `__pycache__/` — deleted as cleanup.
