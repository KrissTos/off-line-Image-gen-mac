# Local AI Image Gen

> Fully offline AI image generation and editing for **Mac Silicon (Apple MPS)**.
> No cloud. No API keys. No subscriptions. Everything runs on your machine.

> **Standalone project.** Started as a fork of [newideas99/ultra-fast-image-gen](https://github.com/newideas99/ultra-fast-image-gen) but has diverged significantly — Gradio replaced with a FastAPI backend + React UI, new features (batch img2img, depth maps, watermark removal, LoRA stacking, iterative inpainting), and a different architecture overall. Not intended to track upstream.

---

## What it does

- **Text-to-image** — generate images from a prompt
- **Image-to-image editing** — upload a reference photo and transform it with natural language
- **Inpainting** — mark the area to change in a full-screen mask editor (click-to-select with SAM, brush, polygon) and regenerate just that area
- **Multi-slot reference images** — slot #1 is the image to edit, further slots are material / style references; each slot has its own strength slider and optional mask
- **Iterative multi-mask inpainting** — chain multiple mask passes automatically (one per slot)
- **Video generation** — text-to-video and image-to-video with LTX-Video
- **Multi-LoRA stacking** — load up to 5 `.safetensors` LoRA adapters simultaneously, each with its own strength slider; the dropdown is filtered to only show LoRAs compatible with the active model
- **Batch img2img** — point at a folder of images and run the current prompt + params over all of them automatically; gallery updates after each image
- **Upscaling** — 4× single image or batch-folder upscale with any Spandrel-compatible model
- **Every generation is reproducible** — each run is saved as one folder with its parameters, reference images, masks and outputs; click any gallery thumbnail to reload the whole setup
- **Workflow save/load** — save a setup under a name (model, params, reference images, masks) and reload or overwrite it later
- **Gallery** — strip or grid view; click to reload a run, drag thumbnails into reference slots, upscale or delete
- **Watermark remover** — auto-detect or hand-paint a mask, then fill it with LaMa
- **Depth map generation** — generate 16-bit DA3 depth maps directly from the Gallery; white = near, black = far
- **Auto-outpaint** — when the base image has a different aspect ratio than the output, it is fitted (never stretched) and the borders are filled; FLUX 4B uses a dedicated outpaint LoRA

---

## Supported models

| Model | VRAM | Notes |
|---|---|---|
| **FLUX.2-klein-4B** (4-bit SDNQ) | < 8 GB @ 512 px | Fastest FLUX — text + image editing |
| **FLUX.2-klein-9B** (4-bit SDNQ) | ~12 GB @ 512 px | Higher quality |
| **FLUX.2-klein-4B** (Int8) | ~16 GB | Alternative quantization |
| **Z-Image Turbo** (Quantized) | ~8 GB | Fastest overall — text-to-image only |
| **Z-Image Turbo** (Full) | ~24 GB | LoRA support |
| **LTX-Video** 0.9.8-13B-distilled | — | Text-to-video / image-to-video, fast-preview mode |

Models are downloaded automatically the first time you select them. They are cached in the standard HuggingFace cache (`~/.cache/huggingface/hub`), shared with other tools.

---

## Requirements

| | Minimum |
|---|---|
| **Mac** | Apple Silicon (M1 or later) — macOS 13+ |
| **Python** | 3.11 or 3.12 |
| **RAM** | 16 GB recommended (more = better) |
| **Disk** | ~20 GB per model |

---

## Quick start — Mac (1-click)

```bash
git clone https://github.com/KrissTos/off-line-Image-gen-mac.git
cd off-line-Image-gen-mac
```

Then **double-click `Launch.command`** in Finder.

The first launch installs all dependencies (~5 min). The UI opens automatically in Google Chrome (default browser if Chrome is missing) at `http://localhost:7860`.

> **Terminal lifecycle**: The Terminal window that opens is managed automatically. When you close the browser tab the server shuts down and the script exits to the shell; **close the Terminal window yourself** (the launcher no longer closes it, since that could hit the wrong window). Refreshing the page reconnects within ~1 s and cancels the shutdown.

---

## Manual start

```bash
# Install uv (package manager) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the venv (in ./venv) and install deps
UV_PROJECT_ENVIRONMENT=venv uv sync

# Build the frontend
cd frontend && npm install && npm run build && cd ..

# Start the server
venv/bin/python server.py --port 7860
```

The server exits 60 s after the last open browser tab closes. Add `--no-auto-shutdown` to keep it running (e.g. for API use).

Open `http://localhost:7860` in your browser.

### Dev mode (hot-reload frontend)

```bash
./Launch.command --dev
# FastAPI on :7861, Vite HMR on :5173
```

---

## First run — model download

1. Open the UI and select a model from the **Model** accordion in the sidebar
2. Click **Load Model** — the model downloads from HuggingFace and is cached locally
3. Some models are **gated** (require a free HuggingFace account + accepting terms):
   - Create an account at [huggingface.co](https://huggingface.co)
   - Accept the model terms on the model page
   - Paste your **Read** token in **Settings → HuggingFace Login**

---

## How the UI works

```
┌──────────────────────────────────────────────────────────┐
│  TopBar — model name · device · VRAM · status            │
├──────────────┬───────────────────────────────────────────┤
│              │  Canvas (result image / video)            │
│   Sidebar    ├───────────────────────────────────────────┤
│   (params)   │  Reference image slots + mask editor      │
│              ├───────────────────────────────────────────┤
│              │  Gallery (recent outputs)                 │
└──────────────┴───────────────────────────────────────────┘
```

### Sidebar sections

| Section | What it does |
|---|---|
| **Model** | Pick and load a model |
| **Parameters** | Steps, guidance scale, seed, repeat count |
| **Size** | Output resolution — presets change per model |
| **LoRA** | Stack up to 5 LoRA adapters, each with its own strength; filtered by active model |
| **Upscale** | 4× single image or batch folder |
| **Batch Img2Img** | Process a whole folder of images with the current settings |
| **Video** | LTX-Video settings (only visible with LTX model) |
| **Depth Map** | Generate a 16-bit depth PNG for an image file |
| **Watermark Remover** | Detect or paint a watermark mask, then remove it (LaMa) |
| **Workflows** | Save / load / overwrite named setups |

### Reference image slots

- Drop an image on **+ ref img** (or click it to upload, or drag a gallery thumbnail onto it)
- Click a slot's **mask box** ("draw mask") to open the mask editor; a filled box reopens it, **×** clears it. Only the masked area (white) is regenerated
- Adjust the **strength slider** per slot (how much the model can change the image)
- Slot #1 is always the **base image**; slots #2+ are material / style references. In FLUX prompts, `image 1` = base, `image 2` = the first extra ref, and so on
- The generate pass uses only slot #1's mask; masks on slots #2+ are used by **Iterate Masks**

### Mask editor

A full-screen editor that always works on slot #1's image.

| Tool / key | What it does |
|---|---|
| **SAM click** (S) | Click an object to add it; **Shift+click** refines the last object with another point |
| **SAM box** (D) | Drag a box around an object to select it |
| **Brush** (B) | Paint the mask; `[` `]` change the brush size |
| **Polygon** (P) | Click points, Enter or click the first point to close |
| **Alt** | Subtract instead of add, for every tool |
| Invert (I) · Grow · Shrink | Whole-mask operations |
| Import… | Load a PNG (white = masked) as the mask |
| Undo / redo | Cmd+Z / Shift+Cmd+Z |
| Navigation | Wheel = zoom at cursor, +/− zoom, 0 = fit, Space+drag = pan, M = show/hide mask |
| Enter / Esc | Apply / cancel |

### Inpainting modes

| Mode | When to use |
|---|---|
| **Crop & Composite (Fast)** | Quick edits — crops the masked region, generates at lower res, composites back |
| **Inpainting Pipeline (Quality)** | Full-resolution inpainting — slower but cleaner results (Z-Image Full and FLUX.2-klein) |

### Iterate Masks button

When **Inpainting Pipeline (Quality)** is selected and you have masks on multiple slots, the button changes to **Iterate Masks**. This chains the generation passes: output of pass N becomes the input of pass N+1, applying each mask in sequence.

---

## Gallery

- **Click** a thumbnail → it shows in the canvas and the whole run is reloaded: prompt, model, size, steps, LoRAs, every reference image with its strength and mask, and **that output's seed** (an upscale reloads with its source's seed). While a generation is running, a click only previews.
- **Hover** → info, upscale ×4, delete. Deleting the last output of a run moves the whole run folder to the macOS Trash.
- The toggle at the top-right switches between strip and grid view.

---

## Workflows

A saved workflow is a named setup — model, parameters, reference images, masks, strengths — stored under `workflows/yy-mm-dd_name/` in the same format as a run folder, without outputs. Use it for setups you want to come back to before (or without) generating.

- **Save** — type a name and click Save
- **Load** — pick from the dropdown (last 15 shown) and click Load, or click the folder icon to open any workflow folder
- After loading, **Save (overwrite _name_)** rewrites that same folder; **Save as new** makes a copy under a new name

---

## Output files (run folders)

Outputs go to `~/Pictures/ultra-fast-image-gen/` by default (change in Settings). Every generation is one self-contained folder:

```
260927-151805_edit-image-1-a-photo-of/          yymmdd-HHMMSS_<prompt slug>
  workflow.json                                 all parameters (version 2)
  refs/slot_1.png  slot_2.png …                 reference images
  masks/slot_1.png                              masks
  outputs/
    260927-151805_edit-image-1_s812345.png      one file per repeat, named with its seed
    260927-151805_edit-image-1_s812345_3520x4736.png   upscale of it
```

A generation that is stopped or fails before producing anything leaves no folder. Output folders from older versions (flat `image.png` + `image.json`) can be converted once:

```bash
venv/bin/python -m core.run_store migrate ~/Pictures/ultra-fast-image-gen            # dry run: prints the plan
venv/bin/python -m core.run_store migrate ~/Pictures/ultra-fast-image-gen --apply    # do it
venv/bin/python -m core.run_store migrate workflows --apply                          # old saved workflows
```

---

## Settings

Open **Settings** (gear icon, top-right):

| Setting | Description |
|---|---|
| Output folder | Where run folders are saved |
| HuggingFace token | Required for gated models |
| Models | See which models are cached, download, delete |
| Upscale models | Manage upscaler weights |
| Model Sources | Curated list of base models, LoRAs, and upscalers — open HF page or download; locally cached entries highlighted with a green border |
| Server log | View and save the current session log |

---

## Benchmarks

### FLUX.2-klein-4B (4-bit SDNQ) — 512×512, 20 steps

| Hardware | Time |
|---|---|
| M3 Max (36 GB) | ~11 s |
| M2 Max (32 GB) | ~15 s |

### Z-Image Turbo (Quantized) — 512×512, 4 steps

| Hardware | Time |
|---|---|
| M2 Max | ~14 s |
| M1 Max | ~23 s |

---

## Project structure

```
off-line-Image-gen-mac/
├── server.py              ← FastAPI backend + static file server (main entry point)
├── app.py                 ← Generation logic, model management (no Gradio)
├── pipeline.py            ← Async bridge: FastAPI ↔ generation thread (SSE)
├── generate.py            ← CLI for Z-Image Turbo only
├── Launch.command         ← 1-click Mac launcher (production + dev modes)
│
├── frontend/              ← React + Vite + TypeScript UI → builds to frontend/dist/
│   └── src/
│       ├── App.tsx            ← Root: SSE handler, ref-slot logic, iterate loop, workflow restore
│       ├── workflow.ts        ← Run / workflow → params (pure, node-tested)
│       ├── store.ts           ← useReducer global state
│       ├── api.ts             ← Typed fetch helpers
│       ├── types.ts           ← Shared TypeScript types
│       └── components/
│           ├── Sidebar.tsx        ← All generation params + accordions
│           ├── Canvas.tsx         ← Result image / video + progress overlay
│           ├── RefImagesRow.tsx   ← Reference image slots + mask editor
│           ├── Gallery.tsx        ← Recent outputs, strip / grid
│           ├── TopBar.tsx         ← Brand, model, device, VRAM status
│           ├── SettingsDrawer.tsx ← HF login, model list, storage, log
│           ├── MaskEditor.tsx     ← Full-screen SAM / brush / polygon mask editor
│           ├── EraseEditorModal.tsx ← Watermark mask editor
│           └── HelpTip.tsx        ← Inline ⓘ tooltips
│
├── core/
│   ├── run_store.py       ← Run folders: create, record outputs, list, load, save workflows, migrate
│   ├── segment.py         ← SAM click-to-mask (mask editor)
│   ├── erase.py           ← Watermark detect + LaMa removal
│   ├── depth_map.py       ← DA3 / DA2 depth estimation → 16-bit PNG
│   ├── lora_flux2.py      ← LoRA for FLUX.2-klein (PEFT)
│   ├── lora_zimage.py     ← LoRA for Z-Image (forward-patch)
│   ├── quantized_flux2.py ← 4-bit SDNQ + int8 quantization
│   └── workflow_utils.py  ← Workflow save/load, ComfyUI importer
│
├── lora_uploads/          ← User-uploaded LoRA files (gitignored)
├── upscale_models/        ← Upscaler weights (gitignored)
├── workflows/             ← Saved workflow folders
└── logs/                  ← Server logs (server.log + timestamped snapshots)
```

---

## Contributing

All contributions are welcome — bug reports, feature ideas, new model support, UI improvements, docs.

- **Bug?** → open an [Issue](../../issues)
- **Idea?** → start a [Discussion](../../discussions/categories/ideas)
- **Question?** → use [Q&A Discussions](../../discussions/categories/q-a)
- **Code?** → fork, branch, PR — please describe what you changed and why

---

## Credits

- [FLUX.2-klein](https://huggingface.co/black-forest-labs) by Black Forest Labs
- [Z-Image Turbo](https://github.com/Tongyi-MAI/Z-Image) by Alibaba / Tongyi
- [SDNQ quantization](https://huggingface.co/Disty0) by Disty0
- [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) by Lightricks
- [diffusers](https://github.com/huggingface/diffusers) by HuggingFace
- [Spandrel](https://github.com/chaiNNer-org/spandrel) for upscaling

---

## License

See the individual model licenses for usage terms. Project source code is provided as-is for personal and research use.
