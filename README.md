# Local AI Image Gen

> Fully offline AI image generation and editing for **Mac Silicon (MPS)** and **NVIDIA (CUDA)**.
> No cloud. No API keys. No subscriptions. Everything runs on your machine.

![Local AI Image Gen UI](docs/screenshot.png)

---

## What it does

- **Text-to-image** — generate images from a prompt
- **Image-to-image editing** — upload a reference photo and transform it with natural language
- **Inpainting** — draw a rectangle mask on any region and regenerate just that area
- **Multi-slot reference images** — up to 6 reference images, each with its own mask and strength slider
- **Iterative multi-mask inpainting** — chain multiple mask passes automatically (one per slot)
- **Video generation** — text-to-video and image-to-video with LTX-Video
- **LoRA support** — load `.safetensors` LoRA adapters for Z-Image Full and FLUX.2-klein
- **Upscaling** — 4× single image or batch-folder upscale with any Spandrel-compatible model
- **Workflow save/load** — save your full setup (model, params, reference images, masks) and reload it later
- **Gallery** — browse recent outputs, drag them into reference slots, upscale or delete
- **Auto-outpaint** — automatically fill borders when the reference image is a different aspect ratio

---

## Supported models

| Model | VRAM | Notes |
|---|---|---|
| **FLUX.2-klein-4B** (4-bit SDNQ) | < 8 GB @ 512 px | Fastest FLUX — text + image editing |
| **FLUX.2-klein-9B** (4-bit SDNQ) | ~12 GB @ 512 px | Higher quality |
| **FLUX.2-klein-4B** (Int8) | ~16 GB | Alternative quantization |
| **Z-Image Turbo** (Quantized) | ~8 GB | Fastest overall — text-to-image only |
| **Z-Image Turbo** (Full) | ~24 GB | LoRA support |
| **LTX-Video** | — | Text-to-video / image-to-video |

Models are downloaded automatically the first time you select them. They are cached in `./models/`.

---

## Requirements

| | Minimum |
|---|---|
| **Mac** | Apple Silicon (M1 or later) — macOS 13+ |
| **NVIDIA** | CUDA-capable GPU, 8 GB VRAM |
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

The first launch installs all dependencies (~5 min). A browser tab opens automatically at `http://localhost:7860`.

---

## Manual start

```bash
# Install uv (package manager) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv sync

# Build the frontend
cd frontend && npm install && npm run build && cd ..

# Start the server
python server.py --port 7860
```

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
| **LoRA** | Upload and apply a LoRA adapter |
| **Upscale** | 4× single image or batch folder |
| **Video** | LTX-Video settings (only visible with LTX model) |
| **Workflows** | Save / load your full setup |

### Reference image slots

- Click **+** to add a reference image (upload, drag from gallery, or paste a URL)
- Click the **pencil icon** to draw a rectangle mask — only that region will be regenerated
- Adjust the **strength slider** per slot (how much the model can change the image)
- Slot #1 is always the **base image**; slots #2+ are style references

### Inpainting modes

| Mode | When to use |
|---|---|
| **Crop & Composite (Fast)** | Quick edits — crops the masked region, generates at lower res, composites back |
| **Inpainting Pipeline (Quality)** | Full-resolution inpainting — slower but cleaner results (Z-Image only) |

### Iterate Masks button

When **Inpainting Pipeline (Quality)** is selected and you have masks on multiple slots, the button changes to **Iterate Masks**. This chains the generation passes: output of pass N becomes the input of pass N+1, applying each mask in sequence.

---

## Workflows

Workflows save your entire session — model, all parameters, reference images, masks, and strength values — into a folder under `workflows/`.

- **Save** — type a name and click Save → creates `workflows/yy-mm-dd_name/`
- **Load** — pick from the dropdown (last 15 shown) and click Load
- **Open** — click the folder icon to browse to any workflow folder in Finder

---

## Output files

Images are saved to `~/Pictures/ultra-fast-image-gen/` by default (change in Settings).
Each image gets a `.json` sidecar with the prompt, seed, model, and all parameters.

Filename format: `YYYYMMDD_HHMMSS_seed_prompt-slug.png`

---

## Settings

Open **Settings** (gear icon, top-right):

| Setting | Description |
|---|---|
| Output folder | Where generated images are saved |
| HuggingFace token | Required for gated models |
| Models | See which models are cached, download, delete |
| Upscale models | Manage upscaler weights |
| Server log | View and save the current session log |

---

## Benchmarks

### FLUX.2-klein-4B (4-bit SDNQ) — 512×512, 20 steps

| Hardware | Time |
|---|---|
| M3 Max (36 GB) | ~11 s |
| M2 Max (32 GB) | ~15 s |
| RTX 3090 | ~3 s |

### Z-Image Turbo (Quantized) — 512×512, 4 steps

| Hardware | Time |
|---|---|
| M2 Max | ~14 s |
| M1 Max | ~23 s |

---

## Project structure

```
off-line-Image-gen-mac/
├── server.py          ← FastAPI backend (main entry point)
├── app.py             ← Generation logic + model management
├── pipeline.py        ← Async bridge: FastAPI ↔ generation thread
├── Launch.command     ← 1-click Mac launcher
├── frontend/          ← React + Vite + TypeScript UI
│   └── src/
│       ├── App.tsx
│       ├── components/
│       └── ...
├── core/
│   ├── lora_flux2.py
│   ├── lora_zimage.py
│   ├── quantized_flux2.py
│   └── workflow_utils.py
├── models/            ← Downloaded model weights (gitignored)
├── workflows/         ← Saved workflows
└── logs/              ← Server logs
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
