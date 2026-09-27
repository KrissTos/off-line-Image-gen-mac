# Architecture reference

Deep reference for `off-line-Image-gen-mac`. Operational rules live in `CLAUDE.md`; this file is
consulted when working on a specific layer. Update it in the same session as the code it describes.

## Backend files

| File | Role |
|------|------|
| `server.py` | FastAPI. Serves `frontend/dist/`, all routes `/api/*`, SSE via `StreamingResponse`, HTTP 423 when pipeline busy. Writes `.json` sidecar per output. Suppresses resource_tracker semaphore warning at import (`warnings.filterwarnings`). |
| `pipeline.py` | `PipelineManager` singleton wrapping `app.generate_image()` in `asyncio.Lock` + `ThreadPoolExecutor(1)`; turns each `(image, video, status)` yield into an SSE dict. `auto_save=False` prevents double-saving. Stop = `threading.Event` + `_GenerationStopped` raised from the step callback; `finally` always runs `gc.collect()` + `torch.mps.empty_cache()`. `is_batch_running: bool`; `stop_requested` = public accessor for `_stop_event.is_set()`. |
| `app.py` | Pure backend logic. `generate_image()` initialises `image = None` / `video_frames = None` each repeat iteration. `lora_files: list[dict]` (legacy `lora_file/lora_strength` still merged at call time); `current_lora_paths: list`. |
| `generate.py` | CLI, Z-Image Turbo only. |

### Output files
- Filename `{YYYYMMDD}_{slug}.png` (no seed/time; collision → `_2`, `_3`). Sidecar `…{slug}.json` holds ALL params.
- Companion folder `{slug}/` holds `params.json` + `ref_slot_N.png` + `mask.png` when refs/mask exist.
- Default dir `~/Pictures/ultra-fast-image-gen/` (`app.DEFAULT_OUTPUT_DIR`); overridable in Settings (`output_dir`).

### Heartbeat / auto-shutdown
- Server exits 60 s after the last `POST /api/ping` (frontend pings every 5 s). Watcher skips while `manager.is_busy`; resets after system sleep.
- Tab close fires `navigator.sendBeacon('/api/shutdown')` → 4 s cancellable countdown (`_shutdown_task`); the next ping cancels it, so a refresh survives.
- `--no-auto-shutdown` sets env `IMAGEGEN_NO_AUTO_SHUTDOWN=1`, read at module import (`uvicorn.run("server:app")` re-imports the module). Test: `tests/test_auto_shutdown_flag.py`.

## Python modules (`core/`)
- `core/depth_map.py` — DA3/DA2 depth; `generate_depth_map(path, repo_id, invert=True)` → 16-bit PNG bytes, white=near; module-level model cache.
- `core/erase.py` — `detect_watermark(path) → bytes` (Laplacian + brightness anomaly, `np.bincount` CC sizing); `remove_watermark(path, mask_bytes) → bytes` (LaMa, `_lama_cache`); mask resize `Image.NEAREST` (binary mask).
- `core/lora_zimage.py` — LoRA injection for Linear/Conv2d via `load_lora_for_pipeline()`.
- `core/lora_flux2.py` — FLUX.2-klein LoRA, PEFT/fal prefix remap.
- `core/quantized_flux2.py` — 4-bit SDNQ + int8 quantization utilities.
- `core/workflow_utils.py` — workflow parse/save/load, ComfyUI importer.

## Frontend (`frontend/`)
Vite + React + TypeScript + Tailwind v3 → `frontend/dist/`. Tab title `Local AI Image Gen` (`index.html`).

| File | Role |
|------|------|
| `src/App.tsx` | Root: bootstrap, 4 s status poll, 5 s heartbeat, SSE handler, ref-slot handlers, iterate loop, Load Params |
| `src/store.ts` | `useReducer` global state, `useAppState()` → `{ state, dispatch }`; `autoSizeParams()` |
| `src/types.ts` | `AppStatus`, `GenerateParams`, `SSEEvent`, `OutputItem`, `RefImageSlot`, `Workflow` |
| `src/api.ts` | Typed fetch helpers: `streamGenerate`, `streamBatchGenerate`, `uploadImage`, `uploadFromUrl`, `streamBatchUpscale`, `eraseDetect`, `eraseRemove`, … |
| `src/canvasSize.ts` | `canvasForRef()` / `sizeFamily()` — output size per model family |

Center layout: Canvas (flex 5) / RefImagesRow (flex 4) / Gallery (flex 1) → 50/40/10 % via `style={{ flex: 'N 0 0%' }}`.

### Components
- `Sidebar.tsx` (`w-[576px]`) — Accordions: Model, Parameters, Output Size, LoRA, Upscale (single + batch), Batch Img2Img, Depth Map, Watermark Remover, Video (LTX only), Workflows. `Accordion` renders `{open && children}` (children unmount when closed).
- `RefImagesRow.tsx` — horizontal slot strip: 80×80 thumbnail + role badge, 56×56 mask target (pencil → mask editor), per-slot strength slider. Mask editor = rectangle drag on slot #1's image, saved at its natural size; window-level mouse listeners; Esc/Enter.
- `EraseEditorModal.tsx` — watermark mask editor. Offscreen full-res `maskRef` + `displayRef` (≤760×560). Rectangle + brush (Shift = erase), 45% red overlay. Confirm → `toBlob` → `POST /api/upload` → `onConfirm(maskId, maskUrl)`; upload errors inline.
- `HelpTip.tsx` — ⓘ tooltip, `position:fixed` + `getBoundingClientRect()`, `pointer-events-none`, `z-50`.
- `Canvas.tsx` — result image/video + generating overlay (spinner + %).
- `Gallery.tsx` — horizontal scroll, `draggable` thumbs (gallery → ref slot). Hover: Info, Load Params, Upscale ×4, Delete; video thumbs only Load Params + Delete.
- `SettingsDrawer.tsx` (`w-96`) — output folder, default model, HF login, model + upscaler lists, storage, server log, Model Sources.
- `TopBar.tsx` — brand, model, device, VRAM, "generating…" pulse, settings gear.

### Ref-slot state
```typescript
interface RefImageSlot {
  slotId: number; imageId: string; imageUrl: string
  maskId: string | null; maskUrl: string | null
  strength: number; w?: number; h?: number; keepSize?: boolean
}
```
Actions: `ADD_REF_SLOT` · `REMOVE_REF_SLOT` · `SET_SLOT_MASK` · `CLEAR_SLOT_MASK` · `CLEAR_ALL_SLOTS` · `UPDATE_SLOT_STRENGTH` · `SET_SLOT_DIMS`.
`slotsToParams()` sends all slot image ids but only slot #1's mask and no per-slot strength.

### Gallery "Load Params"
`handleLoadParams` (`App.tsx`) restores prompt, model, size, steps, seed, img_strength, mask_mode, outpaint_align, lora_files, repeat_count, upscale, num_frames, fps, fast_preview, then refs + mask from the companion folder (`keepSize: true`). Device not restored.

## API endpoints
| Method | Path | Notes |
|--------|------|-------|
| POST | `/api/ping` | Heartbeat |
| POST | `/api/stop` | Stop at next step boundary |
| POST | `/api/shutdown` | sendBeacon on tab close; 4 s delay; no-op with `--no-auto-shutdown` |
| GET | `/api/status` | `{model, device, loaded, busy, is_batch_running, vram_gb, total_vram_gb}` — `vram_gb` = current alloc, `total_vram_gb` = GPU ceiling (`app.get_total_memory_gb()`) |
| GET/POST | `/api/models` · `/api/models/load` | List / load model |
| DELETE | `/api/models/{name}` | Delete cached model |
| GET | `/api/models/check-updates` | HF Hub latest hashes |
| POST | `/api/generate` | SSE stream |
| POST | `/api/batch/generate` | SSE, folder of images; yields `batch_progress` |
| POST | `/api/upload` | Temp image → `{id, url}` |
| GET | `/api/temp/{id}` | Serve temp file |
| GET | `/api/outputs` | Recent outputs + sidecar data |
| GET/POST | `/api/workflows` · `/api/workflows/{name}` · `/api/workflows/save` · `/api/workflows/import` | Workflow CRUD + ComfyUI import |
| GET | `/api/lora/list` | `{files:[{name,path,model_type}]}`, `model_type` = `flux`/`zimage`/`unknown` |
| POST | `/api/upscale/upload` · `/api/upscale/batch` · `/api/upscale/single` | Upscale |
| GET | `/api/open-file-dialog` · `/api/open-folder-dialog` | macOS pickers → `{path, cancelled}` |
| POST | `/api/logs/save` | Snapshot `logs/server.log` |
| GET/POST | `/api/settings` | App settings (`app_settings.json`) |
| GET | `/api/storage` | Directory sizes |
| GET/POST | `/api/hf/status` · `/api/hf/login` · `/api/hf/logout` | HF auth |
| POST | `/api/depth-map` | `{filename, model_repo}`, `ThreadPoolExecutor(1)` |
| POST | `/api/erase/detect` | FFT heuristic → `{image_id, image_url, mask_id, mask_url}` |
| POST | `/api/erase` | LaMa fill → `{url, filename}`; `_erased_2.png` collision suffix |
| GET | `/api/model-sources/discover` | Scan HF orgs + `mps` tag, merge new → `{added, sources}`; base entries filtered to `app.KNOWN_MODELS` |

### SSE events
```json
{"type":"progress","message":"Step 5/20","step":5,"total":20}
{"type":"image","url":"/api/output/foo.png","info":"Seed: 42 | Model: … | Mode: …"}
{"type":"done"}  {"type":"error","message":"…"}
{"type":"batch_progress","current":3,"total":12,"filename":"photo_003.jpg"}
```

## Model Sources (Settings drawer)
- List shows only loadable base models: `server._drop_unusable_base()` keeps base repos in `app.KNOWN_MODELS`, on read and on discover (self-heals `model_sources.json`).
- Grouped Models / LoRAs / Upscalers (`TYPE_GROUPS`). `recommendedSourceId()` tags the largest-VRAM image model (LTX excluded) fitting in 90% of `total_vram_gb` as ★ Recommended.
