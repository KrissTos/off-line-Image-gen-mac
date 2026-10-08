# Architecture reference

Deep reference for `off-line-Image-gen-mac`. Operational rules live in `CLAUDE.md`; this file is
consulted when working on a specific layer. Update it in the same session as the code it describes.

## Backend files

| File | Role |
|------|------|
| `server.py` | FastAPI. Serves `frontend/dist/`, all routes `/api/*`, SSE via `StreamingResponse`, HTTP 423 when pipeline busy. Each generation → one run folder via `core/run_store.py`; generation runs in its own asyncio task (`_run_events`, `_RUN_TASKS`) so outputs finishing after a client disconnect are still recorded. Suppresses resource_tracker semaphore warning at import (`warnings.filterwarnings`). |
| `pipeline.py` | `PipelineManager` singleton wrapping `app.generate_image()` in `asyncio.Lock` + `ThreadPoolExecutor(1)`; turns each `(image, video, status)` yield into an SSE dict. `auto_save=False` prevents double-saving. Stop = `threading.Event` + `_GenerationStopped` raised from the step callback; `finally` always runs `gc.collect()` + `torch.mps.empty_cache()`. `is_batch_running: bool`; `stop_requested` = public accessor for `_stop_event.is_set()`. |
| `app.py` | Pure backend logic. `generate_image()` initialises `image = None` / `video_frames = None` each repeat iteration. `lora_files: list[dict]` (legacy `lora_file/lora_strength` still merged at call time); `current_lora_paths: list`. |
| `generate.py` | CLI, Z-Image Turbo only. |

### Run folders (outputs) and saved workflows
One self-contained folder per generation; saved workflows use the same format without `outputs/`.
```
<output_dir>/
  260927-151805_edit-image-1-a-photo-of/        one run: yymmdd-HHMMSS_<slug(prompt[:30])>, collision -2, -3
    workflow.json
    refs/slot_1.png  slot_2.jpg …               source extension kept
    masks/slot_1.png                            any slot that has a mask
    outputs/
      260927-151805_edit-image-1_s812345.png           <run name[:40]>_s<seed>.<ext>; repeat seed → -2
      260927-151805_edit-image-1_s812345_3520x4736.png upscale (upscaled_from its source, same seed)
<repo>/workflows/
  26-09-27_bathroom_floor/                      yy-mm-dd_<name>; workflow.json + refs/ + masks/
```
- `workflow.json` v2: `version: 2`, `name`, `created` (runs) / `saved` (workflows), every `run_store.PARAM_KEYS`
  scalar (`seed` = requested, -1 random), `ref_slots: [{image, mask|null, strength}]`,
  `outputs: [{file, kind, seed?, upscaled_from?}]`. Paths relative to the folder; loaders reject
  paths escaping it. Every rewrite merges into the existing JSON, so unknown keys survive.
- `core/run_store.py` (pure file logic): `create_run`, `output_filename`, `add_output` (same file →
  replaces its entry; upscale inherits source seed), `list_outputs`, `load` (URLs + `warnings` for
  missing ref/mask files, never fatal), `save_workflow(…, overwrite=)`, `remove_output`, `trash`
  (`/usr/bin/trash`), `migrate`.
- A generation that produced nothing (stopped, failed) is trashed once generation has really ended.
- Migration of legacy flat outputs (`X.png` + `X.json` sidecar + `X/` companion + `X_WxH` upscales)
  and v1 workflows (`slot_N_image.png`): `venv/bin/python -m core.run_store migrate <dir> [--apply]`
  (dry run by default, idempotent, moves only; sidecars/companions → Trash). Applied 2026-09-27.
- Default dir `~/Pictures/ultra-fast-image-gen/` (`app.DEFAULT_OUTPUT_DIR`); overridable in Settings (`output_dir`).

### External-writer contract (`/generate` skill)
- `local_run.py` reads event `path` (absolute, inside `<run>/outputs/`) then `url`; no change needed.
  Requests without `ref_slots` get slots derived from `input_image_ids` (slot 1: mask + `img_strength`).
- An external writer may create a v2 folder in `<repo>/workflows/<yy-mm-dd>_<job>/` with minimum keys
  `version`, `prompt`, `model_choice`, `width`, `height`, `ref_slots`; everything else falls back to UI
  defaults. Masks: L-mode PNG at slot image resolution, white = regenerate.
- Path A (generate in the app): the run folder is the event `path`'s grandparent
  (`workflow.json`, `refs/`, `masks/`, `outputs/`).
- Path B (fix the mask in the app, Save): Save overwrites that same folder, so the writer re-reads
  `masks/slot_1.png` from the path it wrote.

### Heartbeat / auto-shutdown
- Server exits 60 s after the last `POST /api/ping` (frontend pings every 5 s). Watcher skips while `manager.is_busy`; resets after system sleep.
- Tab close fires `navigator.sendBeacon('/api/shutdown')` → 4 s cancellable countdown (`_shutdown_task`); the next ping cancels it, so a refresh survives.
- `--no-auto-shutdown` sets env `IMAGEGEN_NO_AUTO_SHUTDOWN=1`, read at module import (`uvicorn.run("server:app")` re-imports the module). Test: `tests/test_auto_shutdown_flag.py`.

## Python modules (`core/`)
- `core/depth_map.py` — DA3/DA2 depth; `generate_depth_map(path, repo_id, invert=True)` → 16-bit PNG bytes, white=near; module-level model cache.
- `core/erase.py` — `detect_watermark(path) → bytes` (Laplacian + brightness anomaly, `np.bincount` CC sizing); `remove_watermark(path, mask_bytes) → bytes` (LaMa, `_lama_cache`); mask resize `Image.NEAREST` (binary mask).
- `core/civitai.py` — CivitAI LoRA discovery (pure, network injected): `BASE_FAMILY` (CivitAI `baseModel` → klein-9B / klein-4B / Z-Image), `rows_from_model`, `discover`, `merge_rows`, `annotate`, `http_fetch_models`.
- `core/civitai_install.py` — CivitAI API key (`civitai/token`, mode 600), registry `civitai_installed.json`, `start_download` (background thread, SHA256 + per-family `verify_lora`, `.tmp_<name>.part` temp), `delete_installed`, `trained_trigger`.
- `core/lora_names.py` — `clean_name` / `display_name`: friendly LoRA names for the dropdown (display only; paths and workflows keep the file name).
- `core/lora_zimage.py` — LoRA injection for Linear/Conv2d via `load_lora_for_pipeline()`.
- `core/lora_flux2.py` — FLUX.2-klein LoRA, PEFT/fal prefix remap.
- `core/quantized_flux2.py` — 4-bit SDNQ + int8 quantization utilities.
- `core/workflow_utils.py` — workflow parse/save/load, ComfyUI importer.

## Frontend (`frontend/`)
Vite + React + TypeScript + Tailwind v3 → `frontend/dist/`. Tab title `Local AI Image Gen` (`index.html`).

| File | Role |
|------|------|
| `src/App.tsx` | Root: bootstrap, 4 s status poll, 5 s heartbeat, SSE handler, ref-slot handlers, iterate loop, `applyWorkflow` |
| `src/store.ts` | `useReducer` global state, `useAppState()` → `{ state, dispatch }`; `autoSizeParams()` |
| `src/types.ts` | `AppStatus`, `GenerateParams`, `SSEEvent`, `OutputItem`, `RefImageSlot`, `Workflow` |
| `src/api.ts` | Typed fetch helpers: `streamGenerate`, `streamBatchGenerate`, `uploadImage`, `uploadFromUrl`, `streamBatchUpscale`, `eraseDetect`, `eraseRemove`, … |
| `src/workflow.ts` | `workflowToParams(wf, {seed})` — pure run/workflow → params (`!= null`, seed 0 survives); node-tested |
| `src/canvasSize.ts` | `canvasForRef()` / `sizeFamily()` — output size per model family |
| `src/mask/maskOps.ts`, `history.ts`, `viewMath.ts` | Pure mask editor logic: combine/invert/grow/shrink ops, undo/redo history, screen↔image view math. Node-tested (`npm test`), no DOM. |

Center layout: Canvas (flex 5) / RefImagesRow (flex 4) / Gallery (flex 1) → 50/40/10 % via `style={{ flex: 'N 0 0%' }}`.

### Components
- `Sidebar.tsx` (`w-[576px]`) — Accordions: Model, Parameters, Output Size, LoRA, Upscale (single + batch), Batch Img2Img, Depth Map, Watermark Remover, Video (LTX only), Workflows. `Accordion` renders `{open && children}` (children unmount when closed).
- `RefImagesRow.tsx` — two sections: **Base** (slot #1; empty → "+ base img") and **References** (slots #2+, "+ ref img" disabled until a base exists). Each card: thumbnail + role badge (click = enlarge in `SlotLightbox.tsx`: Esc / arrows between slots / Replace button; Ctrl/Cmd+click or drop a file/gallery image = replace; helpers `neighborSlot`, `isReplaceClick` in `src/slots.ts`), 56×56 mask box (empty → "draw mask", filled → click to edit, X clears), per-slot strength slider. A ref card dragged onto the base swaps them (`application/x-ref-slot` drag type); dragged onto another ref it swaps their numbers (`swapRefs`: whole card — image, dims, mask, strength — moves, ids stay positional; `SWAP_REFS`). The base X shows only when no refs are left, so refs never shift into the base. The mask box opens `MaskEditor.tsx` on slot #1's image. Slot #2+ boxes carry a hint that their mask is only used by Iterate Masks, and read "unused" outside Inpainting Pipeline mode.
- `MaskEditor.tsx` — full-screen SAM click-to-mask editor (replaces the old rectangle-drag modal). SAM point/box add+subtract, brush, polygon, invert, grow/shrink, undo/redo; keys S/D/B/P select tools, Alt = subtract everywhere, Shift+click = refine last SAM object, wheel = zoom at cursor, +/− = zoom at centre, 0 = fit, [ ] = brush size, Space+drag = pan, Enter = apply (or close polygon), Esc = cancel. 96 px left column: tools, brush slider (Brush tool only), grow/shrink, undo/redo, zoom −/%/+ · Fit · 100%; Import… loads a PNG (white = masked) as the mask, undoable; Mask color swatches/picker + Opacity slider recolor the overlay (display only, saved in `localStorage`); each control has a `HelpTip` hover popup, ⓘ top-right of the canvas lists navigation keys. Saved mask → `onApply(File)` → existing `uploadImage` → `SET_SLOT_MASK` path, unchanged downstream.
- `EraseEditorModal.tsx` — watermark mask editor. Offscreen full-res `maskRef` + `displayRef` (≤760×560). Rectangle + brush (Shift = erase), 45% red overlay. Confirm → `toBlob` → `POST /api/upload` → `onConfirm(maskId, maskUrl)`; upload errors inline.
- `HelpTip.tsx` — ⓘ tooltip, `position:fixed` + `getBoundingClientRect()`, `pointer-events-none`, `z-50`. `text` accepts JSX; pass `children` to use them as the hover trigger instead of the ⓘ.
- `Canvas.tsx` — result image/video + generating overlay (spinner + %); `stale` dims the result with a badge after a saved-workflow load (`resultStale` in the store), click or a new result clears it.
- `Gallery.tsx` — toggle (top-right) between horizontal strip (wheel scrolls sideways) and vertical auto-fill grid (native scroll, letterboxed thumbs); choice kept in `localStorage['gallery.layout']`. `draggable` thumbs (gallery → ref slot). Click = open `GalleryPreview`; Cmd/Ctrl-click = load the whole run directly (see below). Hover: Info, Upscale ×4, Delete; video thumbs only Delete.
- `GalleryPreview.tsx` — modal portaled to `document.body`: image/video, prompt (Copy), params + LoRAs, ref thumbs (img 1 · base / img N, mask badge), "Upscaled from" link, Left/Right/Esc (`keyAction`: arrows ignored with a modifier or while a video/input has focus), Tab cycles inside the dialog, focus returns to the opener on close, backdrop closes only on press+release (a text selection ending there doesn't), Load in workflow / Download / Delete. Fetches `loadRun(item.run)`; legacy outputs without a run show the image only (Load disabled). Pure helpers in `src/previewModel.ts` (`paramRows`, `loraRows`, `neighbor`, `upscaleSource`), tested in `frontend/tests/previewModel.test.ts`. `App.tsx` owns `previewUrl`; "Load" = `handleLoadOutput` (old gallery-click behavior).
- Workflows panel (`Sidebar.tsx` `WorkflowPanel`) — after a saved workflow loads, "Save (overwrite <name>)" rewrites that folder; "Save as new" makes a copy. A gallery run load clears the target.
- `SettingsDrawer.tsx` (`w-96`) — output folder, default model, model + upscaler lists, storage, server log, Model Sources (collapsible categories Models / LoRAs / Upscalers; LoRAs nested in family folders klein-9B, klein-4B, klein, Z-Image, LTX-Video 0.9, Other; function chips filter a folder; open state in `localStorage['modelSources.open']`; rows show a 2-line description). Pure grouping in `src/sourceGroups.ts` (also `civitaiRowState`, `formatSize`, `progressPct`, `civitaiNote`; installed rows sort first), folder header in `SourceFolder.tsx`, one "Accounts & API keys" card at the top of Model Sources holding the HuggingFace token row (`HuggingFaceLogin.tsx`, props from the drawer) and the CivitAI key + NSFW toggle (`CivitaiKeyPanel.tsx`), per-row Download / Installed / Update / Delete in `CivitaiRowActions.tsx` (polls the job; fires the window event `lora-library-changed` so the Sidebar LoRA panel reloads).
- `TopBar.tsx` — brand, model, device, VRAM, "generating…" pulse, settings gear.

### Ref-slot state
```typescript
interface RefImageSlot {
  slotId: number; imageId: string; imageUrl: string
  maskId: string | null; maskUrl: string | null
  strength: number; w?: number; h?: number; keepSize?: boolean
}
```
Actions: `ADD_REF_SLOT` · `REMOVE_REF_SLOT` · `REPLACE_SLOT_IMAGE` · `SWAP_WITH_BASE` · `SET_SLOT_MASK` · `CLEAR_SLOT_MASK` · `CLEAR_ALL_SLOTS` · `UPDATE_SLOT_STRENGTH` · `SET_SLOT_DIMS`.
Slot-list logic lives in pure `src/slots.ts` (tested by `tests/slots.test.ts`). Every mask is drawn on the base image, so a base change (replace/swap) clears all masks; replacing a ref keeps them.
`slotsToParams()` sends all slot image ids and slot #1's mask to the pipeline; generate also sends `ref_slots: [{imageId, maskId, strength}]` so the run records every slot.

### Workflow restore (`applyWorkflow`)
One path for gallery clicks (`GET /api/runs/{run}`, seed = clicked output's seed; ignored while
generating) and saved workflows (`GET /api/workflows/{name}`): `workflowToParams()` → `SET_PARAMS`,
`CLEAR_ALL_SLOTS`, then each slot image/strength/mask with `keepSize: true`; missing files are
listed in the status line. Guarded by `isRestoringWorkflow`; the Save overwrite target is set only
when the restore actually runs.

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
| GET | `/api/outputs?limit=N` | `run_store.list_outputs`: newest first; item `name` = `<run>/outputs/<file>`, plus `run`, `file`, `seed`, run params |
| GET | `/api/output/{path}` | Serve a file under the output dir (traversal → 400) |
| DELETE | `/api/output/{path}` | Remove one output; the run's last one → run folder to macOS Trash |
| GET | `/api/runs/{run}` | `run_store.load`, URLs under `/api/output/<run>/…`, `warnings` |
| GET/POST | `/api/workflows` · `/api/workflows/{name}` · `/api/workflows/save` · `/api/workflows/import` | Saved workflows (v2 folders); save takes optional `overwrite` (folder name, 400 if outside `workflows/`) → `{status, name}`; ComfyUI import |
| GET | `/api/workflow-assets/{name}/{path}` | Serve `refs/…` / `masks/…` of a saved workflow |
| GET | `/api/lora/list` | `{files:[{name,display,path,model_type,variant,trigger…}]}`, `model_type` = `flux`/`zimage`/`unknown`; `display` = CivitAI model name (registry) else the file name without extension, underscores as spaces (`core/lora_names.py`); `name` stays the identity |
| POST | `/api/upscale/upload` · `/api/upscale/batch` · `/api/upscale/single` | Upscale; gallery `single` takes `<run>/outputs/<file>`, writes beside it and records it in the run |
| GET | `/api/open-file-dialog` · `/api/open-folder-dialog` | macOS pickers → `{path, cancelled}` |
| POST | `/api/logs/save` | Snapshot `logs/server.log` |
| GET/POST | `/api/settings` | App settings (`app_settings.json`) |
| GET | `/api/storage` | Directory sizes |
| GET/POST | `/api/hf/status` · `/api/hf/login` · `/api/hf/logout` | HF auth |
| POST | `/api/depth-map` | `{filename, model_repo}`, `ThreadPoolExecutor(1)` |
| POST | `/api/erase/detect` | FFT heuristic → `{image_id, image_url, mask_id, mask_url}` |
| POST | `/api/erase` | LaMa fill → `{url, filename}`; `_erased_2.png` collision suffix |
| GET | `/api/segment/status` | `{loaded}`: SAM weights in memory (no load); mask editor shows a first-load overlay + "SAM ready" toast when false |
| POST | `/api/segment/prepare` | `{image_id}` → `{ready, ms}`; embeds the image once (SAM encoder), lazy model load |
| POST | `/api/segment` | `{image_id, points, box}` → PNG mask (255=object); serialized on a dedicated 1-thread executor |
| GET | `/api/model-sources/discover` | "Update": scan HF orgs + `mps` tag, screen new repos against what the app can use (`core/model_sources.screen`), merge, fill descriptions from the HF cards; then query CivitAI (`core/civitai.discover`, top 100 per base) → `{added, skipped, described, failed, civitai: {added, failed: [family], updates}, sources}`; skipped repos are remembered in `model_sources.json` `ignored` |
| GET | `/api/civitai/status` | `{has_key, show_nsfw}`; never returns the key |
| POST/DELETE | `/api/civitai/key` | `{key}` → `{has_key}` (422 on blank/whitespace); delete removes it |
| POST | `/api/civitai/download` | `{version_id}` (must be a listed row) → job state; one job per version |
| GET | `/api/civitai/download/{version_id}` | `{state: idle/queued/downloading/verifying/done/error, bytes, total, error, file}` |
| DELETE | `/api/civitai/{version_id}` | removes a downloaded file (registry files only); 404 unknown, 409 when loaded |

### SSE events
```json
{"type":"progress","message":"Step 5/20","step":5,"total":20}
{"type":"image","url":"/api/output/<run>/outputs/<file>","path":"<abs path>","info":"Seed: 42 | Model: … | Mode: …"}
{"type":"video", …same keys…}
{"type":"done"}  {"type":"error","message":"…"}
{"type":"batch_progress","current":3,"total":12,"filename":"photo_003.jpg"}
```

## Model Sources (Settings drawer)
- List shows only loadable base models: `server._drop_unusable_base()` keeps base repos in `app.KNOWN_MODELS`, on read and on discover (self-heals `model_sources.json`).
- Model Sources filter (`core/model_sources.py`, pure + injected network): `lora_family(name)` decides which of the app's models a LoRA is for (klein 9B/4B/any size, Z-Image, LTX-Video 0.9.x; LTX-2.x, FLUX.1/FLUX.2-dev, SDXL, Qwen, Krea, Wan = unsupported); `prune_list` drops unsupported LoRAs on read unless `custom: true` (set by "Add source") and sets `family`/`function`; `screen` needs a `.safetensors` file (LoRA) or `.safetensors/.pth` plus an upscaler name (upscaler); `enrich` fills empty `description`/`function` from the card (first real sentence; upscalers prefer a name summary like `4x upscaler · ATD architecture`), never overwriting text and retrying only failed fetches (`described` marks done). Keep `lora_family` in sync when a new model family is added.
- CivitAI rows (`core/civitai.py`) live in `model_sources.json` as `type: lora`, `provider: civitai`, id `civ-<modelId>-<family>`; the family comes from the version's `baseModel` (never the name; `prune_list` passes them through). One row per family a model has a usable `.safetensors` version for (newest version). Update rebuilds them (top 100 per base model, anonymous listing), keeping old rows only for a family that failed to fetch or for installed files. NSFW = model `nsfw` flag or version `nsfwLevel & 28` (R/X/XXX; the API's `nsfw` boolean alone misses R-level LoRAs): hidden unless setting `civitai_show_nsfw` is on, installed ones stay visible; `POST /api/model-sources` keeps hidden NSFW rows and strips `installed`/`update`/`installed_version`.
- CivitAI downloads go to `lora_uploads/` (so the LoRA panel sees them) via `core/civitai_install.py`: key sent only to `civitai.com` (not to the storage host it redirects to), file name sanitised (`safe_filename`, collisions with user uploads get `__civ<versionId>`), SHA256 check, `verify_lora` per family (klein: `check_lora_compatibility` + size match; Z-Image: readable only, the klein check rejects its keys), temp name `.tmp_<name>.part` (a `.safetensors` temp would show in `/api/lora/list`), stale `.part` swept after 1 h, a newer version replaces the old file only after it verified. `civitai_installed.json` is the only record of which files came from CivitAI (drives Installed/Update badges and Delete); its `trained` words feed the trigger hint (`origin: civitai`, after seed and user override). About 40% of popular klein LoRAs return 401 without an API key (creator-set "login required").
- Grouped by `groupSources()` (`src/sourceGroups.ts`). `recommendedSourceId()` tags the largest-VRAM image model (LTX excluded) fitting in 90% of `total_vram_gb` as ★ Recommended.
