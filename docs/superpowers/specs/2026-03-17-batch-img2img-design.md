# Batch Img2Img — Design Spec
Date: 2026-03-17

## Summary
Process a folder of reference images one at a time: each image becomes the base reference (slot #1), the current sidebar prompt and all current params are applied, the output is saved, and the loop continues with the next image.

## Scope
- **In:** folder of images (jpg/jpeg/png/webp), current sidebar params (prompt, model, steps, seed, img_strength, lora_files, etc.)
- **Out:** one generated image per input, saved to the normal output directory. Gallery refreshes after each image.
- **Not in scope:** per-image masks, per-image prompts, custom output folder (uses default output dir).

## Architecture

### Backend — `pipeline.py`
Add `is_batch_running: bool = False` field to `PipelineManager`.
Exposed via `/api/status` so the frontend can disable the main Generate button for the full batch duration.

### Backend — `server.py`
New Pydantic model `BatchGenerateRequest`:
- All fields from `GenerateRequest` except `input_image_ids` / `mask_image_id`
- `input_folder: str` — local path to the images folder

New endpoint `POST /api/batch/generate` — SSE stream:
1. Lists `*.jpg, *.jpeg, *.png, *.webp` in `input_folder`, sorted alphabetically. Returns error if empty.
2. Sets `manager.is_batch_running = True`
3. For each image:
   a. Yields `{type:"batch_progress", current:N, total:M, filename:"foo.jpg"}`
   b. Loads image as PIL, injects as `input_images`
   c. Awaits `manager.generate(params)`, forwarding all events downstream
   d. On `_GenerationStopped` / client disconnect: breaks loop cleanly
4. Sets `manager.is_batch_running = False` in `finally`
5. Yields `{type:"done", processed:N}`

Stop behaviour: client aborts SSE → loop exits between images. Mid-image stop uses existing `_stop_event` mechanism via `POST /api/stop`.

### Frontend — `api.ts`
Add `streamBatchGenerate(params, folder, onEvent, signal)` — mirrors `streamBatchUpscale` signature.

### Frontend — `Sidebar.tsx`
New self-contained `BatchImgImgPanel` component (follows `BatchUpscalePanel` pattern):
- Props: `params: GenerateParams`, `isGenerating: boolean`
- State: `inputFolder`, `running`, `log: string[]`, `progress: {current,total,filename} | null`
- UI: folder picker row (path field + Browse button), progress line (`Image N/M — filename`), scrolling log div (last lines), Run/Stop button
- Disabled entirely when `isGenerating` is true

New accordion **"Batch Img2Img"** inserted after the Upscale accordion in `Sidebar.tsx`.

### Frontend — `App.tsx`
- Pass `params` and `isGenerating` down to the new panel via `Sidebar` props (already available in scope)
- After each generated image event in batch stream: call `refreshOutputs()` so gallery updates live

## SSE Event additions
```json
{"type":"batch_progress","current":3,"total":12,"filename":"photo_003.jpg"}
```
All existing event types (`progress`, `image`, `video`, `error`, `done`) flow through unchanged — the frontend log panel renders them as lines.

## Error handling
- Empty folder → immediate SSE error event, no generation
- Single image fails → log the error, continue to next image (don't abort batch)
- Image unreadable → log warning, skip
- All errors surfaced in the log panel

## Files changed
| File | Change |
|------|--------|
| `pipeline.py` | Add `is_batch_running` field, expose in `current_status()` |
| `server.py` | `BatchGenerateRequest` model + `POST /api/batch/generate` endpoint |
| `api.ts` | `streamBatchGenerate()` helper |
| `Sidebar.tsx` | `BatchImgImgPanel` component + new accordion |
| `App.tsx` | Pass `params` to Sidebar (if not already), call `refreshOutputs` on batch image events |
