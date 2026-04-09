# Watermark Remover — Design Spec
Date: 2026-04-09

## Overview
Offline watermark removal feature for Local AI Image Gen. Uses heuristic detection (FFT + contrast thresholding) to auto-locate watermark regions, lets the user refine the mask with rectangle and brush tools, then fills the region using LaMa inpainting (no diffusion model required).

---

## Architecture

5 files changed:

| File | Change |
|------|--------|
| `core/erase.py` | New — heuristic detection + LaMa removal |
| `server.py` | 2 new endpoints: `/api/erase/detect` + `/api/erase` |
| `frontend/src/components/Sidebar.tsx` | New `WatermarkPanel` component + `<Accordion>` entry |
| `frontend/src/api.ts` | 2 new typed helpers |
| `pyproject.toml` | Add `simple-lama-inpainting` dep |

No new types in `types.ts` — result is a standard output image that lands in the Gallery.

---

## Backend

### `core/erase.py`

Same structure as `core/depth_map.py` — module-level model cache, two public functions.

**`detect_watermark(path: Path) → bytes`**
- Load image as numpy array
- Run FFT on each RGB channel; identify high-frequency periodic overlay regions
- Combine with contrast thresholding (semi-transparent overlays show characteristic edge halos)
- Produce binary mask PNG (white = watermark area)
- Return raw PNG bytes
- No model required — pure numpy/PIL

**`remove_watermark(image_path: Path, mask_bytes: bytes) → bytes`**
- Load LaMa model on first call via `simple-lama-inpainting`; cache in module-level dict under key `"lama"`
- Model downloads ~60MB to `./models/lama/` on first use
- Run LaMa inpainting with the provided mask
- LANCZOS-resize result back to source resolution (LaMa may resize internally)
- Return PNG bytes

### Endpoints in `server.py`

**`POST /api/erase/detect`**
- Body: `{file_path: str}`
- Runs `detect_watermark()` in `ThreadPoolExecutor(1)` (same executor as depth map)
- Not gated on `manager.is_busy` — LaMa doesn't use the diffusion pipeline
- Uploads mask as temp file via existing temp upload mechanism
- Returns: `{mask_id: str, mask_url: str}`

**`POST /api/erase`**
- Body: `{file_path: str, mask_id: str}`
- Fetches mask from temp store, runs `remove_watermark()` in `ThreadPoolExecutor(1)`
- Saves result as `{stem}_erased.png` in output dir (`app.DEFAULT_OUTPUT_DIR`)
- Writes `.json` sidecar with source path + operation metadata
- Returns: `{url: str, filename: str}`

---

## Frontend

### Accordion entry (in `Sidebar.tsx`)
```tsx
<Accordion label="Watermark Remover" icon={<Eraser size={13} />}>
  <WatermarkPanel onStatus={setStatus} onRefresh={fetchOutputs} />
</Accordion>
```
Placed after the Depth Map accordion.

### `WatermarkPanel` component (in `Sidebar.tsx`)

**State:**
```ts
filePath: string          // absolute path from file picker
previewUrl: string | null // /api/temp/... of original image
maskId: string | null     // temp id of current mask
maskUrl: string | null    // /api/temp/... of mask overlay
detectionMsg: string      // e.g. "2 region(s) detected" or "nothing detected"
isDetecting: boolean
isRemoving: boolean
showEditor: boolean       // open/close EraseEditorModal
```

**Flow:**
1. **Browse** — calls `/api/open-file-dialog` → populates `filePath` + `previewUrl`
2. **Detect Watermark** button — calls `POST /api/erase/detect` → sets `maskId`/`maskUrl`/`detectionMsg`; compact preview shows image with semi-transparent red mask overlay
3. **Edit Mask** button — sets `showEditor=true` → opens `EraseEditorModal` with image + current mask; on confirm updates `maskId`/`maskUrl`
4. **Remove Watermark** button — calls `POST /api/erase` → on success dispatches result URL as new `OutputItem` (appears in Gallery + Canvas)

### `EraseEditorModal` component (new file: `frontend/src/components/EraseEditorModal.tsx`)

Full-screen modal, same visual style as `MaskEditorModal`.

**Canvas:**
- Draws source image as background
- Draws current mask as semi-transparent red overlay
- All mask edits are accumulated in an offscreen canvas

**Tools (toggle via toolbar icon buttons):**
- **Rectangle** (default, `Square` icon) — drag to add filled rectangle to mask
- **Brush** (`Pencil` icon) — freehand paint; Shift held = erase mode; brush size slider (8–64px)

**Keyboard:**
- Escape = cancel (discard edits)
- Enter = confirm (return updated mask PNG bytes)

**Mouse handling:**
- Window-level `mousemove`/`mouseup` listeners (keeps drag alive past canvas edge), same pattern as `MaskEditorModal`

**Output:**
- On confirm: uploads updated mask canvas as temp file via `/api/upload`, returns `{maskId, maskUrl}` to parent

---

## API helpers in `api.ts`

```ts
eraseDetect(filePath: string): Promise<{mask_id: string, mask_url: string}>
eraseRemove(filePath: string, maskId: string): Promise<{url: string, filename: string}>
```

---

## Dependency

```toml
# pyproject.toml
"simple-lama-inpainting>=0.1.1"
```

Model auto-downloads to `./models/lama/` on first use (~60MB). No entry needed in the model list UI — it's a utility model, not a generation model.

---

## Error handling

- File not found → HTTP 400 with message surfaced in panel status line
- Detection finds nothing → returns blank mask + message "Nothing detected — draw manually"
- LaMa model download fails → HTTP 500 with message; user can retry
- `is_busy` NOT checked — LaMa runs independently of the diffusion pipeline

---

## Out of scope (v1)
- Batch watermark removal (folder of images)
- YOLO/EasyOCR-based detection
- Undo/redo in mask editor
- Preview of result before saving
