# Depth Map Generation — Design Spec
**Date:** 2026-04-01  
**Status:** Approved

## Overview

Add a "Depth Map" button to the Gallery hover buttons that generates an accurate depth map from any gallery image using Depth Anything 3 (DA3MONO-LARGE). The output is saved alongside the original image in the output folder and appears in the gallery automatically. The active model is user-selectable via a new section in SettingsDrawer, so future models can be swapped in without code changes.

## Files Changed

| File | Change |
|------|--------|
| `core/depth_map.py` | New — model loader + inference |
| `server.py` | Add `POST /api/depth-map`, extend `GET/POST /api/settings` |
| `frontend/src/api.ts` | Add `generateDepthMap()` |
| `frontend/src/components/Gallery.tsx` | Add 5th hover button (Layers icon, teal) |
| `frontend/src/App.tsx` | Add `handleDepthMapGalleryItem` callback |
| `frontend/src/components/SettingsDrawer.tsx` | Add "Depth Map Model" section |
| `frontend/src/store.ts` | Add `depth_model_repo` to `GenerateParams`, bootstrapped from settings |

`frontend/src/types.ts` — no changes needed.

## Backend

### `core/depth_map.py`

```
load_depth_model(repo_id: str) -> pipeline
  - Uses transformers AutoImageProcessor + AutoModelForDepthEstimation
  - Device: MPS (torch.backends.mps.is_available()) → CPU fallback
  - Model cached in module-level dict: _model_cache: dict[str, pipeline]
  - HF_HUB_CACHE already set to ./models/ by app.py env vars

generate_depth_map(image_path: str, repo_id: str, invert: bool = True) -> bytes
  - Opens image with PIL
  - Runs depth pipeline
  - Normalises output to 0.0–1.0 float
  - Inverts by default: depth = 1.0 - depth  (white=near, black=far — Blender convention)
    DA3 predicts depth directly (larger=farther), so inversion is required.
    DA2 predicted disparity (larger=nearer) so did NOT need inversion — this is a DA3 gotcha.
  - Scales to 0–65535, saves as 16-bit grayscale PNG
  - Returns PNG bytes via io.BytesIO
```

### `POST /api/depth-map`

```python
class DepthMapRequest(BaseModel):
    filename:   str        # filename in output dir
    model_repo: str = "depth-anything/DA3MONO-LARGE"

# Returns:
{
  "url":      "/api/output/<stem>_depth.png",
  "filename": "<stem>_depth.png"
}
```

- Resolves `filename` against `_output_dir()`
- Saves result as `<stem>_depth.png` in same directory
- Returns HTTP 400 if file not found
- Returns HTTP 503 if pipeline busy (`manager.is_busy`)
- No SSE — synchronous POST (inference is 1–3s on MPS)

### `app_settings.json` additions

```json
{
  "depth_model_repo": "depth-anything/DA3MONO-LARGE"
}
```

Loaded/saved via existing `GET/POST /api/settings` endpoints.

## Frontend

### `SettingsDrawer.tsx` — "Depth Map Model" section

- Positioned below the Upscale Models section
- Dropdown with options:
  - `depth-anything/DA3MONO-LARGE` (default, ~1.3 GB, best quality)
  - `depth-anything/DA3-BASE` (~400 MB, faster)
  - `apple/coreml-depth-anything-v2-small` (ANE-optimised, <0.5s, V2 quality — future comparison)
- Selection saved immediately via `POST /api/settings`
- Note: model downloads automatically on first use

### Depth inversion note

DA3 outputs direct depth (large=far). The Blender addon used DA2 which output disparity (large=near) — already white=near without inversion. DA3 **requires explicit inversion** to match Blender's white=near convention. This is handled in `core/depth_map.py` with `invert=True` by default.

### `Gallery.tsx` — 5th hover button

- Icon: `Layers` from lucide-react, teal hover (`hover:bg-teal-600`)
- Visible only for `item.kind !== 'video'`
- Spinner (`Loader2 animate-spin`) while processing — same pattern as Upscale
- `depthMappingItem: string | null` state tracks which item is active
- Props added to Gallery:
  ```typescript
  onDepthMap?:       (item: OutputItem) => void
  depthMappingItem?: string | null
  ```

### `App.tsx` — `handleDepthMapGalleryItem`

```typescript
const handleDepthMapGalleryItem = useCallback(async (item: OutputItem) => {
  setDepthMappingGalleryUrl(item.url)
  try {
    await generateDepthMap({
      filename:    item.name,
      model_repo:  state.params.depth_model_repo ?? 'depth-anything/DA3MONO-LARGE',
    })
    await refreshOutputs()
  } catch (err) {
    dispatch({ type: 'SET_ERROR', message: (err as Error).message })
  } finally {
    setDepthMappingGalleryUrl(null)
  }
}, [state.params.depth_model_repo, dispatch, refreshOutputs])
```

`depth_model_repo` added to `GenerateParams` in `store.ts` and bootstrapped from settings.

### `api.ts`

```typescript
export interface DepthMapResult {
  url:      string
  filename: string
}

export const generateDepthMap = (params: {
  filename:    string
  model_repo?: string
}) => post<DepthMapResult>('/api/depth-map', params)
```

## Model Download

DA3MONO-LARGE (~1.3 GB) downloads on first button click via `transformers` `from_pretrained()`. HF_HUB_CACHE is already set to `./models/` — no extra config needed. The HuggingFace token stored in `huggingface/token` is used automatically if the model requires auth (DA3 is public — no token needed).

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| Source file not found | HTTP 400 → toast error in UI |
| Pipeline busy (generating) | HTTP 503 → toast error in UI |
| Model download fails (no internet) | Exception propagated → toast error |
| MPS not available | Falls back to CPU silently |

## What Does Not Change

Generation, batch img2img, upscale (batch + single), LoRA, workflows, iterative masking — all untouched. No new SSE event types. No changes to `types.ts`.
