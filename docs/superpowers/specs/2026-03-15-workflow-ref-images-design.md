# Workflow — Save & Restore Reference Images, Masks, and Slot Settings

**Date:** 2026-03-15
**Status:** Approved

## Problem

When saving a workflow from the React UI, reference images, per-slot masks, per-slot strength values, `mask_mode`, and `outpaint_align` are not persisted. On load, the full inpainting/img2img setup is lost.

Root cause: `SaveWorkflowRequest` (FastAPI) carries only scalar params. The Gradio `save_workflow()` in `app.py` has an `input_images` parameter but it is never called from the FastAPI path.

## Approach

Option A — extend the save request with slot data; `api_save_workflow` in `server.py` implements saving fully (bypassing `a.save_workflow()`), copies temp files into the workflow folder at save time. A new asset-serving route exposes those files. On load the backend returns asset URLs; the frontend re-uploads them via `uploadFromUrl()` to get fresh temp IDs, then populates ref slot state.

## Data Model

### `workflow.json` additions

```json
{
  "ref_slots": [
    { "image": "slot_1_image.png", "mask": "slot_1_mask.png", "strength": 0.8 },
    { "image": "slot_2_image.png", "mask": null,               "strength": 1.0 }
  ],
  "mask_mode":      "Inpainting Pipeline (Quality)",
  "outpaint_align": "center"
}
```

- `ref_slots` is ordered (slot #1 first).
- `mask` is `null` when the slot has no mask.
- A slot whose temp image file is missing at save time is **omitted from the array entirely** — no stub entry is written.
- Old workflows without `ref_slots` load as before (slots left empty).

### Asset filenames

`slot_{N}_image.png` and `slot_{N}_mask.png` where N is 1-based slot index, assigned sequentially for successfully saved slots only.

## Backend Changes

### `SaveWorkflowRequest` (`server.py`)

Add fields:
```python
ref_slots:      list[dict] = []   # [{imageId: str, maskId: str|null, strength: float}]
mask_mode:      str = ""
outpaint_align: str = ""
```

### `api_save_workflow` (`server.py`) — full rewrite

**Bypass `a.save_workflow()` entirely.** The FastAPI handler owns the complete save logic. This avoids the problem of `a.save_workflow()` computing and hiding the `wf_dir` path.

```python
@app.post("/api/workflows/save")
async def api_save_workflow(req: SaveWorkflowRequest):
    a = _app()
    os.makedirs(a.WORKFLOWS_DIR, exist_ok=True)
    from datetime import datetime
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom      = (req.name or "").strip().replace(" ", "_")
    slug        = "".join(c if c.isalnum() else "_" for c in (req.prompt or "")[:30]).strip("_")
    folder_name = f"{timestamp}_{custom}" if custom else (f"{timestamp}_{slug}" if slug else timestamp)
    wf_dir      = Path(a.WORKFLOWS_DIR) / folder_name
    wf_dir.mkdir(parents=True, exist_ok=True)

    # Copy ref slot files
    saved_slots = []
    for idx, slot in enumerate(req.ref_slots, start=1):
        img_src = TEMP_DIR / slot["imageId"]
        if not img_src.exists():
            continue  # skip slot entirely if image temp file is missing
        img_dst = wf_dir / f"slot_{idx}_image.png"
        shutil.copy2(img_src, img_dst)
        mask_fname = None
        if slot.get("maskId"):
            mask_src = TEMP_DIR / slot["maskId"]
            if mask_src.exists():
                mask_dst = wf_dir / f"slot_{idx}_mask.png"
                shutil.copy2(mask_src, mask_dst)
                mask_fname = f"slot_{idx}_mask.png"
        saved_slots.append({
            "image":    f"slot_{idx}_image.png",
            "mask":     mask_fname,
            "strength": float(slot.get("strength", 1.0)),
        })

    data = {
        "name":              req.name or folder_name,
        "timestamp":         timestamp,
        "prompt":            req.prompt,
        "height":            req.height,
        "width":             req.width,
        "steps":             req.steps,
        "seed":              req.seed,
        "guidance":          req.guidance,
        "device":            req.device,
        "model_choice":      req.model_choice,
        "model_source":      req.model_source,
        "lora_strength":     req.lora_strength,
        "img_strength":      req.img_strength,
        "repeat_count":      req.repeat_count,
        "upscale_enabled":   req.upscale_enabled,
        "upscale_model_path": req.upscale_model_path,
        "num_frames":        req.num_frames,
        "fps":               req.fps,
        "ref_slots":         saved_slots,
        "mask_mode":         req.mask_mode,
        "outpaint_align":    req.outpaint_align,
    }
    with open(wf_dir / "workflow.json", "w") as f:
        json.dump(data, f, indent=2)

    return {"status": f"✓ Saved: {folder_name}", "name": folder_name}
```

### Asset-serving route (`server.py`)

Use the prefix `/api/workflow-assets/` (distinct from `/api/workflows/{name:path}`) to avoid route shadowing:

```python
@app.get("/api/workflow-assets/{name}/{filename}")
async def api_workflow_asset(name: str, filename: str):
    a    = _app()
    path = Path(a.WORKFLOWS_DIR) / name / filename
    if not path.exists():
        raise HTTPException(404, "Asset not found")
    return FileResponse(str(path))
```

Register this route **before** the SPA wildcard.

### `api_load_workflow` (`server.py`)

Extend the returned dict:
```python
ref_slots_out = []
for slot in d.get("ref_slots", []):
    img_file = slot.get("image")
    if not img_file or not os.path.exists(os.path.join(wf_dir, img_file)):
        continue  # skip missing assets
    mask_url = None
    if slot.get("mask") and os.path.exists(os.path.join(wf_dir, slot["mask"])):
        mask_url = f"/api/workflow-assets/{name}/{slot['mask']}"
    ref_slots_out.append({
        "imageUrl": f"/api/workflow-assets/{name}/{img_file}",
        "maskUrl":  mask_url,
        "strength": slot.get("strength", 1.0),
    })

result["ref_slots"]      = ref_slots_out
result["mask_mode"]      = d.get("mask_mode", "")
result["outpaint_align"] = d.get("outpaint_align", "")
```

## Frontend Changes

### `api.ts`

No changes needed — `saveWorkflow` already accepts `Record<string, unknown>`.

### `Sidebar.tsx` — new props and `WorkflowPanel` threading

`Sidebar` receives one new prop: `refSlots: RefImageSlot[]`. (`mask_mode` and `outpaint_align` are already in `params`.)

`WorkflowPanel` (the sub-component inside `Sidebar`) also receives `refSlots: RefImageSlot[]`. Its `handleSave` builds the payload:

```ts
await saveWorkflow({
  ...params,
  name: saveName.trim(),
  ref_slots: refSlots.map(s => ({
    imageId:  s.imageId,
    maskId:   s.maskId ?? null,
    strength: s.strength,
  })),
  mask_mode:      params.mask_mode,
  outpaint_align: params.outpaint_align,
})
```

### `App.tsx` — pass new prop + extend `handleWorkflowLoad`

Pass `refSlots={state.refSlots}` to `<Sidebar />`.

`handleWorkflowLoad` — after restoring scalar params, restore ref slots **sequentially** (one slot fully dispatched before starting the next, to preserve correct `slotId` assignment from the reducer):

```ts
const slots = wf.ref_slots as Array<{imageUrl:string; maskUrl:string|null; strength:number}> | undefined
if (slots?.length) {
  setStatusMsg(`Restoring ${slots.length} ref slot(s)…`)  // setStatusMsg from useCallback
  dispatch({ type: 'CLEAR_ALL_SLOTS' })
  for (const slot of slots) {
    try {
      const { id, url } = await uploadFromUrl(slot.imageUrl)
      dispatch({ type: 'ADD_REF_SLOT', imageId: id, imageUrl: url })
      // UPDATE_SLOT_STRENGTH uses the slotId just assigned (state.refSlots.length after ADD)
      // slotId is length of refSlots after ADD_REF_SLOT — get it from the dispatched action result
      // Simplest: track index (slots are added in order, slotId = index + 1)
      const slotId = slots.indexOf(slot) + 1
      dispatch({ type: 'UPDATE_SLOT_STRENGTH', slotId, strength: slot.strength })
      if (slot.maskUrl) {
        const { id: mId, url: mUrl } = await uploadFromUrl(slot.maskUrl)
        dispatch({ type: 'SET_SLOT_MASK', slotId, maskId: mId, maskUrl: mUrl })
      }
    } catch {
      // skip slot on error, continue with others
    }
  }
  setStatusMsg(`✓ Loaded workflow with ${slots.length} ref slot(s)`)
}
if (wf.mask_mode)      dispatch({ type: 'SET_PARAM', key: 'mask_mode',      value: wf.mask_mode })
if (wf.outpaint_align) dispatch({ type: 'SET_PARAM', key: 'outpaint_align', value: wf.outpaint_align })
```

**Sequential processing is required.** Do NOT parallelise slot uploads with `Promise.all` — the `ADD_REF_SLOT` reducer assigns `slotId = state.refSlots.length + 1` at dispatch time, so out-of-order dispatches would assign wrong slot IDs.

`setStatusMsg` is already available in `App.tsx` via the existing `setStatusMsg` state setter.

## Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Temp image file missing at save time | Slot omitted from `ref_slots` array entirely (no stub written) |
| Mask temp file missing at save time (image present) | Image saved, `"mask": null` in slot entry |
| Asset file missing from workflow folder on load | Slot skipped; others loaded normally |
| Old workflow (no `ref_slots` key) | Load as before; slots left empty; no error |
| Save with no ref slots | `ref_slots: []` written; no asset files created |
| Upload error during load restore | That slot skipped; load continues with remaining slots |

## File Touch List

| File | Change |
|------|--------|
| `server.py` | Rewrite `api_save_workflow`; add `ref_slots`/`mask_mode`/`outpaint_align` to `SaveWorkflowRequest`; add `/api/workflow-assets/{name}/{filename}` route; extend `api_load_workflow` return |
| `frontend/src/components/Sidebar.tsx` | Add `refSlots: RefImageSlot[]` prop; thread to `WorkflowPanel`; build `ref_slots` in `handleSave` |
| `frontend/src/App.tsx` | Pass `refSlots` prop to Sidebar; extend `handleWorkflowLoad` for sequential slot restore |
| `frontend/src/types.ts` | No change |
| `frontend/src/api.ts` | No change |
| `app.py` | No change (Gradio path unaffected) |
