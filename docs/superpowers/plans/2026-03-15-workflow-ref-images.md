# Workflow Ref-Images Save & Restore — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When saving a workflow, persist all reference images, per-slot masks, per-slot strength values, `mask_mode`, and `outpaint_align` to the workflow folder; restore them fully on load.

**Architecture:** The FastAPI `api_save_workflow` handler is rewritten to own the full save logic (bypassing `a.save_workflow()`), copy temp image/mask files into the workflow folder, and write extended `workflow.json`. A new asset-serving route `/api/workflow-assets/{name}/{filename}` exposes those files. On load, `api_load_workflow` returns asset URLs; the frontend re-uploads them via `uploadFromUrl()` sequentially to populate ref-slot state.

**Tech Stack:** Python 3.11 / FastAPI / Pydantic v2, React 18 / TypeScript / Vite, `shutil` (stdlib)

---

## Chunk 1: Backend — save workflow with ref slots

### Task 1: Extend `SaveWorkflowRequest` and rewrite `api_save_workflow`

**Files:**
- Modify: `server.py` (lines ~246–264 for the Pydantic model, lines ~632–661 for the handler)

- [ ] **Step 1: Add new fields to `SaveWorkflowRequest`**

Find `class SaveWorkflowRequest(BaseModel):` and add three fields at the end of the class:

```python
lora_file:      str = ""          # preserve LoRA round-trip (was in app.save_workflow but missing from request)
ref_slots:      list[dict] = []   # [{imageId: str, maskId: str|null, strength: float}]
mask_mode:      str = ""
outpaint_align: str = ""
```

- [ ] **Step 2: Rewrite `api_save_workflow` to bypass `a.save_workflow()`**

Replace the entire `api_save_workflow` function body with the implementation below. The handler now owns folder creation, file copying, and JSON writing. `a.save_workflow()` is no longer called from this path.

```python
@app.post("/api/workflows/save")
async def api_save_workflow(req: SaveWorkflowRequest):
    from datetime import datetime
    a = _app()
    os.makedirs(a.WORKFLOWS_DIR, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom      = (req.name or "").strip().replace(" ", "_")
    slug        = "".join(c if c.isalnum() else "_" for c in (req.prompt or "")[:30]).strip("_")
    folder_name = f"{timestamp}_{custom}" if custom else (f"{timestamp}_{slug}" if slug else timestamp)
    wf_dir      = Path(a.WORKFLOWS_DIR) / folder_name
    wf_dir.mkdir(parents=True, exist_ok=True)

    # Copy ref slot image/mask files from temp storage into the workflow folder
    saved_slots = []
    for idx, slot in enumerate(req.ref_slots, start=1):
        img_id  = slot.get("imageId") or ""
        if not img_id:
            continue  # skip slot — no imageId provided
        img_src = TEMP_DIR / img_id
        if not img_src.exists():
            continue  # skip slot entirely — image temp file missing
        img_dst = wf_dir / f"slot_{idx}_image.png"
        shutil.copy2(img_src, img_dst)
        mask_fname = None
        mask_id = slot.get("maskId") or ""
        if mask_id:
            mask_src = TEMP_DIR / mask_id
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
        "name":               req.name or folder_name,
        "timestamp":          timestamp,
        "prompt":             req.prompt,
        "height":             req.height,
        "width":              req.width,
        "steps":              req.steps,
        "seed":               req.seed,
        "guidance":           req.guidance,
        "device":             req.device,
        "model_choice":       req.model_choice,
        "model_source":       req.model_source,
        "lora_file":          req.lora_file,
        "lora_strength":      req.lora_strength,
        "img_strength":       req.img_strength,
        "repeat_count":       req.repeat_count,
        "upscale_enabled":    req.upscale_enabled,
        "upscale_model_path": req.upscale_model_path,
        "num_frames":         req.num_frames,
        "fps":                req.fps,
        "ref_slots":          saved_slots,
        "mask_mode":          req.mask_mode,
        "outpaint_align":     req.outpaint_align,
    }
    with open(wf_dir / "workflow.json", "w") as f:
        json.dump(data, f, indent=2)

    return {"status": f"✓ Saved: {folder_name}", "name": folder_name}
```

- [ ] **Step 3: Verify server starts cleanly**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac
source venv/bin/activate
python -c "import server; print('OK')"
```
Expected: `OK` with no import errors.

- [ ] **Step 4: Commit**

```bash
git add server.py
git commit -m "feat(backend): rewrite api_save_workflow to persist ref slots"
```

---

### Task 2: Add asset-serving route and extend `api_load_workflow`

**Files:**
- Modify: `server.py` (add route before SPA wildcard; extend `api_load_workflow`)

- [ ] **Step 1: Add the asset-serving route**

Add this route immediately **before** the SPA catch-all (`/{path:path}`) at the bottom of `server.py`. (Search for `# ── SPA` or `/{path:path}` to find the right location.)

```python
@app.get("/api/workflow-assets/{name}/{filename}")
async def api_workflow_asset(name: str, filename: str):
    """Serve a saved workflow asset (ref image or mask)."""
    a        = _app()
    base     = Path(a.WORKFLOWS_DIR).resolve()
    path     = (base / name / filename).resolve()
    # Guard against path traversal (e.g. name="../../../etc")
    if not str(path).startswith(str(base)):
        raise HTTPException(400, "Invalid path")
    if not path.exists():
        raise HTTPException(404, "Asset not found")
    return FileResponse(str(path))
```

- [ ] **Step 2: Extend `api_load_workflow` to return ref slot data**

`api_load_workflow` currently calls `a.load_workflow(name)` which returns an 18-tuple and maps it to a dict. We need to add `ref_slots`, `mask_mode`, and `outpaint_align` to that dict.

The 18-tuple comes from `app.py`'s `load_workflow()` which reads `workflow.json` directly. Rather than changing `app.py`, read `workflow.json` a second time in `api_load_workflow` to extract the new fields:

Replace the existing `api_load_workflow` function with:

```python
@app.get("/api/workflows/{name:path}")
async def api_load_workflow(name: str):
    a         = _app()
    wf_dir    = Path(a.WORKFLOWS_DIR) / name
    json_path = wf_dir / "workflow.json"

    # Check BEFORE calling a.load_workflow() — missing workflow returns
    # gr.update() objects which are not JSON-serializable.
    if not json_path.exists():
        raise HTTPException(404, f"Workflow not found: {name}")

    try:
        result = a.load_workflow(name)
        keys = ["prompt", "height", "width", "steps", "seed", "guidance",
                "device", "model_choice", "model_source", "lora_strength",
                "img_strength", "repeat_count", "upscale_enabled",
                "upscale_model_path", "num_frames", "fps", "input_images", "status"]
        d = dict(zip(keys, result))
    except Exception as e:
        raise HTTPException(500, str(e))

    # input_images contains PIL Image objects — not JSON-serializable; frontend
    # now uses ref_slots URLs instead.
    d.pop("input_images", None)

    ref_slots_out: list[dict] = []
    with open(json_path) as fh:
        raw = json.load(fh)
    for slot in raw.get("ref_slots", []):
        img_file = slot.get("image")
        if not img_file or not (wf_dir / img_file).exists():
            continue
        mask_url = None
        if slot.get("mask") and (wf_dir / slot["mask"]).exists():
            mask_url = f"/api/workflow-assets/{name}/{slot['mask']}"
        ref_slots_out.append({
            "imageUrl": f"/api/workflow-assets/{name}/{img_file}",
            "maskUrl":  mask_url,
            "strength": slot.get("strength", 1.0),
        })
    d["mask_mode"]      = raw.get("mask_mode", "")
    d["outpaint_align"] = raw.get("outpaint_align", "")
    d["ref_slots"]      = ref_slots_out
    return d
```

- [ ] **Step 3: Verify server starts cleanly**

```bash
python -c "import server; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Smoke-test save endpoint with curl**

Start server in a terminal: `python server.py --port 7860 --no-auto-shutdown`

In another terminal:
```bash
curl -s -X POST http://localhost:7860/api/workflows/save \
  -H "Content-Type: application/json" \
  -d '{"name":"test_wf","prompt":"a cat","height":512,"width":512,"steps":20,"seed":-1,"guidance":3.5,"device":"mps","model_choice":"","model_source":"Local","lora_strength":1.0,"img_strength":1.0,"repeat_count":1,"upscale_enabled":false,"upscale_model_path":"","num_frames":25,"fps":24,"ref_slots":[],"mask_mode":"","outpaint_align":""}' \
  | python -m json.tool
```
Expected: `{"status": "✓ Saved: 2026..._test_wf", "name": "2026..._test_wf"}`

- [ ] **Step 5: Smoke-test load endpoint with curl**

Use the folder name returned in step 4:
```bash
curl -s http://localhost:7860/api/workflows/2026XXXXXX_test_wf | python -m json.tool
```
Expected: JSON with `"ref_slots": []`, `"mask_mode": ""`, `"outpaint_align": ""`

- [ ] **Step 6: Commit**

```bash
git add server.py
git commit -m "feat(backend): add workflow asset route + extend load with ref slots"
```

---

## Chunk 2: Frontend — save ref slots with workflow

### Task 3: Thread `refSlots` prop through `Sidebar` → `WorkflowPanel` and include in save payload

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Add `refSlots` to `SidebarProps`**

Find `interface SidebarProps {` (around line 816) and add:
```ts
refSlots:             RefImageSlot[]
```

Also add `RefImageSlot` to the import from `'../types'` if not already there:
```ts
import type { GenerateParams, RefImageSlot } from '../types'
```

- [ ] **Step 2: Destructure `refSlots` in the `Sidebar` function signature**

Find the destructuring line (around line 837):
```ts
params, models, availableModels, devices, workflows, isGenerating,
hasIteratableMasks, hasRefImage, refImageSize,
onParamChange, onParamsChange, onGenerate, onStop, onIterate,
onWorkflowLoad, onWorkflowRefresh, onStatus,
```
Add `refSlots,` to the list.

- [ ] **Step 3: Add `refSlots` to `WorkflowPanelProps`**

Find `interface WorkflowPanelProps {` (around line 705) and add:
```ts
refSlots: RefImageSlot[]
```

- [ ] **Step 4: Destructure `refSlots` in `WorkflowPanel` function signature**

Find `function WorkflowPanel({ workflows, params, onLoad, onRefresh, onImportComfyUI, onStatus }:` and add `refSlots` to the destructuring.

- [ ] **Step 5: Update `handleSave` in `WorkflowPanel` to include ref slot data**

Find `async function handleSave()` (around line 729). Replace the `saveWorkflow(...)` call:

```ts
async function handleSave() {
  if (!saveName.trim()) return
  try {
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
    onStatus(`✓ Saved workflow: ${saveName}`)
    setSaveName('')
    onRefresh()
  } catch (e: unknown) {
    onStatus(`Error: ${(e as Error).message}`)
  }
}
```

- [ ] **Step 6: Pass `refSlots` from `Sidebar` to `WorkflowPanel`**

Find the `<WorkflowPanel` usage (around line 954) and add the prop:
```tsx
<WorkflowPanel
  workflows={workflows}
  params={params}
  refSlots={refSlots}       {/* ← add this */}
  onLoad={onWorkflowLoad}
  onRefresh={onWorkflowRefresh}
  onImportComfyUI={handleImportComfyUI}
  onStatus={onStatus}
/>
```

- [ ] **Step 7: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend
npm run build 2>&1 | tail -20
```
Expected: build succeeds with no TS errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat(frontend): pass refSlots to WorkflowPanel, include in save payload"
```

---

## Chunk 3: Frontend — pass prop from App.tsx + restore slots on load

### Task 4: Pass `refSlots` to Sidebar and restore slots in `handleWorkflowLoad`

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Pass `refSlots` prop to `<Sidebar />`**

Find the `<Sidebar` usage (around line 436) and add:
```tsx
refSlots={state.refSlots}
```

- [ ] **Step 2: Make `handleWorkflowLoad` async and restore ref slots**

`handleWorkflowLoad` currently is a sync `useCallback`. Replace it entirely:

```ts
const handleWorkflowLoad = useCallback(async (wf: Record<string, unknown>) => {
  // Restore scalar params
  const p: Partial<GenerateParams> = {}
  if (wf.prompt)       p.prompt       = String(wf.prompt)
  if (wf.height)       p.height       = Number(wf.height)
  if (wf.width)        p.width        = Number(wf.width)
  if (wf.steps)        p.steps        = Number(wf.steps)
  if (wf.seed)         p.seed         = Number(wf.seed)
  if (wf.guidance)     p.guidance     = Number(wf.guidance)
  if (wf.model_choice) p.model_choice = String(wf.model_choice)
  if (wf.device)       p.device       = String(wf.device)
  dispatch({ type: 'SET_PARAMS', params: p })

  // Restore ref slots (sequential — slotId assignment depends on dispatch order)
  const slots = wf.ref_slots as Array<{ imageUrl: string; maskUrl: string | null; strength: number }> | undefined
  if (slots?.length) {
    setStatusMsg(`Restoring ${slots.length} ref slot(s)…`)
    dispatch({ type: 'CLEAR_ALL_SLOTS' })
    for (let i = 0; i < slots.length; i++) {
      const slot   = slots[i]
      const slotId = i + 1  // ADD_REF_SLOT assigns slotId = refSlots.length + 1 sequentially
      try {
        const { id, url } = await uploadFromUrl(slot.imageUrl)
        dispatch({ type: 'ADD_REF_SLOT', imageId: id, imageUrl: url })
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

  // Restore mask_mode and outpaint_align
  if (wf.mask_mode)      dispatch({ type: 'SET_PARAM', key: 'mask_mode',      value: wf.mask_mode as string })
  if (wf.outpaint_align) dispatch({ type: 'SET_PARAM', key: 'outpaint_align', value: wf.outpaint_align as string })
}, [dispatch])
```

Also make sure `uploadFromUrl` is in the imports at the top of `App.tsx` (it should already be).

- [ ] **Step 3: Update `WorkflowPanelProps.onLoad` type to accept async callback**

In `Sidebar.tsx`, find `onLoad: (wf: Record<string, unknown>) => void` in `WorkflowPanelProps` and change to:
```ts
onLoad: (wf: Record<string, unknown>) => void | Promise<void>
```

Also update `handleLoad` in `WorkflowPanel` to await it:
```ts
async function handleLoad() {
  if (!selected) return
  try {
    const wf = await loadWorkflow(selected)
    await onLoad(wf)                          // ← await so status fires after slots restore
    onStatus(`✓ Loaded workflow: ${selected}`)
  } catch (e: unknown) {
    onStatus(`Load failed: ${(e as Error).message}`)
  }
}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend
npm run build 2>&1 | tail -20
```
Expected: build succeeds with no TS errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/Sidebar.tsx
git commit -m "feat(frontend): restore ref slots + mask settings on workflow load"
```

---

## Chunk 4: End-to-end verification and push

### Task 5: Manual end-to-end test and push

**Files:** None (verification only)

- [ ] **Step 1: Build the frontend**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npm run build
```
Expected: Build succeeds.

- [ ] **Step 2: Start the server**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac
source venv/bin/activate && python server.py --port 7860 --no-auto-shutdown
```

- [ ] **Step 3: Test save with ref images**

In the browser at `http://localhost:7860`:
1. Load a model
2. Upload 1–2 reference images into the ref-image slots
3. Draw a mask on slot #1
4. Set mask mode to "Inpainting Pipeline (Quality)"
5. Adjust per-slot strength sliders
6. Open Workflows accordion, type a name, click Save
7. Check that the workflow folder in `workflows/` contains `workflow.json` + `slot_1_image.png` + `slot_1_mask.png`
8. Verify `workflow.json` has `ref_slots`, `mask_mode`, `outpaint_align` fields populated

- [ ] **Step 4: Test load restores everything**

1. Clear all ref slots (click × on each)
2. Select the saved workflow from the dropdown and click Load
3. Verify: ref images appear in slots, mask thumbnail appears, per-slot strength matches saved values, mask mode dropdown matches, outpaint_align matches

- [ ] **Step 5: Test old workflow backward compatibility**

1. Open `workflows/` and find an old workflow folder (no `ref_slots` in its JSON)
2. Load it — should load scalar params normally with no errors and empty ref slots

- [ ] **Step 6: Push to GitHub**

```bash
bash ~/.claude/skills/git-pushing/scripts/smart_commit.sh "feat: persist and restore ref images/masks with workflows"
```

---

## Edge Case Reference

| Scenario | Expected |
|----------|----------|
| Temp image missing at save | Slot skipped from `ref_slots` |
| Mask temp missing at save | Image saved, `"mask": null` |
| Asset file missing on load | That slot skipped; others restored |
| Old workflow (no `ref_slots`) | Loads scalar params, slots empty |
| No ref slots at save | `ref_slots: []`, no asset files |
| Upload error during load restore | Slot skipped, load continues |
