# Remove IP-Adapter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the non-functional IP-Adapter feature entirely from the frontend and backend, leaving the ref-slot (img2img) system as the sole image-guidance mechanism.

**Architecture:** Pure removal — delete dead code, types, state, API routes, and backend params. No new abstractions needed. Backend `core/ip_adapter_flux.py` stays on disk but is no longer imported or called.

**Tech Stack:** React 18 + TypeScript (frontend), FastAPI + Python (backend), Vite build (`cd frontend && npm run build`)

---

## Chunk 1: Frontend — types, store, api

### Task 1: Strip IP-Adapter from `types.ts`

**Files:**
- Modify: `frontend/src/types.ts:35-37` (IPA fields in GenerateParams), `71-83` (IpAdapterSlot + IpAdapterStatus interfaces)

- [ ] **Step 1: Remove the three IPA fields from `GenerateParams`**

  In `frontend/src/types.ts`, delete lines 35–37:
  ```typescript
  // DELETE these three lines:
  ip_adapter_image_ids: string[]
  ip_adapter_scales:    number[]
  ip_adapter_enabled:   boolean
  ```
  `GenerateParams` should end at `outpaint_align: string` (line 34).

- [ ] **Step 2: Remove the `IpAdapterSlot` and `IpAdapterStatus` interfaces**

  Delete lines 71–83 (the comment "One IP-Adapter reference image slot", `IpAdapterSlot`, the comment "Status of IP-Adapter weights…", and `IpAdapterStatus`). The file should end at the closing brace of `Workflow`.

- [ ] **Step 3: Verify file**

  The final `types.ts` should contain exactly: `AppStatus`, `GenerateParams` (ending at `outpaint_align`), `RefImageSlot`, `SSEEvent`, `OutputItem`, `Workflow`. No IPA symbols anywhere.

---

### Task 2: Strip IP-Adapter from `store.ts`

**Files:**
- Modify: `frontend/src/store.ts`

- [ ] **Step 1: Remove IPA imports on line 2**

  Change:
  ```typescript
  import type { GenerateParams, OutputItem, AppStatus, RefImageSlot, IpAdapterSlot, IpAdapterStatus } from './types'
  ```
  To:
  ```typescript
  import type { GenerateParams, OutputItem, AppStatus, RefImageSlot } from './types'
  ```

- [ ] **Step 2: Remove IPA state fields from the `State` interface**

  Delete lines 29–31:
  ```typescript
  // DELETE:
  // IP-Adapter
  ipAdapterSlots:   IpAdapterSlot[]
  ipAdapterEnabled: boolean
  ipAdapterStatus:  IpAdapterStatus | null
  ```

- [ ] **Step 3: Remove IPA fields from `DEFAULT_PARAMS`**

  Delete lines 61–63:
  ```typescript
  // DELETE:
  ip_adapter_image_ids: [],
  ip_adapter_scales:    [],
  ip_adapter_enabled:   false,
  ```

- [ ] **Step 4: Remove IPA fields from `initialState`**

  Delete lines 81–83:
  ```typescript
  // DELETE:
  ipAdapterSlots:   [],
  ipAdapterEnabled: false,
  ipAdapterStatus:  null,
  ```

- [ ] **Step 5: Remove 6 IPA action types from the `Action` union**

  Delete lines 113–118:
  ```typescript
  // DELETE:
  | { type: 'ADD_IPA_SLOT';     imageId: string; imageUrl: string }
  | { type: 'REMOVE_IPA_SLOT';  slotId: number }
  | { type: 'UPDATE_IPA_SCALE'; slotId: number; scale: number }
  | { type: 'CLEAR_IPA_SLOTS' }
  | { type: 'TOGGLE_IPA' }
  | { type: 'SET_IPA_STATUS';   status: IpAdapterStatus }
  ```

- [ ] **Step 6: Remove the `ipaSlotsToParams` helper function**

  Delete lines 130–137:
  ```typescript
  // DELETE entire function:
  function ipaSlotsToParams(slots: IpAdapterSlot[], enabled: boolean):
    Pick<GenerateParams, 'ip_adapter_image_ids' | 'ip_adapter_scales' | 'ip_adapter_enabled'> {
    return {
      ip_adapter_image_ids: slots.map(s => s.imageId),
      ip_adapter_scales:    slots.map(s => s.scale),
      ip_adapter_enabled:   enabled && slots.length > 0,
    }
  }
  ```

- [ ] **Step 7: Remove the 6 IPA reducer case blocks**

  Delete lines 248–287 — the full `case 'ADD_IPA_SLOT'`, `case 'REMOVE_IPA_SLOT'`, `case 'UPDATE_IPA_SCALE'`, `case 'CLEAR_IPA_SLOTS'`, `case 'TOGGLE_IPA'`, and `case 'SET_IPA_STATUS'` blocks.

- [ ] **Step 8: Commit**

  ```bash
  git add frontend/src/types.ts frontend/src/store.ts
  git commit -m "refactor: remove IpAdapterSlot, IpAdapterStatus types and all IPA store state"
  ```

---

### Task 3: Strip IP-Adapter from `api.ts`

**Files:**
- Modify: `frontend/src/api.ts`

- [ ] **Step 1: Remove `fetchIpAdapterStatus`, `deleteIpAdapter`, `streamIpAdapterDownload`**

  Locate and delete:
  - The `fetchIpAdapterStatus` export line
  - The `deleteIpAdapter` export line
  - The `ModelExtras` type export (the block with `upscale_models` and `ip_adapter`)
  - The `fetchModelExtras` export line
  - The `deleteUpscaleModel` export line
  - The `streamIpAdapterDownload` async function (the full function)

- [ ] **Step 2: Add back trimmed `ModelExtras` and related exports (upscale only)**

  Replace the deleted `ModelExtras` block with:
  ```typescript
  export type ModelExtras = {
    upscale_models: { name: string; size: string }[]
  }
  export const fetchModelExtras   = () => get<ModelExtras>('/api/models/extras')
  export const deleteUpscaleModel = (filename: string) =>
    del<{ status: string }>(`/api/upscale/${encodeURIComponent(filename)}`)
  ```

- [ ] **Step 3: Verify no IPA symbols remain in `api.ts`**

  ```bash
  grep -n "ip_adapter\|IpAdapter\|ipa\b" frontend/src/api.ts
  ```
  Expected: no output.

- [ ] **Step 4: Commit**

  ```bash
  git add frontend/src/api.ts
  git commit -m "refactor: remove ip-adapter api helpers from api.ts"
  ```

---

## Chunk 2: Frontend — components and App

### Task 4: Update `App.tsx`

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Remove `fetchIpAdapterStatus` from the import on line 7**

  Change the api import to remove `fetchIpAdapterStatus`:
  ```typescript
  // BEFORE (line 7):
  fetchSettings, deleteOutput, fetchIpAdapterStatus,
  // AFTER:
  fetchSettings, deleteOutput,
  ```

- [ ] **Step 2: Remove the IPA bootstrap call (lines 174–176)**

  Delete:
  ```typescript
  fetchIpAdapterStatus()
    .then(s => dispatch({ type: 'SET_IPA_STATUS', status: s }))
    .catch(() => {})
  ```

- [ ] **Step 3: Remove the three IPA prop pass-throughs to `<Sidebar>` (lines 429–431)**

  Delete from the `<Sidebar>` JSX element:
  ```typescript
  // DELETE these three lines:
  ipAdapterSlots={state.ipAdapterSlots}
  ipAdapterEnabled={state.ipAdapterEnabled}
  ipAdapterStatus={state.ipAdapterStatus}
  ```

- [ ] **Step 4: Verify no IPA symbols remain in App.tsx**

  ```bash
  grep -n "ip_adapter\|IpAdapter\|ipAdapter\|SET_IPA\|TOGGLE_IPA" frontend/src/App.tsx
  ```
  Expected: no output.

- [ ] **Step 5: Commit**

  ```bash
  git add frontend/src/App.tsx
  git commit -m "refactor: remove IPA bootstrap call and prop threading from App.tsx"
  ```

---

### Task 5: Update `Sidebar.tsx`

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`
- Delete: `frontend/src/components/IpAdapterPanel.tsx`

- [ ] **Step 1: Remove IPA type imports (line 7)**

  Change:
  ```typescript
  // BEFORE:
  import type { GenerateParams, IpAdapterSlot, IpAdapterStatus } from '../types'
  // AFTER:
  import type { GenerateParams } from '../types'
  ```

- [ ] **Step 2: Remove `fetchIpAdapterStatus` and `IpAdapterPanel` imports (lines 9–10)**

  In the api import on line 9, remove `fetchIpAdapterStatus`.
  Delete the `import IpAdapterPanel from './IpAdapterPanel'` line entirely.

- [ ] **Step 3: Remove IPA props from `SidebarProps` (lines 695–697)**

  Delete:
  ```typescript
  // DELETE:
  ipAdapterSlots:       IpAdapterSlot[]
  ipAdapterEnabled:     boolean
  ipAdapterStatus:      IpAdapterStatus | null
  ```

- [ ] **Step 4: Remove IPA props from the function destructure (line 712)**

  Change:
  ```typescript
  // BEFORE:
  ipAdapterSlots, ipAdapterEnabled, ipAdapterStatus, dispatch,
  // AFTER:
  dispatch,
  ```
  (keeping all other destructured props that remain)

- [ ] **Step 5: Remove the `useEffect` that calls `fetchIpAdapterStatus` (lines 721–724)**

  Delete the entire useEffect block:
  ```typescript
  // DELETE:
  useEffect(() => {
    fetchIpAdapterStatus()
      .then(s => dispatch({ type: 'SET_IPA_STATUS', status: s }))
      .catch(() => {})
  }, [dispatch])
  ```

- [ ] **Step 6: Remove the IP-Adapter accordion section**

  Locate the accordion section that renders `<IpAdapterPanel ... />` (around line 800) and delete the entire accordion item — from its opening `<div>` or accordion trigger through the closing tag that wraps `<IpAdapterPanel>`.

- [ ] **Step 7: Delete `IpAdapterPanel.tsx`**

  ```bash
  rm frontend/src/components/IpAdapterPanel.tsx
  ```

- [ ] **Step 8: Verify no IPA symbols remain in Sidebar.tsx**

  ```bash
  grep -n "ip_adapter\|IpAdapter\|ipAdapter\|IpAdapterPanel" frontend/src/components/Sidebar.tsx
  ```
  Expected: no output.

- [ ] **Step 9: Commit**

  ```bash
  git add frontend/src/components/Sidebar.tsx
  git rm frontend/src/components/IpAdapterPanel.tsx
  git commit -m "refactor: remove IpAdapterPanel and all IPA props from Sidebar"
  ```

---

### Task 6: Update `SettingsDrawer.tsx`

**Files:**
- Modify: `frontend/src/components/SettingsDrawer.tsx`

- [ ] **Step 1: Remove `deleteIpAdapter` from the api import (line 9)**

  Change:
  ```typescript
  // BEFORE:
  fetchModelExtras, deleteUpscaleModel, deleteIpAdapter,
  // AFTER:
  fetchModelExtras, deleteUpscaleModel,
  ```
  Also remove `type ModelExtras` re-import if it was imported separately — keep it if it's the only `ModelExtras` usage.

- [ ] **Step 2: Remove `deletingIpa` state variable (line 49)**

  Delete:
  ```typescript
  // DELETE:
  const [deletingIpa, setDeletingIpa]         = useState(false)
  ```

- [ ] **Step 3: Remove `handleDeleteIpa()` function (lines 184–196)**

  Delete the entire function:
  ```typescript
  // DELETE:
  async function handleDeleteIpa() {
    setDeletingIpa(true)
    try {
      await deleteIpAdapter()
      const e = await fetchModelExtras()
      setExtras(e)
      await fetchStorage().then(setStorage).catch(() => {})
    } catch (e: unknown) {
      setStatusMsg(`Delete failed: ${(e as Error).message}`)
    } finally {
      setDeletingIpa(false)
    }
  }
  ```

- [ ] **Step 4: Remove the IP-Adapter JSX block and fix the render guard**

  Locate the "Other Models" section render guard (around line 510):
  ```typescript
  // BEFORE:
  {extras && (extras.upscale_models.length > 0 || extras.ip_adapter) && (
  ```
  Change to:
  ```typescript
  // AFTER:
  {extras && extras.upscale_models.length > 0 && (
  ```

  Then delete the entire IP-Adapter JSX block inside "Other Models" — the block that starts with `{/* IP-Adapter */}` and renders `extras.ip_adapter.downloaded`, the `handleDeleteIpa` button, etc. Keep the upscale model `.map()` block intact.

- [ ] **Step 5: Verify no IPA symbols remain in SettingsDrawer.tsx**

  ```bash
  grep -n "ip_adapter\|IpAdapter\|ipAdapter\|deletingIpa\|handleDeleteIpa\|deleteIpAdapter" \
    frontend/src/components/SettingsDrawer.tsx
  ```
  Expected: no output.

- [ ] **Step 6: Build check**

  ```bash
  cd frontend && npm run build 2>&1 | tail -20
  ```
  Expected: `✓ built in ...ms` with zero TypeScript errors. If errors appear, fix them before proceeding.

- [ ] **Step 7: Commit**

  ```bash
  git add frontend/src/components/SettingsDrawer.tsx
  git commit -m "refactor: remove IPA delete handler and row from SettingsDrawer"
  ```

---

## Chunk 3: Backend

### Task 7: Strip IP-Adapter from `server.py`

**Files:**
- Modify: `server.py`

- [ ] **Step 1: Remove the ip_adapter_flux import block (lines 158–164)**

  Delete:
  ```python
  from core.ip_adapter_flux import (
      is_downloaded        as ipa_is_downloaded,
      load_ip_adapter      as ipa_load,
      unload_ip_adapter    as ipa_unload,
      set_scale            as ipa_set_scale,
      download             as ipa_download,
  )
  ```

- [ ] **Step 2: Remove the `_ipa_loaded` global (line 166)**

  Delete:
  ```python
  _ipa_loaded: bool = False   # tracks whether IP-Adapter is injected into current pipe
  ```

- [ ] **Step 3: Remove the three IPA fields from `GenerateRequest` (lines 194–196)**

  Delete:
  ```python
  ip_adapter_image_ids: list[str] = Field(default_factory=list)
  ip_adapter_scales:    list[float] = Field(default_factory=list)
  ip_adapter_enabled:   bool = False
  ```

- [ ] **Step 4: Remove `global _ipa_loaded` and reset in `/api/models/load` (lines 296–298)**

  Change the handler from:
  ```python
  async def api_load_model(req: LoadModelRequest):
      global _ipa_loaded
      status = await _mgr().load_model(req.model_choice, req.device)
      _ipa_loaded = False  # IP-Adapter must be re-injected into new pipeline
      return {"status": status}
  ```
  To:
  ```python
  async def api_load_model(req: LoadModelRequest):
      status = await _mgr().load_model(req.model_choice, req.device)
      return {"status": status}
  ```

- [ ] **Step 5: Remove the IPA image-loading block in the generate endpoint (lines 443–458)**

  Delete:
  ```python
  ip_adapter_images = None
  if req.ip_adapter_enabled and req.ip_adapter_image_ids:
      ip_adapter_images = [_load_pil(fid) for fid in req.ip_adapter_image_ids]
      global _ipa_loaded
      import app as _app_module
      if not _ipa_loaded and _app_module.pipe is not None:
          ...
          _ipa_loaded = True
  ...
  params["ip_adapter_images"]  = ip_adapter_images
  params["ip_adapter_scales"]  = req.ip_adapter_scales or [0.6] * len(req.ip_adapter_image_ids)
  params["ip_adapter_enabled"] = req.ip_adapter_enabled
  ```
  (Delete the entire block; preserve surrounding params dict entries that are not IPA.)

- [ ] **Step 6: Remove the three `/api/ip-adapter/*` route handlers**

  Delete all three route handlers:
  - `GET /api/ip-adapter/status` (`ipa_status()`)
  - `POST /api/ip-adapter/download` (`ipa_download_endpoint()`)
  - `DELETE /api/ip-adapter` (`ipa_delete()`)

- [ ] **Step 7: Clean up `GET /api/models/extras` handler**

  The handler currently reads:
  ```python
  @app.get("/api/models/extras")
  async def api_models_extras():
      from core.ip_adapter_flux import IP_ADAPTER_FILE
      ...
      ipa: dict = {"downloaded": IP_ADAPTER_FILE.exists(), "size": None}
      if IP_ADAPTER_FILE.exists():
          ipa["size"] = _fmt_bytes(IP_ADAPTER_FILE.stat().st_size)
      return {"upscale_models": upscale_list, "ip_adapter": ipa}
  ```

  Change to (keep upscale list, drop IPA parts):
  ```python
  @app.get("/api/models/extras")
  async def api_models_extras():
      """Return info about non-HF-cache models: upscale files."""
      upscale_exts = {".pth", ".safetensors", ".ckpt", ".pt", ".bin"}
      upscale_dir  = ROOT / "upscale_models"
      upscale_list = []
      if upscale_dir.exists():
          for f in sorted(upscale_dir.iterdir()):
              if f.is_file() and f.suffix.lower() in upscale_exts:
                  upscale_list.append({"name": f.name, "size": _fmt_bytes(f.stat().st_size)})
      return {"upscale_models": upscale_list}
  ```

- [ ] **Step 8: Verify no IPA symbols remain in server.py**

  ```bash
  grep -n "ip_adapter\|_ipa\|ipa_load\|ipa_download\|ipa_delete\|ipa_status" server.py
  ```
  Expected: no output.

- [ ] **Step 9: Commit**

  ```bash
  git add server.py
  git commit -m "refactor: remove all IP-Adapter routes and params from server.py"
  ```

---

### Task 8: Strip IP-Adapter from `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Remove the three IPA params from `generate_image()` signature (lines 917–919)**

  Change the function signature from:
  ```python
  def generate_image(
      ...
      ip_adapter_images: list | None = None,
      ip_adapter_scales: list | None = None,
      ip_adapter_enabled: bool = False,
  ):
  ```
  To the same signature without those three lines.

- [ ] **Step 2: Remove the `_ipa_active` block (lines 1097–1101)**

  Delete:
  ```python
  _ipa_active = bool(ip_adapter_enabled and ip_adapter_images)
  if _ipa_active:
      from core.ip_adapter_flux import set_scale as _ipa_set_scale
      _scales = ip_adapter_scales or [0.6] * len(ip_adapter_images)
      _ipa_set_scale(pipe, _scales)
  ```

- [ ] **Step 3: Remove `ip_adapter_image` kwarg from the FLUX inpainting call (line 1124)**

  Change:
  ```python
  **({"ip_adapter_image": ip_adapter_images} if _ipa_active else {}),
  ```
  To nothing — delete that line entirely.

- [ ] **Step 4: Remove `ip_adapter_image` kwarg from the FLUX txt2img call (line 1169)**

  Delete:
  ```python
  **({"ip_adapter_image": ip_adapter_images} if _ipa_active else {}),
  ```

- [ ] **Step 5: Remove the three IPA kwargs from `pipeline.py` (lines 194–196)**

  In `pipeline.py`, locate the `generate_image()` call inside the executor worker. Delete these three lines:
  ```python
  ip_adapter_images = params.get("ip_adapter_images"),
  ip_adapter_scales = params.get("ip_adapter_scales"),
  ip_adapter_enabled= params.get("ip_adapter_enabled", False),
  ```

- [ ] **Step 6: Verify no IPA symbols remain in app.py or pipeline.py**

  ```bash
  grep -n "ip_adapter\|_ipa_active\|ipa_set_scale" app.py pipeline.py
  ```
  Expected: no output.

- [ ] **Step 7: Commit**

  ```bash
  git add app.py pipeline.py
  git commit -m "refactor: remove ip_adapter params and _ipa_active block from generate_image() and pipeline.py"
  ```

---

## Chunk 4: Verification

### Task 9: Full build and smoke test

- [ ] **Step 1: Frontend build — must be zero errors**

  ```bash
  cd frontend && npm run build 2>&1
  ```
  Expected output ends with `✓ built in ...ms`. Any TypeScript error is a blocker — fix before continuing.

- [ ] **Step 2: Verify no IPA symbols anywhere in frontend source**

  ```bash
  grep -rn "ip_adapter\|IpAdapter\|ipAdapter\|IpAdapterPanel\|SET_IPA\|TOGGLE_IPA\|ADD_IPA\|REMOVE_IPA" \
    frontend/src/
  ```
  Expected: no output.

- [ ] **Step 3: Verify no IPA symbols anywhere in backend**

  ```bash
  grep -rn "ip_adapter\|_ipa\|ipa_load\|ipa_download" server.py app.py pipeline.py
  ```
  Expected: no output. (`core/ip_adapter_flux.py` itself will still contain these — that's correct, it stays on disk.)

- [ ] **Step 4: Start server and confirm clean boot**

  ```bash
  venv/bin/python server.py --no-auto-shutdown --port 7860 &
  sleep 3
  curl -s http://localhost:7860/api/status | python -m json.tool
  ```
  Expected: JSON with `model`, `device`, `loaded`, `busy`, `vram_gb` — no errors.

- [ ] **Step 5: Confirm `/api/ip-adapter/*` routes are gone (404)**

  ```bash
  curl -s -o /dev/null -w "%{http_code}" http://localhost:7860/api/ip-adapter/status
  ```
  Expected: `404`.

- [ ] **Step 6: Confirm `/api/models/extras` still works (upscale only)**

  ```bash
  curl -s http://localhost:7860/api/models/extras | python -m json.tool
  ```
  Expected: `{"upscale_models": [...]}` — no `ip_adapter` key.

- [ ] **Step 7: Stop test server**

  ```bash
  pkill -f "server.py --no-auto-shutdown" || true
  ```

- [ ] **Step 8: Final commit**

  ```bash
  git add -A
  git commit -m "refactor: complete IP-Adapter removal — frontend + backend clean"
  ```

  If nothing left to stage, the previous per-task commits are sufficient.
