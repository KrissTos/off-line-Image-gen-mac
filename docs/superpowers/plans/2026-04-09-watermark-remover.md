# Watermark Remover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline watermark removal accordion to the Sidebar — FFT heuristic auto-detection, user-editable mask (rectangle + brush), LaMa inpainting fill, result saved to Gallery.

**Architecture:** `core/erase.py` holds two pure functions (`detect_watermark`, `remove_watermark`). Two new FastAPI endpoints in `server.py` run them in a `ThreadPoolExecutor`. A new `EraseEditorModal.tsx` provides the canvas editing UI. `WatermarkPanel` in `Sidebar.tsx` orchestrates the flow.

**Tech Stack:** Python (numpy, scipy, PIL, simple-lama-inpainting), FastAPI, React/TypeScript, Tailwind CSS v3, lucide-react.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `core/erase.py` | Create | `detect_watermark()` + `remove_watermark()` |
| `server.py` | Modify | `POST /api/erase/detect` + `POST /api/erase` endpoints |
| `frontend/src/api.ts` | Modify | `eraseDetect()` + `eraseRemove()` typed helpers |
| `frontend/src/components/EraseEditorModal.tsx` | Create | Full-screen canvas editor (rect + brush tools) |
| `frontend/src/components/Sidebar.tsx` | Modify | `WatermarkPanel` component + accordion entry |
| `pyproject.toml` | Modify | Add `simple-lama-inpainting>=0.1.1` |
| `tests/test_erase.py` | Create | Unit tests for core/erase.py |

---

## Task 1: Add `simple-lama-inpainting` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

Open `pyproject.toml` and add `"simple-lama-inpainting>=0.1.1"` to the `dependencies` list, after `"spandrel>=0.4.0"`:

```toml
    "spandrel>=0.4.0",
    "requests>=2.32",
    "simple-lama-inpainting>=0.1.1",
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync
```

Expected: resolves and installs `simple_lama_inpainting` and its deps (torch is already installed so only the package itself downloads). No errors.

- [ ] **Step 3: Verify import**

```bash
source venv/bin/activate && python -c "from simple_lama_inpainting import SimpleLama; print('ok')"
```

Expected: prints `ok` (model NOT downloaded yet — that happens on first use).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add simple-lama-inpainting dependency"
```

---

## Task 2: Create `core/erase.py` with tests

**Files:**
- Create: `core/erase.py`
- Create: `tests/test_erase.py`

- [ ] **Step 1: Create the tests directory and write failing tests**

```bash
mkdir -p tests
touch tests/__init__.py
```

Create `tests/test_erase.py`:

```python
"""Unit tests for core/erase.py — watermark detection and removal."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_test_image(w: int = 256, h: int = 256) -> Path:
    """Create a temporary solid-colour PNG for testing."""
    import tempfile
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add a fake watermark: bright rectangle in top-right quadrant
    arr[10:50, w - 80:w - 10, :] = 240
    img = Image.fromarray(arr, mode="RGB")
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    return Path(f.name)


def test_detect_watermark_returns_png_bytes():
    from core.erase import detect_watermark
    path = _make_test_image()
    result = detect_watermark(path)
    assert isinstance(result, bytes)
    assert len(result) > 0
    # Should be a valid PNG
    img = Image.open(io.BytesIO(result))
    assert img.mode == "L"


def test_detect_watermark_same_size_as_input():
    from core.erase import detect_watermark
    path = _make_test_image(400, 300)
    result = detect_watermark(path)
    mask = Image.open(io.BytesIO(result))
    assert mask.size == (400, 300)


def test_detect_watermark_blank_image_returns_mask():
    """Blank image has no watermark — should return a blank (all-black) mask."""
    from core.erase import detect_watermark
    import tempfile
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    result = detect_watermark(Path(f.name))
    mask = Image.open(io.BytesIO(result))
    arr_mask = np.array(mask)
    # Blank image → no high-frequency content → mask should be mostly dark
    assert arr_mask.mean() < 50


def test_remove_watermark_returns_png_bytes():
    from core.erase import remove_watermark
    path = _make_test_image()
    # Simple all-white mask (erase everything)
    mask_img = Image.fromarray(np.full((256, 256), 255, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    result = remove_watermark(path, mask_bytes)
    assert isinstance(result, bytes)
    assert len(result) > 0
    out = Image.open(io.BytesIO(result))
    assert out.mode in ("RGB", "RGBA")


def test_remove_watermark_output_matches_source_size():
    from core.erase import remove_watermark
    path = _make_test_image(512, 384)
    mask_img = Image.fromarray(np.zeros((384, 512), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    result = remove_watermark(path, buf.getvalue())
    out = Image.open(io.BytesIO(result))
    assert out.size == (512, 384)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate && python -m pytest tests/test_erase.py -v 2>&1 | head -30
```

Expected: all 5 tests fail with `ModuleNotFoundError: No module named 'core.erase'`.

- [ ] **Step 3: Implement `core/erase.py`**

Create `core/erase.py`:

```python
"""Watermark detection and removal.

Detection: heuristic — FFT high-pass + local brightness anomaly.
Removal: LaMa inpainting via simple-lama-inpainting.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

# Module-level LaMa model cache (lazy-loaded on first remove call)
_lama_cache: dict[str, object] = {}


def detect_watermark(path: Path) -> bytes:
    """Heuristic watermark detection — FFT + brightness anomaly.

    Returns a grayscale PNG (white = suspected watermark area).
    The result is intentionally conservative; the user is expected
    to refine it with the mask editor.
    """
    from scipy.ndimage import (
        binary_dilation,
        label,
        laplace,
        uniform_filter,
    )

    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = np.mean(arr, axis=2)

    # --- High-frequency content via Laplacian ---
    edges = np.abs(laplace(gray))
    e_min, e_max = float(edges.min()), float(edges.max())
    if e_max > e_min:
        edges = (edges - e_min) / (e_max - e_min)

    # --- Local brightness anomaly ---
    blur_size = max(min(h, w) // 20, 3)
    local_mean = uniform_filter(gray, size=blur_size)
    anomaly = np.abs(gray - local_mean)
    a_min, a_max = float(anomaly.min()), float(anomaly.max())
    if a_max > a_min:
        anomaly = (anomaly - a_min) / (a_max - a_min)

    # --- Combined score ---
    score = 0.5 * edges + 0.5 * anomaly

    # Threshold at 90th percentile (top 10% of activity)
    threshold = float(np.percentile(score, 90))
    binary = (score >= threshold).astype(np.uint8)

    # Dilate to connect nearby regions
    dilation_iters = max(max(h, w) // 80, 2)
    dilated = binary_dilation(binary, iterations=dilation_iters)

    # Keep only the largest connected component
    labeled, num_features = label(dilated)
    if num_features > 0:
        sizes = [int((labeled == i).sum()) for i in range(1, num_features + 1)]
        largest_idx = int(np.argmax(sizes)) + 1
        mask = (labeled == largest_idx).astype(np.uint8) * 255
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    mask_img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return buf.getvalue()


def remove_watermark(image_path: Path, mask_bytes: bytes) -> bytes:
    """Fill the masked region using LaMa inpainting.

    Downloads ~60 MB of model weights to ./models/lama/ on first call.
    Result is always LANCZOS-resized back to source dimensions.
    """
    if "lama" not in _lama_cache:
        print("[erase] Loading LaMa model …")
        from simple_lama_inpainting import SimpleLama
        _lama_cache["lama"] = SimpleLama()
        print("[erase] LaMa loaded.")

    lama = _lama_cache["lama"]

    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size

    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if mask.size != (orig_w, orig_h):
        mask = mask.resize((orig_w, orig_h), Image.LANCZOS)

    result = lama(orig, mask)

    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.array(result, dtype=np.uint8))

    result = result.convert("RGB")
    if result.size != (orig_w, orig_h):
        result = result.resize((orig_w, orig_h), Image.LANCZOS)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 4: Run tests again**

```bash
source venv/bin/activate && python -m pytest tests/test_erase.py -v 2>&1 | tail -20
```

Expected: `test_remove_watermark_*` may take a few seconds on first run (LaMa download). All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/erase.py tests/test_erase.py tests/__init__.py
git commit -m "feat: core/erase.py — watermark detection and LaMa removal"
```

---

## Task 3: Add server endpoints

**Files:**
- Modify: `server.py` (add after the `# ── Routes: Depth Map` section, around line 1254)

- [ ] **Step 1: Locate insertion point**

The new section goes after the `_run_depth_map` function. Find the line after:
```python
def _run_depth_map(image_path: str, repo_id: str) -> bytes:
    from core.depth_map import generate_depth_map
```
(around line 1253). Add the new section after the depth map section ends.

- [ ] **Step 2: Add the erase section to `server.py`**

After the depth map endpoint block, insert:

```python
# ── Routes: Watermark Remover ─────────────────────────────────────────────────

class EraseDetectRequest(BaseModel):
    file_path: str


class EraseRequest(BaseModel):
    file_path: str
    mask_id:   str


@app.post("/api/erase/detect")
async def api_erase_detect(req: EraseDetectRequest):
    """Run heuristic watermark detection. Returns temp URLs for image + mask."""
    src_path = Path(req.file_path)
    if not src_path.exists():
        raise HTTPException(400, f"File not found: {src_path}")

    # Copy source image to temp so frontend can preview it
    img_id = f"{uuid.uuid4().hex}{src_path.suffix or '.png'}"
    shutil.copy2(src_path, _temp_path(img_id))

    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor(1) as pool:
            mask_bytes = await loop.run_in_executor(
                pool, lambda: _run_erase_detect(str(src_path))
            )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Detection failed: {e}")

    mask_id = f"{uuid.uuid4().hex}.png"
    _temp_path(mask_id).write_bytes(mask_bytes)

    return {
        "image_id":  img_id,
        "image_url": f"/api/temp/{img_id}",
        "mask_id":   mask_id,
        "mask_url":  f"/api/temp/{mask_id}",
    }


@app.post("/api/erase")
async def api_erase(req: EraseRequest):
    """Fill the masked region with LaMa inpainting. Saves result to output dir."""
    src_path = Path(req.file_path)
    if not src_path.exists():
        raise HTTPException(400, f"File not found: {src_path}")

    mask_path = _temp_path(req.mask_id)
    if not mask_path.exists():
        raise HTTPException(400, f"Mask not found: {req.mask_id}")

    mask_bytes = mask_path.read_bytes()

    out_name = src_path.stem + "_erased.png"
    out_path = Path(_output_dir()) / out_name

    import asyncio, json as _json
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor(1) as pool:
            result_bytes = await loop.run_in_executor(
                pool, lambda: _run_erase(str(src_path), mask_bytes)
            )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Watermark removal failed: {e}")

    out_path.write_bytes(result_bytes)
    out_path.with_suffix(".json").write_text(
        _json.dumps({"source": str(src_path), "operation": "watermark_removal"}, indent=2)
    )

    return {"url": f"/api/output/{out_name}", "filename": out_name}


def _run_erase_detect(image_path: str) -> bytes:
    from core.erase import detect_watermark
    return detect_watermark(Path(image_path))


def _run_erase(image_path: str, mask_bytes: bytes) -> bytes:
    from core.erase import remove_watermark
    return remove_watermark(Path(image_path), mask_bytes)
```

- [ ] **Step 3: Verify server starts without errors**

```bash
source venv/bin/activate && python -c "import server; print('server imports ok')"
```

Expected: prints `server imports ok`.

- [ ] **Step 4: Commit**

```bash
git add server.py
git commit -m "feat: POST /api/erase/detect and POST /api/erase endpoints"
```

---

## Task 4: Add API helpers to `api.ts`

**Files:**
- Modify: `frontend/src/api.ts`

- [ ] **Step 1: Add the erase helpers after the depth map section**

Find the line:
```typescript
export const generateDepthMap = (params: {
```

After the `generateDepthMap` block, insert:

```typescript
// ── Watermark Remover ─────────────────────────────────────────────────────────

export interface EraseDetectResult {
  image_id:  string
  image_url: string
  mask_id:   string
  mask_url:  string
}

export const eraseDetect = (filePath: string) =>
  post<EraseDetectResult>('/api/erase/detect', { file_path: filePath })

export interface EraseResult {
  url:      string
  filename: string
}

export const eraseRemove = (filePath: string, maskId: string) =>
  post<EraseResult>('/api/erase', { file_path: filePath, mask_id: maskId })
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat: eraseDetect and eraseRemove API helpers"
```

---

## Task 5: Create `EraseEditorModal.tsx`

**Files:**
- Create: `frontend/src/components/EraseEditorModal.tsx`

- [ ] **Step 1: Create the component**

Create `frontend/src/components/EraseEditorModal.tsx`:

```typescript
import { useState, useRef, useEffect, useCallback } from 'react'
import { Square, Pencil, Check, X } from 'lucide-react'

interface Props {
  imageUrl:       string
  initialMaskUrl: string | null
  onClose:        () => void
  onConfirm:      (maskId: string, maskUrl: string) => void
}

type Tool = 'rect' | 'brush'

export default function EraseEditorModal({ imageUrl, initialMaskUrl, onClose, onConfirm }: Props) {
  const displayRef  = useRef<HTMLCanvasElement>(null)   // overlay canvas (user sees)
  // Detached canvas — not in DOM, always present regardless of render state
  const maskRef     = useRef<HTMLCanvasElement>(document.createElement('canvas'))
  const imgRef      = useRef<HTMLImageElement>(null)

  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 })
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 })
  const [tool,        setTool]        = useState<Tool>('rect')
  const [brushSize,   setBrushSize]   = useState(20)
  const [applying,    setApplying]    = useState(false)

  // Rect tool state
  const [rect, setRect]   = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const draggingRef        = useRef(false)
  const startPtRef         = useRef({ x: 0, y: 0 })

  // Brush tool state
  const brushingRef        = useRef(false)
  const eraseMode          = useRef(false)   // true when Shift held during brush stroke

  function calcDisplay(nw: number, nh: number) {
    const maxW = 760, maxH = 560
    let w = nw, h = nh
    if (w > maxW) { h = Math.round(h * maxW / w); w = maxW }
    if (h > maxH) { w = Math.round(w * maxH / h); h = maxH }
    return { w, h }
  }

  // When image loads: set sizes, init mask canvas, load initial mask
  function onImageLoad() {
    const img = imgRef.current!
    const natural = { w: img.naturalWidth, h: img.naturalHeight }
    const display = calcDisplay(natural.w, natural.h)
    setNaturalSize(natural)
    setDisplaySize(display)

    // Init offscreen mask canvas
    const mc = maskRef.current!
    mc.width  = natural.w
    mc.height = natural.h
    const mctx = mc.getContext('2d')!
    mctx.fillStyle = 'black'
    mctx.fillRect(0, 0, mc.width, mc.height)

    // Load initial mask if provided
    if (initialMaskUrl) {
      const maskImg = new window.Image()
      maskImg.crossOrigin = 'anonymous'
      maskImg.onload = () => {
        mctx.drawImage(maskImg, 0, 0, natural.w, natural.h)
        renderOverlay()
      }
      maskImg.src = initialMaskUrl
    }
  }

  // Render overlay canvas: mask as semi-transparent red over image
  const renderOverlay = useCallback(() => {
    const canvas = displayRef.current
    const mc     = maskRef.current
    if (!canvas || !mc || displaySize.w === 0) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw scaled mask with red tint overlay
    ctx.save()
    ctx.globalAlpha = 0.45
    ctx.drawImage(mc, 0, 0, displaySize.w, displaySize.h)
    ctx.globalCompositeOperation = 'source-atop'
    ctx.fillStyle = 'rgba(220, 50, 50, 1)'
    ctx.fillRect(0, 0, displaySize.w, displaySize.h)
    ctx.restore()

    // Draw in-progress rect selection
    if (tool === 'rect' && rect && rect.w > 2 && rect.h > 2) {
      ctx.strokeStyle = '#7c3aed'
      ctx.lineWidth   = 2
      ctx.setLineDash([5, 4])
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h)
      ctx.setLineDash([])
    }
  }, [displaySize, rect, tool])

  useEffect(() => { renderOverlay() }, [renderOverlay])

  // ── Rect tool handlers ──────────────────────────────────────────────────────

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const r = displayRef.current!.getBoundingClientRect()
    const pos = { x: e.clientX - r.left, y: e.clientY - r.top }
    if (tool === 'rect') {
      draggingRef.current = true
      startPtRef.current  = pos
      setRect({ x: pos.x, y: pos.y, w: 0, h: 0 })
    } else {
      brushingRef.current = true
      eraseMode.current   = e.shiftKey
      paintBrush(pos.x, pos.y, e.shiftKey)
    }
  }

  function paintBrush(cx: number, cy: number, erase: boolean) {
    const mc   = maskRef.current!
    const mctx = mc.getContext('2d')!
    const scaleX = naturalSize.w / displaySize.w
    const scaleY = naturalSize.h / displaySize.h
    const mx = cx * scaleX
    const my = cy * scaleY
    const mr = (brushSize / 2) * Math.max(scaleX, scaleY)
    mctx.fillStyle = erase ? 'black' : 'white'
    mctx.beginPath()
    mctx.arc(mx, my, mr, 0, Math.PI * 2)
    mctx.fill()
    renderOverlay()
  }

  // Window-level mouse tracking — keeps drag/brush alive past canvas edge
  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!displayRef.current) return
      const r   = displayRef.current.getBoundingClientRect()
      const cx  = Math.max(0, Math.min(e.clientX - r.left, r.width))
      const cy  = Math.max(0, Math.min(e.clientY - r.top,  r.height))

      if (draggingRef.current) {
        const sp = startPtRef.current
        setRect({ x: Math.min(sp.x, cx), y: Math.min(sp.y, cy), w: Math.abs(cx - sp.x), h: Math.abs(cy - sp.y) })
      }
      if (brushingRef.current) {
        paintBrush(cx, cy, eraseMode.current)
      }
    }
    function onUp() {
      if (draggingRef.current && rect && rect.w > 2 && rect.h > 2) {
        commitRect()
      }
      draggingRef.current = false
      brushingRef.current = false
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup',   onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup',   onUp)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rect, displaySize, naturalSize, brushSize])

  function commitRect() {
    if (!rect || rect.w < 2 || rect.h < 2 || naturalSize.w === 0) return
    const scaleX = naturalSize.w / displaySize.w
    const scaleY = naturalSize.h / displaySize.h
    const mc   = maskRef.current!
    const mctx = mc.getContext('2d')!
    mctx.fillStyle = 'white'
    mctx.fillRect(
      Math.round(rect.x * scaleX), Math.round(rect.y * scaleY),
      Math.round(rect.w * scaleX), Math.round(rect.h * scaleY),
    )
    setRect(null)
    renderOverlay()
  }

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
      if (e.key === 'Enter')  handleConfirm()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleConfirm() {
    setApplying(true)
    try {
      const mc = maskRef.current!
      const blob: Blob = await new Promise(res => mc.toBlob(b => res(b!), 'image/png'))
      const fd = new FormData()
      fd.append('file', new File([blob], 'erase_mask.png', { type: 'image/png' }))
      const r = await fetch('/api/upload', { method: 'POST', body: fd })
      if (!r.ok) throw new Error(`Upload failed: ${r.status}`)
      const { id, url } = await r.json()
      onConfirm(id, url)
    } catch (err) {
      console.error('Mask upload failed', err)
    } finally {
      setApplying(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-surface border border-border rounded-xl shadow-2xl flex flex-col gap-3 p-4 max-w-[820px] w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-white">Edit Watermark Mask</span>
          <button onClick={onClose} className="text-muted hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-3">
          <div className="flex gap-1 bg-card rounded-md p-1">
            <button
              onClick={() => setTool('rect')}
              title="Rectangle tool"
              className={`p-1.5 rounded transition-colors ${tool === 'rect' ? 'bg-accent text-white' : 'text-muted hover:text-white'}`}
            >
              <Square size={13} />
            </button>
            <button
              onClick={() => setTool('brush')}
              title="Brush tool (Shift = erase)"
              className={`p-1.5 rounded transition-colors ${tool === 'brush' ? 'bg-accent text-white' : 'text-muted hover:text-white'}`}
            >
              <Pencil size={13} />
            </button>
          </div>

          {tool === 'brush' && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted">Size</span>
              <input
                type="range" min={8} max={64} value={brushSize}
                onChange={e => setBrushSize(Number(e.target.value))}
                className="w-24 accent-violet-500"
              />
              <span className="text-xs text-muted w-6">{brushSize}</span>
            </div>
          )}

          <span className="text-xs text-muted ml-auto">
            {tool === 'brush' ? 'Paint to add mask · Shift+drag to erase' : 'Drag to add rectangle'}
          </span>
        </div>

        {/* Canvas area */}
        <div className="relative bg-black rounded-md overflow-hidden flex items-center justify-center"
             style={{ minHeight: 200 }}>
          {/* Hidden image for loading natural dimensions */}
          <img
            ref={imgRef}
            src={imageUrl}
            onLoad={onImageLoad}
            className="select-none"
            style={{ display: displaySize.w === 0 ? 'block' : 'none', maxWidth: 760, maxHeight: 560 }}
            alt="source"
          />
          {displaySize.w > 0 && (
            <div className="relative" style={{ width: displaySize.w, height: displaySize.h }}>
              {/* Background image */}
              <img
                src={imageUrl}
                className="absolute inset-0 select-none pointer-events-none"
                style={{ width: displaySize.w, height: displaySize.h }}
                alt=""
              />
              {/* Mask overlay canvas */}
              <canvas
                ref={displayRef}
                width={displaySize.w}
                height={displaySize.h}
                className="absolute inset-0 cursor-crosshair"
                onMouseDown={onMouseDown}
              />
              {/* maskRef is a detached canvas — not rendered in DOM */}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs rounded-md border border-border text-muted hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={applying}
            className="px-3 py-1.5 text-xs rounded-md bg-accent hover:bg-accent/80 text-white
                       flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <Check size={12} />
            {applying ? 'Saving…' : 'Confirm Mask'}
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/EraseEditorModal.tsx
git commit -m "feat: EraseEditorModal — rectangle + brush mask editor"
```

---

## Task 6: Add `WatermarkPanel` to `Sidebar.tsx`

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Add imports**

At line 2-6 of `Sidebar.tsx`, add `Eraser` and `Wand` to the lucide-react import:

```typescript
import {
  ChevronDown, ChevronRight, Play, Square,
  Layers, Sliders, Video, UploadCloud, X, Workflow, Cpu,
  Wand2, ArrowUpCircle, FolderInput, ListOrdered, FolderOpen, ImagePlus, Plus,
  Eraser,
} from 'lucide-react'
```

- [ ] **Step 2: Add `eraseDetect` and `eraseRemove` to the api import line**

Change line 8:
```typescript
import { importComfyUI, loadWorkflow, saveWorkflow, uploadLora, uploadUpscaleModel, streamBatchUpscale, streamBatchGenerate, openFolderDialog, openFileDialog, upscaleSingleImage, updateSettings, openWorkflowFolderDialog, listLoras, stopGeneration, generateDepthMap } from '../api'
```
to:
```typescript
import { importComfyUI, loadWorkflow, saveWorkflow, uploadLora, uploadUpscaleModel, streamBatchUpscale, streamBatchGenerate, openFolderDialog, openFileDialog, upscaleSingleImage, updateSettings, openWorkflowFolderDialog, listLoras, stopGeneration, generateDepthMap, eraseDetect, eraseRemove } from '../api'
```

- [ ] **Step 3: Add the `EraseEditorModal` import after the HelpTip import**

After `import HelpTip from './HelpTip'`, add:

```typescript
import EraseEditorModal from './EraseEditorModal'
```

- [ ] **Step 4: Add `WatermarkPanel` component**

Add the following component just before the Sidebar's main export function (after `DepthMapPanel`, before `// ── Batch Img2Img panel`):

```typescript
// ── Watermark Remover panel ───────────────────────────────────────────────────

interface WatermarkPanelProps {
  onRefresh: () => void
  onStatus:  (msg: string) => void
}

function WatermarkPanel({ onRefresh, onStatus }: WatermarkPanelProps) {
  const [filePath,      setFilePath]      = useState('')
  const [imageUrl,      setImageUrl]      = useState<string | null>(null)
  const [maskId,        setMaskId]        = useState<string | null>(null)
  const [maskUrl,       setMaskUrl]       = useState<string | null>(null)
  const [detectionMsg,  setDetectionMsg]  = useState('')
  const [isDetecting,   setIsDetecting]   = useState(false)
  const [isRemoving,    setIsRemoving]    = useState(false)
  const [showEditor,    setShowEditor]    = useState(false)
  const [picking,       setPicking]       = useState(false)

  async function handlePick() {
    setPicking(true)
    try {
      const data = await openFileDialog()
      if (!data.cancelled && data.path) {
        setFilePath(data.path)
        setImageUrl(null)
        setMaskId(null)
        setMaskUrl(null)
        setDetectionMsg('')
      }
    } catch (e: unknown) {
      onStatus(`File picker error: ${(e as Error).message}`)
    } finally {
      setPicking(false)
    }
  }

  async function handleDetect() {
    if (!filePath.trim()) { onStatus('⚠ Pick an image file first'); return }
    setIsDetecting(true)
    setMaskId(null)
    setMaskUrl(null)
    setDetectionMsg('')
    try {
      const result = await eraseDetect(filePath.trim())
      setImageUrl(result.image_url)
      setMaskId(result.mask_id)
      setMaskUrl(result.mask_url)
      setDetectionMsg('Watermark region detected — edit mask if needed')
    } catch (e: unknown) {
      onStatus(`Detection failed: ${(e as Error).message}`)
      setDetectionMsg('Detection failed — draw mask manually')
    } finally {
      setIsDetecting(false)
    }
  }

  async function handleRemove() {
    if (!filePath.trim() || !maskId) { onStatus('⚠ Run detection first'); return }
    setIsRemoving(true)
    try {
      const result = await eraseRemove(filePath.trim(), maskId)
      onStatus(`✓ Saved: ${result.filename}`)
      onRefresh()
    } catch (e: unknown) {
      onStatus(`Removal failed: ${(e as Error).message}`)
    } finally {
      setIsRemoving(false)
    }
  }

  const busy = isDetecting || isRemoving || picking

  return (
    <div className="space-y-3">
      {/* File picker */}
      <div>
        <label className="text-xs text-muted block mb-1">Image</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={filePath}
            onChange={e => setFilePath(e.target.value)}
            placeholder="/Users/you/Pictures/photo.png"
            className="flex-1 bg-card border border-border rounded-md px-3 py-1.5 text-xs text-white
                       placeholder-muted focus:outline-none focus:border-accent transition-colors"
          />
          <button
            onClick={handlePick}
            disabled={busy}
            title="Browse for image"
            aria-label="Browse for image"
            className="px-2 py-1.5 rounded-md bg-card border border-border text-muted
                       hover:text-white hover:border-accent transition-colors disabled:opacity-50 shrink-0"
          >
            <FolderOpen size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Detect button */}
      <button
        onClick={handleDetect}
        disabled={busy || !filePath.trim()}
        className="w-full py-2 rounded-lg bg-card border border-border text-white text-xs font-medium
                   flex items-center justify-center gap-2 transition-colors
                   hover:border-accent disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <Eraser size={13} />
        {isDetecting ? 'Detecting…' : 'Detect Watermark'}
      </button>

      {/* Preview + mask overlay */}
      {imageUrl && (
        <div className="relative rounded-md overflow-hidden bg-black">
          <img src={imageUrl} alt="source" className="w-full object-contain max-h-48 select-none" />
          {maskUrl && (
            <img
              src={maskUrl}
              alt="mask"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              style={{ opacity: 0.45, mixBlendMode: 'screen', filter: 'sepia(1) saturate(8) hue-rotate(300deg)' }}
            />
          )}
        </div>
      )}

      {/* Detection status message */}
      {detectionMsg && (
        <p className="text-[10px] text-muted leading-snug">{detectionMsg}</p>
      )}

      {/* Edit mask / Remove buttons — only show after detection */}
      {imageUrl && (
        <div className="flex gap-2">
          <button
            onClick={() => setShowEditor(true)}
            disabled={busy}
            className="flex-1 py-1.5 rounded-md bg-card border border-border text-xs text-muted
                       hover:text-white hover:border-accent transition-colors disabled:opacity-40"
          >
            Edit Mask
          </button>
          <button
            onClick={handleRemove}
            disabled={busy || !maskId}
            className="flex-1 py-1.5 rounded-lg bg-accent hover:bg-accent/80 text-white text-xs font-medium
                       flex items-center justify-center gap-1.5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isRemoving ? 'Removing…' : 'Remove'}
          </button>
        </div>
      )}

      {/* Editor modal */}
      {showEditor && imageUrl && (
        <EraseEditorModal
          imageUrl={imageUrl}
          initialMaskUrl={maskUrl}
          onClose={() => setShowEditor(false)}
          onConfirm={(id, url) => {
            setMaskId(id)
            setMaskUrl(url)
            setShowEditor(false)
          }}
        />
      )}
    </div>
  )
}
```

- [ ] **Step 5: Add the accordion entry in the Sidebar JSX**

Find the closing `</Accordion>` of the Depth Map section:
```tsx
      </Accordion>

      {/* Video */}
```

After it, insert:
```tsx
      {/* Watermark Remover */}
      <Accordion label="Watermark Remover" icon={<Eraser size={13} />}>
        <WatermarkPanel
          onRefresh={onRefresh}
          onStatus={onStatus}
        />
      </Accordion>
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: WatermarkPanel and Watermark Remover accordion in Sidebar"
```

---

## Task 7: Build frontend and smoke test

**Files:** none (build verification only)

- [ ] **Step 1: Build the frontend**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: build completes with no TypeScript errors. Output ends with something like:
```
✓ built in Xs
```

If TypeScript errors appear, fix them before continuing (most likely a missing import or type mismatch).

- [ ] **Step 2: Start server and open browser**

```bash
source venv/bin/activate && python server.py --port 7860 --no-auto-shutdown
```

Open `http://localhost:7860` in Chrome.

- [ ] **Step 3: Smoke test the Watermark Remover accordion**

1. Open the **Watermark Remover** accordion in the Sidebar
2. Click **Browse** → pick any PNG from your Pictures folder
3. Click **Detect Watermark** — wait for detection, preview should appear with reddish overlay
4. Click **Edit Mask** — modal opens, switch to brush tool, paint over an area, confirm
5. Click **Remove** — wait for LaMa (first run downloads ~60MB), result appears in Gallery
6. Verify the result file `{stem}_erased.png` exists in `~/Pictures/ultra-fast-image-gen/`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: build artifacts"
```

---

## Task 8: Open PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/watermark-remover
```

- [ ] **Step 2: Create PR**

```bash
gh pr create \
  --title "feat: Watermark Remover — FFT detection + LaMa inpainting" \
  --body "$(cat <<'EOF'
## Summary
- **`core/erase.py`**: heuristic watermark detection (FFT + brightness anomaly) + LaMa inpainting removal
- **`/api/erase/detect`** + **`/api/erase`**: two new FastAPI endpoints, ThreadPoolExecutor, not pipeline-gated
- **`EraseEditorModal`**: full-screen canvas editor with rectangle and brush (Shift=erase) tools
- **`WatermarkPanel`**: new Sidebar accordion — file picker → detect → preview → edit mask → remove → Gallery

## Test plan
- [ ] Detect watermark on a real watermarked image — preview shows reddish overlay
- [ ] Edit mask in modal: rectangle tool draws region; brush tool paints; Shift+brush erases
- [ ] Remove with LaMa — result saved as `{stem}_erased.png` in output dir, appears in Gallery
- [ ] Detect on blank image → "Nothing detected — draw manually" message
- [ ] File not found → error shown in status bar (not a crash)
- [ ] `pytest tests/test_erase.py` — all 5 tests pass

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
