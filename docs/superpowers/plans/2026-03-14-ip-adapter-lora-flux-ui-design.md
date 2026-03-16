# Design: IP-Adapter + FLUX LoRA + Gallery Drag + UI Help

**Date:** 2026-03-14
**Status:** Approved — proceeding to implementation

---

## Summary

Four features added in one cohesive milestone:

1. **IP-Adapter** — InstantX FLUX.1-dev-IP-Adapter, up to 3 reference images
2. **LoRA for FLUX.2-klein** — extend existing LoRA system with key-remapping for CivitAI compatibility
3. **Gallery → Ref slot drag** — drag any gallery output into a ref image slot
4. **UI help system** — inline `ⓘ` tooltips on every non-obvious control

---

## 1. IP-Adapter

### Weights

| File | Repo | Size | License |
|------|------|------|---------|
| `ip_adapter.bin` | `InstantX/FLUX.1-dev-IP-Adapter` | ~5.3 GB | Non-commercial |
| SigLIP encoder | `google/siglip-so400m-patch14-384` | ~1.8 GB | Apache 2.0 |

Downloaded on demand to `./models/ip_adapter/instantx/`. Not bundled.

### Backend — `core/ip_adapter_flux.py` (new module)

Responsibilities:
- `download_ip_adapter_weights(progress_cb)` — streams from HF Hub, reports progress
- `load_ip_adapter(pipe)` — calls `pipe.load_ip_adapter("InstantX/FLUX.1-dev-IP-Adapter", weight_name="ip_adapter.bin")`
- `set_ip_adapter_scale(pipe, scales: list[float])` — calls `pipe.set_ip_adapter_scale(scales)`
- `is_downloaded() -> bool` — checks local file exists
- `unload_ip_adapter(pipe)` — calls `pipe.unload_ip_adapter()`

IP-Adapter state is **model-scoped**: loaded/unloaded when the FLUX model loads/unloads. Cached as a flag on `PipelineManager`.

### Integration with `app.generate_image()`

New params passed through the chain:
```python
ip_adapter_images: list[PIL.Image] | None = None   # 1–3 reference images
ip_adapter_scales: list[float] | None = None        # per-image scale (0.0–1.0)
ip_adapter_enabled: bool = False
```

In `generate_image()`, before pipeline call:
```python
if ip_adapter_enabled and ip_adapter_images:
    pipe.set_ip_adapter_scale(ip_adapter_scales)
    kwargs["ip_adapter_image"] = ip_adapter_images
```

### API changes — `server.py`

New endpoints:
- `GET /api/ip-adapter/status` → `{downloaded, loaded, downloading, progress}`
- `POST /api/ip-adapter/download` → SSE stream of download progress events
- `DELETE /api/ip-adapter` → unload + optionally delete weights

`GenerateRequest` gets new optional fields:
```python
ip_adapter_image_ids: list[str] = []
ip_adapter_scales:    list[float] = []
ip_adapter_enabled:   bool = False
```

### UI — new "IP-Adapter" accordion in Sidebar

**State: not downloaded**
```
IP-Adapter                              [▼]
  Style/identity transfer from reference images
  Weights not installed (~7 GB total)
  [ Download IP-Adapter weights ]
  ⓘ IP-Adapter lets you guide generation using a reference photo.
    Useful for style, faces, color mood, or object appearance.
```

**State: ready / active**
```
IP-Adapter                         [●on] [▼]
  Reference images (up to 3)
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ drop img │ │ drop img │ │   + add  │
  └──────────┘ └──────────┘ └──────────┘
  Scale  ●──────────  0.60   ●──────────  0.60
  ⓘ Higher scale = stronger influence. 0.4–0.7 recommended.
    Multiple images blend their visual features together.
```

Each ref slot: click to upload, drag-and-drop, clear (×) button, individual scale slider.

---

## 2. LoRA for FLUX.2-klein

### Problem

`Flux2Transformer2DModel` uses different internal key names than the `FluxTransformer2DModel` that most CivitAI/ai-toolkit LoRAs are trained against. `load_lora_weights()` silently finds no matching keys.

### Fix — key remapping

New function in `core/lora_flux2.py`:
```python
def remap_lora_keys(state_dict: dict) -> dict:
    """
    Remap ai-toolkit / CivitAI FLUX LoRA keys to Flux2Transformer2DModel keys.
    Primary mapping: 'transformer.' prefix handling + attention projection names.
    """
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        # Strip redundant transformer prefix if double-nested
        new_k = re.sub(r'^transformer\.transformer\.', 'transformer.', new_k)
        # CivitAI uses 'lora_unet_' prefix for FLUX LoRAs
        new_k = re.sub(r'^lora_unet_', 'transformer.', new_k)
        # Dot-separated to underscore block names
        # (extend based on research findings)
        remapped[new_k] = v
    return remapped
```

Load sequence:
```python
state_dict = load_file(lora_path)
state_dict = remap_lora_keys(state_dict)
pipe.load_lora_weights(state_dict, prefix=None)
```

If remapping still yields 0 matching keys → surface a specific error in the UI: "LoRA not compatible with FLUX.2-klein. Try a FLUX-native LoRA."

### UI changes

LoRA accordion: currently shows only for Z-Image Full. Extended to show for all FLUX models.

```
LoRA                                    [▼]
  [ Upload LoRA (.safetensors) ]
  Current: none
  Strength  ●──────────  0.80
  ⓘ LoRAs trained for standard FLUX.1 may not be compatible with FLUX.2-klein.
    If loading fails, try a LoRA specifically trained for FLUX.2-klein or FLUX.2.
    Compatible LoRAs: linoyts/Flux2-Klein-Delight-LoRA (HuggingFace)
```

---

## 3. Gallery → Ref Slot Drag

### Behaviour

- Gallery thumbnails become **draggable** (`draggable="true"`, `onDragStart` sets `dataTransfer` with the output image URL)
- Ref image slots in `RefImagesRow` become **drop targets** (`onDragOver`, `onDrop`)
- IP-Adapter slots (in the new accordion) are also drop targets
- On drop: calls `uploadFromUrl(url)` to create a new temp upload, then dispatches `ADD_REF_SLOT` or sets IP-Adapter slot

### Visual feedback

- Gallery thumbnail gets a subtle drag cursor + opacity on drag
- Ref slot highlights with accent border on `dragover`
- After drop: slot immediately shows thumbnail

---

## 4. UI Help System

### Approach: inline `ⓘ` icon with tooltip

Every control that isn't self-explanatory gets a small `ⓘ` icon next to its label. Hover (desktop) or tap (touch) shows a tooltip with 1–2 sentence explanation.

`HelpTip` component:
```tsx
<HelpTip text="Guidance scale controls how strictly the model follows your prompt. Lower = more creative, higher = more literal." />
```

### Controls getting help text

| Control | Help text |
|---------|-----------|
| Guidance scale | How strictly the model follows your prompt. Lower = creative, higher = literal. Z-Image Turbo always uses 0. |
| Steps | Denoising iterations. More steps = more detail but slower. FLUX.2: 20, Z-Image Turbo: 4. |
| Img strength | How much the reference image is preserved. 0 = keep everything, 1 = ignore ref. |
| LoRA strength | How strongly the LoRA style is applied. |
| IP-Adapter scale | How much the reference image influences generation. 0.4–0.7 recommended. |
| Mask mode | Crop & Composite = fast, replaces only masked area. Inpainting Pipeline = quality, rerenders coherently. |
| Outpaint align | Which corner/edge to place the reference image when output size differs. |
| Seed | -1 = random every time. Set a fixed number to reproduce a result. |
| Repeat count | Generate N images in sequence with the same settings. |

---

## Architecture Changes Summary

| Layer | Changes |
|-------|---------|
| `core/ip_adapter_flux.py` | New module — download, load, unload IP-Adapter |
| `core/lora_flux2.py` | New module — key remapping for FLUX.2-klein LoRAs |
| `app.py` | New params: `ip_adapter_images`, `ip_adapter_scales`, `ip_adapter_enabled`; extend LoRA to FLUX models |
| `pipeline.py` | Pass new IP-Adapter params; expose `ip_adapter_loaded` state |
| `server.py` | 3 new endpoints; extend `GenerateRequest` |
| `frontend/src/components/Sidebar.tsx` | New IP-Adapter accordion; extend LoRA accordion |
| `frontend/src/components/HelpTip.tsx` | New reusable tooltip component |
| `frontend/src/components/Gallery.tsx` | Add `draggable` + `onDragStart` |
| `frontend/src/components/RefImagesRow.tsx` | Add `onDragOver` + `onDrop` |
| `frontend/src/api.ts` | New IP-Adapter API calls |
| `frontend/src/types.ts` | New `IpAdapterSlot` type, extend `GenerateParams` |
| `frontend/src/store.ts` | New IP-Adapter state + actions |

---

## Out of Scope (v2)

- Regional IP-Adapter (per-face spatial masking)
- InstantID for multi-face
- IP-Adapter for Z-Image Turbo (SD-based, different weights)
- LoRA training UI
