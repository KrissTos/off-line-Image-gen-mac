# Remove IP-Adapter — Design Spec

**Date:** 2026-03-14
**Status:** Approved

## Background

The IP-Adapter feature (InstantX/FLUX.1-dev-IP-Adapter) was built into the UI and backend but is architecturally incompatible with `Flux2KleinPipeline`, the pipeline used for all FLUX models in this app. `Flux2KleinPipeline` has no `load_ip_adapter`, `set_ip_adapter_scale`, or `ip_adapter_image` support — every IP-Adapter call at runtime would raise `AttributeError`. The InstantX weights target FLUX.1-dev, not FLUX.2-klein.

**Decision:** Remove IP-Adapter cleanly. The existing ref slot system (img2img with reference images) already provides image-guided generation and is fully functional.

---

## Scope

### Frontend — removals

| File | Change |
|------|--------|
| `src/types.ts` | Remove `IpAdapterSlot`, `IpAdapterStatus` interfaces |
| `src/store.ts` | Remove `ipAdapterSlots`, `ipAdapterEnabled`, `ipAdapterStatus` state fields; remove 6 IPA actions: `ADD_IPA_SLOT`, `REMOVE_IPA_SLOT`, `UPDATE_IPA_SCALE`, `CLEAR_IPA_SLOTS`, `TOGGLE_IPA`, `SET_IPA_STATUS` |
| `src/api.ts` | Remove `fetchIpAdapterStatus`, `deleteIpAdapter`, `streamIpAdapterDownload`; remove `ip_adapter` field from `ModelExtras` type and `fetchModelExtras` response handling |
| `src/App.tsx` | Remove all IPA state threading: `ipAdapterSlots`, `ipAdapterEnabled`, `ipAdapterStatus` dispatch/props; remove `fetchIpAdapterStatus` bootstrap call |
| `src/components/Sidebar.tsx` | Remove IP-Adapter accordion section entirely |
| `src/components/IpAdapterPanel.tsx` | Delete file |
| `src/components/SettingsDrawer.tsx` | Remove IP-Adapter row from "Other Models" section; keep upscale model listing |

### Backend — removals

| File | Change |
|------|--------|
| `server.py` | Remove `ip_adapter_image_ids: list[str]`, `ip_adapter_scales: list[float]`, `ip_adapter_enabled: bool` from `GenerateRequest`; remove `/api/ip-adapter/status`, `/api/ip-adapter/download`, `DELETE /api/ip-adapter` routes; remove `from core.ip_adapter_flux import ...` import block; remove `_ipa_loaded` global flag; remove IPA logic in the generate endpoint (image loading + params passing) |
| `app.py` | Remove `ip_adapter_images`, `ip_adapter_scales`, `ip_adapter_enabled` parameters from `generate_image()`; remove all `_ipa_active` blocks and `ip_adapter_image` kwargs from every pipeline call |
| `api/models/extras` endpoint | Remove `ip_adapter` field from response; only return `upscale_models` |

### What stays

| Item | Reason |
|------|--------|
| `core/ip_adapter_flux.py` | File stays on disk — no harm, easy to revisit if FLUX.2-klein IP-Adapter support lands in diffusers |
| `./models/ip_adapter/` weights on disk | User can delete manually via OS; not worth auto-deleting |
| Ref slot system (img2img) | Unchanged — provides image-guided generation today |
| Upscale model management in Settings | Keep — unrelated feature |
| `GET /api/models/extras` endpoint | Keep — returns `upscale_models` only |
| `DELETE /api/upscale/{filename}` endpoint | Keep — unrelated |

---

## Data flow after removal

```
GenerateRequest (server.py)
  prompt, height, width, steps, guidance, seed
  ref_image_ids[], ref_image_masks[], ref_strengths[]   ← unchanged
  upscale_*, lora_*, video_*                             ← unchanged
  # ip_adapter_* fields: gone
```

```
generate_image() (app.py)
  # ip_adapter_images / ip_adapter_scales / ip_adapter_enabled: removed
  # _ipa_active block: removed from all pipe() calls
```

---

## Non-goals

- Do **not** add any replacement or stub for IP-Adapter
- Do **not** modify ref slot UX or tooltips to mention IP-Adapter
- Do **not** delete `core/ip_adapter_flux.py` or downloaded weights

---

## Testing checklist

- [ ] App boots without error (no missing imports)
- [ ] Generate with FLUX model works (txt2img, img2img, inpainting)
- [ ] Generate with Z-Image and LTX-Video works
- [ ] Settings drawer opens; "Other Models" shows upscale models only
- [ ] No TypeScript compiler errors (`npm run build` clean)
- [ ] No broken references to `ipAdapterSlots` / `IpAdapterPanel` in console
