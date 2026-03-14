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
| `src/types.ts` | Remove `IpAdapterSlot`, `IpAdapterStatus` interfaces; also remove `ip_adapter_image_ids: string[]`, `ip_adapter_scales: number[]`, `ip_adapter_enabled: boolean` from the `GenerateParams` interface |
| `src/store.ts` | Remove `IpAdapterSlot, IpAdapterStatus` from the `./types` import (line 2); remove `ipAdapterSlots`, `ipAdapterEnabled`, `ipAdapterStatus` state fields and their entries in `DEFAULT_PARAMS`; remove 6 IPA action type definitions from the `Action` union; remove the 6 corresponding reducer `case` blocks (`ADD_IPA_SLOT`, `REMOVE_IPA_SLOT`, `UPDATE_IPA_SCALE`, `CLEAR_IPA_SLOTS`, `TOGGLE_IPA`, `SET_IPA_STATUS`); remove the `ipaSlotsToParams()` helper function |
| `src/api.ts` | Remove `fetchIpAdapterStatus`, `deleteIpAdapter`, `streamIpAdapterDownload`; remove `ip_adapter` field from the `ModelExtras` type |
| `src/App.tsx` | Remove `fetchIpAdapterStatus` from the `./api` import statement; remove `fetchIpAdapterStatus` bootstrap call; remove all IPA state threading (`ipAdapterSlots`, `ipAdapterEnabled`, `ipAdapterStatus` dispatch/props); specifically remove the three IPA prop pass-throughs to `<Sidebar>` at the `<Sidebar>` call site |
| `src/components/Sidebar.tsx` | Remove `IpAdapterSlot, IpAdapterStatus` from the `../types` import; remove `fetchIpAdapterStatus` from the `../api` import; remove `IpAdapterPanel` import; remove `ipAdapterSlots`, `ipAdapterEnabled`, `ipAdapterStatus` from `SidebarProps`; remove the `useEffect` that calls `fetchIpAdapterStatus`; remove IP-Adapter accordion section entirely |
| `src/components/IpAdapterPanel.tsx` | Delete file |
| `src/components/SettingsDrawer.tsx` | Remove `deleteIpAdapter` from the `../api` import; remove `deletingIpa` state variable; remove `handleDeleteIpa()` function; remove the entire IP-Adapter JSX block from "Other Models" (the `extras.ip_adapter && (...)` conditional block); simplify the section render guard from `extras && (extras.upscale_models.length > 0 \|\| extras.ip_adapter)` to `extras && extras.upscale_models.length > 0` |

### Backend — removals

| File | Change |
|------|--------|
| `server.py` | Remove `from core.ip_adapter_flux import ...` import block; remove `_ipa_loaded` global flag declaration; remove `ip_adapter_image_ids: list[str]`, `ip_adapter_scales: list[float]`, `ip_adapter_enabled: bool` from `GenerateRequest`; remove `global _ipa_loaded` declaration and `_ipa_loaded = False` assignment inside the `/api/models/load` handler; remove IPA image-loading and param-passing logic in the generate endpoint; remove `/api/ip-adapter/status`, `/api/ip-adapter/download`, `DELETE /api/ip-adapter` route handlers; in the `GET /api/models/extras` handler, remove the `IP_ADAPTER_FILE` import, the `ipa` dict assembly block, and the `ip_adapter` key from the returned dict — keep the `upscale_models` list only |
| `app.py` | Remove `ip_adapter_images`, `ip_adapter_scales`, `ip_adapter_enabled` parameters from `generate_image()`; remove all `_ipa_active` blocks and `ip_adapter_image` kwargs from every pipeline call |

### What stays

| Item | Reason |
|------|--------|
| `core/ip_adapter_flux.py` | File stays on disk — no harm, easy to revisit if FLUX.2-klein IP-Adapter support lands in diffusers |
| `./models/ip_adapter/` weights on disk | User can delete manually; not worth auto-deleting |
| Ref slot system (img2img) | Unchanged — provides image-guided generation today |
| Upscale model management in Settings | Keep — unrelated feature |
| `GET /api/models/extras` endpoint | Keep — returns `upscale_models` only after IPA removal |
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

- [ ] `npm run build` succeeds with zero TypeScript errors
- [ ] App boots without error (no missing imports, no `NameError` on `_ipa_loaded`)
- [ ] Generate with FLUX model works (txt2img, img2img, inpainting)
- [ ] Generate with Z-Image and LTX-Video works
- [ ] Loading a model via Settings does not raise `NameError`
- [ ] Settings drawer opens; "Other Models" shows upscale models only, no IP-Adapter row
- [ ] No broken references to `ipAdapterSlots` / `IpAdapterPanel` in browser console
