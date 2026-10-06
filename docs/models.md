# Models & generation pipeline reference

Per-model behaviour, sizing and masking internals. Rules that bite every session are in `CLAUDE.md`;
this file holds the detail. Update in the same session as the code.

## Models
Cached in the shared global HF cache `~/.cache/huggingface/hub` (`get_local_models_dir()` → `get_hf_global_cache_dir()`; no `HF_HUB_CACHE` override). `./models/` holds only `da3mono-large/` + `CACHEDIR.TAG` + `.locks/`. `sync_from_hf_cache()` has a `src==dst` guard. No revision pin → auto-update on next load.

| Model | VRAM | guidance | steps | Notes |
|-------|------|----------|-------|-------|
| FLUX.2-klein-4B (4bit SDNQ) | <8 GB @ 512px | 0 | 20 | fast |
| FLUX.2-klein-9B (4bit SDNQ) | ~12 GB @ 512px | 0 | 20 | higher quality |
| FLUX.2-klein-4B (Int8) | ~16 GB | 0 | 20 | |
| Z-Image Turbo (Quantized) | ~8 GB | 0 | 4 | fastest; no ref/img2img path (refs ignored) |
| Z-Image Turbo (Full) | ~24 GB | 0 | 4 | LoRA, img2img, inpaint |
| LTX-Video 0.9.8-13B-distilled | ~26 GB bf16 | 1.0 | fixed timesteps | + `a-r-r-o-w/LTX-0.9.8-Latent-Upsampler` (lazy) |

All distilled → guidance slider hidden; defaults from `guidanceForModel()` / `stepsForModel()` (`App.tsx`). LTX ignores the steps slider (`LTX_BASE_TIMESTEPS` / `LTX_DENOISE_TIMESTEPS`).

## Output size & slot #1 fit
- **UI auto-size** (`frontend/src/canvasSize.ts` `canvasForRef()`, applied in `store.ts` `autoSizeParams()`): output w/h = slot #1 aspect at the family budget — FLUX/Z-Image ~1 MP /16, LTX 768×512 /32. E.g. 4000×3000→1184×880, 1080×1920→768×1360; LTX 768×1365→480×832, 1920×1080→832×480, 1024×1024→640×640.
- Fires only on: slot #1 first dims (`SET_SLOT_DIMS` with `!prev.w`), slot #1 removed/replaced, model **family** change (4B↔9B keeps size). A preset picked afterwards is kept (= outpaint). Restored slots (`keepSize`) keep the saved size.
- **Backend fit** (`app.fit_ref_to_canvas()`, `tests/test_fit_canvas.py`): slot #1 scaled to fit (up or down) at `outpaint_align`, never stretched. Uncovered area = blurred cover-scaled copy of the ref, unioned into the mask; padded → mask mode forced to Inpainting (full-frame + composite). `_SNAP_REL=0.03`: ≤3% aspect drift fills instead of outpainting a sliver.
- User mask gets the same transform. Mask aspect ≠ slot #1 → stretched + `⚠` in result info, except when slot #1 is already canvas-size (Iterate pass ≥2) → mask fitted on its own aspect.
- FLUX slots #2+ go at native size (`prepare_flux_refs()`); `Flux2KleinPipeline` keeps ref aspect itself (≤1 MP, /16).
- **Outpaint LoRA (FLUX 4B only, automatic)**: when slot #1 needs padding on `flux2-klein-sdnq`/`-int8` (`wants_outpaint_lora()`), the pad is pure green (`pad_fill="green"`), `fal/flux-2-klein-4B-outpaint-lora` is appended to the request's LoRAs at 1.1 (`get_outpaint_lora_path()`, HF cache, one-time 76 MB download; failure → blur pad), and the prompt becomes `outpaint_prompt()` = trigger "Fill the green spaces according to the image. " + user prompt. Seamless fills; the LoRA was trained on klein-4B *base* but works on our distilled 4B.
- 9B / Z-Image: blurred-image pad. FLUX 9B often copies the blur (weak outpaint) — prefer 4B for outpaint.
- `apply_mask_composite` soft mask = max(hard mask, blurred): masked/pad pixels stay 100% generated, the ramp falls on the original side (else the pad fill bleeds in as a line).
- LTX has no fixed resolution: any /32 dims + 8k+1 frames; mismatched aspect stretches the ref into the canvas.

## Masking
- **SAM click-to-mask editor** (`core/segment.py`, `frontend/src/components/MaskEditor.tsx`): `facebook/sam-vit-large` (transformers, MPS), lazy-loaded and kept resident. Image embedding cached per image id (LRU of 4) so repeat clicks on the same image only re-run the (fast) prompt decoder. `_segment_image_loader()` in `server.py` applies `ImageOps.exif_transpose()` before SAM sees the image, so a phone JPEG's orientation tag is normalized and click coordinates match what the browser already shows the user. MPS quirks: cast float64→float32 before `.to("mps")`; keep `original_sizes`/`reshaped_input_sizes` on CPU for `post_process_masks`. One point + no box → `multimask_output=True`, keep the best `iou_scores` candidate. Full-screen editor combines SAM point/box with brush, polygon and invert/grow to build masks tiles/floor SAM can't segment directly (see `docs/handoffs/2026-09-27-sam-click-mask-and-generate-local-provider.md` for the Bagno 1 recipe this replaced).
- **Crop & Composite**: generate only the mask bbox (`get_mask_bbox`, +32 px, /64), paste back with a blurred mask. Only slot #1 is swapped for its crop; slots #2+ pass through (`crop_flux_refs()`, `tests/test_mask_crop.py`).
- **Inpainting Pipeline**: Z-Image Full uses `ZImageInpaintPipeline`; FLUX uses `Flux2KleinInpaintPipeline` (diffusers ≥ 0.38; `FluxInpaintPipeline` ≠ Flux2Klein), built from `pipe.components` and cached in `inpaint_pipe` (rebuilt when the klein pipe changes). Slot #1 = `image`, its mask = `mask_image`, slots #2+ = `image_reference`, `strength` = `img_strength`. It drifts ~3/255 outside the mask (VAE round trip), so the pixel composite (`masked-composite`) still runs. Not used for auto-outpaint (the 4B outpaint LoRA is trained for img2img) and falls back to img2img on error. Measured 9B @768²: ~88 s (1.4x the crop path) but no box seam. Composite skipped for txt2img results (model never saw the ref).
- **Iterate Masks** (`handleIterateGenerate`, `App.tsx`): one `/api/generate` per masked slot; pass N inputs = `[prev_out, slotN.image]`, mask = slotN mask, strength = slotN strength; `uploadFromUrl()` re-uploads between passes.

## LoRA
- Multi-LoRA: `lora_files: LoraSlot[]` (≤5), named adapters in `load_loras()`; `if not lora_files` (not `is None`) for legacy fallback. `LoraSlot` carries `name?` / `model_type?` for sidecars.
- FLUX.2-klein LoRA needs diffusers git main. BFL-native LoRAs (`double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`) go through our `convert_bfl_flux2_lora()` (`core/lora_flux2.py`): diffusers' converter hardcodes FLUX.2-dev 8/48 blocks, requires MLP keys and rejects embedder/modulation keys. Unmapped key → error, never a half-applied LoRA.
- klein LoRA size (4B/9B) comes from header tensor shapes (`lora_variant()`); `/api/lora/list` returns it, the UI greys mismatches, `assert_lora_matches_model()` guards loading. klein-4B = 5 double / 20 single, hidden 3072; 9B = 8 / 24, 4096.
- `sync_loras()` loads requested LoRAs (failure → `ensure_loras_loaded()` raises → UI error) and unloads leftovers when a request has none.
- LoRA accordion `key={lora_files.length > 0 ? 'lora-has-files' : 'lora-empty'}` + `defaultOpen` → remounts to auto-open when params are loaded (`useState(defaultOpen)` reads only at mount).

## LTX-Video 0.9.8-13B-distilled (`app.py`)
- Pipeline: `LTXConditionPipeline` (not `LTXPipeline` / `LTXImageToVideoPipeline`). `render_ltx_video()` takes pipelines as args (mockable, `tests/test_ltx.py`), returns PIL frames.
- Multiscale (default): gen @2/3 res `output_type=latent` → `LTXLatentUpsamplePipeline` (2×, `tone_map_compression_ratio=0.6`) → 4-step denoise (`denoise_strength=0.999`) → resize. `fast_preview` = single distilled pass, no upsampler. Upsampler lazy (`video_upsampler`, reset on device switch). `fast_preview` flows via `req.model_dump()`.
- i2v = `LTXVideoCondition(image=ref, frame_index=0)` in `conditions=[…]`; txt2video = `conditions=None`.
- Multi-ref keyframes: `ref_image` = `None | PIL | list[PIL]`; list → one condition per ref, `frame_index` from `_ltx_keyframe_indices(m, n_frames)` (first→0, last→final, even spread, /8 stride, strictly increasing, capped at `last//8+1` — surplus dropped). Call site preprocesses `input_images[:6]` → `preprocessed_video_refs`. Keyframes morph over time (A→B→C), not a spatial blend. All keyframes strength 1.0 (per-slot strength not sent).
- "Blend" (ltx.io blog) = multi-image → fused still (our FLUX.2 multi-ref) → i2v keyframe; not a video-model feature.
- Distilled params: `guidance_scale=1.0`, `guidance_rescale=0.7`, `decode_timestep=0.05`, `image_cond_noise_scale=0.0`.
- Frames = 8k+1 only (9, 17, …, 121); backend re-snaps `((max(9,n)-1)//8)*8+1`. Slider `step={8}` is correct; Video accordion shows a `≈ Ns` readout.
- MP4 export `export_frames_to_video()` imports `imageio` lazily (libx264) → needs `imageio` + `imageio-ffmpeg` deps.
- Download: repo is 93 GB but ~45 GB is a duplicate transformer + text_encoder under `vae/`; `app.DOWNLOAD_IGNORE_PATTERNS` skips `vae/transformer/*`, `vae/text_encoder/*`, `media/*` → ~48 GB. Other repos pass `ignore_patterns=None`.
- Never delete the `vae/transformer` / `vae/text_encoder` blobs of an existing copy: they are HF-deduped with the real weights.
- FP8 variants rejected (not suitable for Apple Silicon).

## Depth map (DA3 / DA2)
- Default repo `istiakiat/DA3MONO-LARGE` (mirror, not official `depth-anything/…`) in `core/depth_map.py` and `server.py`. Weights load first from flat `./models/da3mono-large/` (`_load_da3`); delete it → downloads the mirror into the global cache.
- DA3 = invert, DA2 = no invert; output LANCZOS-resized to source; GS/3D export deps mocked via `sys.modules`.
