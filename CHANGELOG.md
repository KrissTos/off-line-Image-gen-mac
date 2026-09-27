# Changelog

History and "why we did X". Operational rules live in `CLAUDE.md`; reference in `docs/`.

## 2026-09-27
- Output auto-size to slot #1 aspect for all model families (`canvasSize.ts`, `store.ts autoSizeParams`). The earlier LTX-only version was a `SizePanel` effect: the accordion unmounts its children, so it never ran while collapsed and re-fired on reopen, undoing presets. Restored slots carry `keepSize`.
- Slot #1 + mask are fitted into the canvas instead of stretched (`fit_ref_to_canvas`). Before: a mask + different aspect stretched the image; auto-outpaint padded with black and FLUX ran plain img2img with the mask ignored (no composite), so nothing was extended. FLUX refs #2+ now go at native size (were squashed to output dims).
- `--no-auto-shutdown` never worked: `uvicorn.run("server:app")` re-imports `server`, so the `__main__` global didn't reach the app. Now an env var.
- Crop mode used to replace ALL FLUX refs with the crop → masked edits ignored material refs and invented textures (`crop_flux_refs`).
- LTX stretched video: output dims came from the size preset, never the ref aspect (768×1365 photo + Square 512 = stretched).
- Native `.githooks/pre-commit` pytest gate added.

## 2026-08-31
- Models moved to the shared global HF cache; `HF_HUB_CACHE` override removed from `app.py` / `generate.py`.

## 2026-06
- LTX-Video migrated to 0.9.8-13B-distilled + fast-preview; multi-ref keyframes (LTX branch used to discard slot #2+).
- Skip ~45 GB duplicate LTX weights on download; LTX MP4 export deps (`imageio` missing surfaced only at generation time).
- Model Sources show only loadable models: discover used to scrape arbitrary HF base repos (SDXL, Qwen, FLUX.1-dev, text encoders) the loader can't run. `vram_gb` (current alloc, ~0 idle) was wrongly used for "fits" → added `total_vram_gb`.
- Project marked standalone (diverged from upstream fork).

## 2026-04
- Watermark remover (FFT detect + LaMa), HF discover in Model Sources, FP8 repos blocklisted (unsuitable on Apple Silicon).
- FLUX LoRA was never loaded during generation (pre-existing bug) — fixed with multi-LoRA stacking.

## Earlier
- Renamed `ultra-fast-image-gen` → `off-line-Image-gen-mac`; UI brand "Local AI Image Gen".
- Gradio UI removed; `app.py` is backend logic only, UI is the React frontend.
- `server.py` was reworked 16× across sessions → workflow defaults in `CLAUDE.md` (plan first, read before edit, test first).
