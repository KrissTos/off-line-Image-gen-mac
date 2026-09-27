# Changelog

History and "why we did X". Operational rules live in `CLAUDE.md`; reference in `docs/`.

## 2026-09-27
- SAM click-to-mask editor replaces the rectangle-drag mask modal: `core/segment.py` (`facebook/sam-vit-large`, MPS, embedding LRU cache) + `POST /api/segment/prepare`/`POST /api/segment`; full-screen `MaskEditor.tsx` with SAM point/box (add+subtract), brush, polygon, invert, grow/shrink, undo/redo. Verified on the real Bagno 1 wall/floor job: rebuilding the hand-made mask in the UI (SAM objects + plaster polygon + invert + grow 3) reached IoU 0.911 against the original script's mask, and a 3-ref FLUX 9B masked generation logged `img2img (3 ref) (masked-composite)`.
- Mask editor final-review fixes: a SAM result was applied to the mask captured before the request, so an undo or brush stroke made meanwhile was reverted (now read from `maskRef` after the await); one failed click no longer locks SAM; brush strokes repaint only a dirty rect of a persistent overlay and coverage is memoised (was a 48 MB ImageData rebuild + 12M-pixel loop per pointermove on 12 MP); spinner while SAM loads/runs; tool keys ignore Cmd/Ctrl. SAM click latency 2443 → 120 ms cold / 7 ms warm by picking the best-IoU candidate before `post_process_masks`.
- `Launch.command` opens the UI in Google Chrome (falls back to the default browser).
- FLUX 4B outpaint LoRA path (auto green pad + `fal/flux-2-klein-4B-outpaint-lora`); blur pad on 9B often just copied the blur.
- Three LoRA bugs: BFL-native FLUX.2 LoRAs never loaded (diffusers converter hardcodes FLUX.2-dev 8/48 blocks); load failures were swallowed and the result info still listed the LoRA; removing all LoRAs didn't unload them.
- Composite seam: soft mask blur reached into the pad → pad fill bled in as a line. Now max(hard, blurred).
- Output auto-size to slot #1 aspect for all model families (`canvasSize.ts`, `store.ts autoSizeParams`). The earlier LTX-only version was a `SizePanel` effect: the accordion unmounts its children, so it never ran while collapsed and re-fired on reopen, undoing presets. Restored slots carry `keepSize`.
- Slot #1 + mask are fitted into the canvas instead of stretched (`fit_ref_to_canvas`). Before: a mask + different aspect stretched the image; auto-outpaint padded with black and FLUX ran plain img2img with the mask ignored (no composite), so nothing was extended. FLUX refs #2+ now go at native size (were squashed to output dims).
- `--no-auto-shutdown` never worked: `uvicorn.run("server:app")` re-imports `server`, so the `__main__` global didn't reach the app. Now an env var.
- Crop mode used to replace ALL FLUX refs with the crop → masked edits ignored material refs and invented textures (`crop_flux_refs`).
- LTX stretched video: output dims came from the size preset, never the ref aspect (768×1365 photo + Square 512 = stretched).
- Native `.githooks/pre-commit` pytest gate added.
- `POST /api/workflows/save` returned 500 on every save: `timestamp` was an undefined name (only `api_save_log` defined one), so no workflow had saved in the current `yy-mm-dd_name` format. Covered by `tests/test_workflow_save.py` (save + load round trip with mask).
- Slot masks are made in the editor only: the empty mask box opens MaskEditor ("draw mask"), a filled one reopens it; the per-slot file upload and the hover pencil are gone. External PNG masks go through the editor's Import… so they get reviewed. Slot #2+ mask boxes explain they only feed Iterate Masks and show "unused" in other modes (Generate sends only slot #1's mask).
- UI: gallery strip/grid toggle (grid scrolls vertically, choice in `localStorage`); "+ ref img" drop zone after the last slot; mask editor 96 px column with zoom −/+/Fit/100% and +/− keys, brush size slider, `HelpTip` hover popups on every tool.

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
