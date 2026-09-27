# TODO / known issues

- `generate.py` CLI supports Z-Image Turbo only (no FLUX, no LTX).
- Single-pass generate sends only slot #1's mask (`slotsToParams()`); per-slot masks need Iterate Masks.
- Per-slot strength never reaches the backend in single-pass or LTX keyframes (all 1.0 / global `img_strength`).
- No CLIP loader: text encoders are bundled per model and load at model-load time.
- FLUX 9B / Z-Image outpaint is weak (blur pad copied); only 4B has the outpaint LoRA. No maintained 9B equivalent (only merged checkpoint `24aittl/klein-9b-outpaint`, NC license, unquantized).
- Z-Image LoRA loader reports "Loaded N" even when some of several LoRAs failed (only prints the failure).
- Ref badges read `base`, `ref 1`, `ref 2`…, but FLUX.2 prompts refer to images by order (`image 1` = base, `image 2` = badge `ref 1`). Relabel to `img 1..N` offered, not decided (`RefImagesRow.tsx:48`).
- Guidance slider is hidden: unhide when a non-distilled full-precision model is added.
- `frontend/src/App.tsx:18` fails eslint `react-refresh/only-export-components` (pre-existing).
- `.tmp_uploads/` grows without cleanup.
- Text-to-mask (SAM 3 / Grounded-SAM) = phase 2.
- `/api/models` `available` is always empty: `get_locally_available_models()` scans repo `models/`, but weights live in the global HF cache since 2026-08-31 (`server.py:409`). Point it at `get_local_models_dir()`.
- `_load_pil` (`server.py:343`) doesn't apply `exif_transpose`: a phone JPEG with EXIF orientation gets its SAM mask built upright (frontend transposes for display) but generation loads the untransposed pixels, so the mask can land misaligned with the actual image content.
- MaskEditor residual races (rare, from the final review): dirty rect is overwritten not unioned across coalesced pointermoves (overlay gap until next rebuild); `maskRef` synced in a passive `useEffect` (move to render body / `useLayoutEffect`); brush pointermove doesn't clear `lastSam`, so Shift-refine after a mid-stroke SAM result can drop the stroke tail unundoably; a pointermove from a stale closure can overwrite a just-landed SAM result.
- Run folders, deferred minors (2026-09-27 final review): `DELETE /api/output/<run>/workflow.json` or `refs/*` deletes non-output files (hand-built request only; require `parts[1] == "outputs"`); LTX video `info` still says `Saved: <old path>` after the rename; migration leaves `X.png`+`X.mp4` same-stem pairs, upscale-of-upscale and prompt stems ending `_NNNxNNN` untouched (fail safe); overwriting an unmigrated v1 workflow leaves root `slot_N_image.png` strays.
- Depth map `filename` (legacy output-dir name) uses `Path(name).name`, so it can't reach files inside run folders; UI sends `file_path` only.
