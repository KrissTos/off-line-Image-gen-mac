# TODO / known issues

- `generate.py` CLI supports Z-Image Turbo only (no FLUX, no LTX).
- Single-pass generate sends only slot #1's mask (`slotsToParams()`); per-slot masks need Iterate Masks.
- Per-slot strength never reaches the backend in single-pass or LTX keyframes (all 1.0 / global `img_strength`).
- No CLIP loader: text encoders are bundled per model and load at model-load time.
- FLUX 9B / Z-Image outpaint is weak (blur pad copied); only 4B has the outpaint LoRA. No maintained 9B equivalent (only merged checkpoint `24aittl/klein-9b-outpaint`, NC license, unquantized).
- Z-Image LoRA loader reports "Loaded N" even when some of several LoRAs failed (only prints the failure).
- Guidance slider is hidden: unhide when a non-distilled full-precision model is added.
- `frontend/src/App.tsx:18` fails eslint `react-refresh/only-export-components` (pre-existing).
- `.tmp_uploads/` grows without cleanup.
- Text-to-mask (SAM 3 / Grounded-SAM) = phase 2.
- `/generate` local provider (Task B, `~/.claude/skills/generate/`): see `docs/handoffs/2026-09-27-sam-click-mask-and-generate-local-provider.md`.
