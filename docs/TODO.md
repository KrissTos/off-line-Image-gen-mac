# TODO / known issues

## Next up (decided 2026-10-06)
1. Bug pass from the lists below (scope to confirm with the user; the nine testable backend items first, MaskEditor races later).
2. Eyes Direction: plan ready in `docs/superpowers/plans/2026-10-06-eyes-direction.md`.
3. Then CivitAI search in Model Sources; LTX-2.5 spike or dropping local video are parked.

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
- MaskEditor residual races (rare, from the final review): dirty rect is overwritten not unioned across coalesced pointermoves (overlay gap until next rebuild); `maskRef` synced in a passive `useEffect` (move to render body / `useLayoutEffect`); brush pointermove doesn't clear `lastSam`, so Shift-refine after a mid-stroke SAM result can drop the stroke tail unundoably; a pointermove from a stale closure can overwrite a just-landed SAM result.
- Run folders, deferred minors (2026-09-27 final review): `DELETE /api/output/<run>/workflow.json` or `refs/*` deletes non-output files (hand-built request only; require `parts[1] == "outputs"`); LTX video `info` still says `Saved: <old path>` after the rename; migration leaves `X.png`+`X.mp4` same-stem pairs, upscale-of-upscale and prompt stems ending `_NNNxNNN` untouched (fail safe); overwriting an unmigrated v1 workflow leaves root `slot_N_image.png` strays.
- Depth map `filename` (legacy output-dir name) uses `Path(name).name`, so it can't reach files inside run folders; UI sends `file_path` only.
- Evaluate on diffusers 0.40: `black-forest-labs/FLUX.2-small-decoder` (1.4x faster decode, 0.38 s vs 0.54 s at 1024px), native diffusers SDNQ backend; switch `torch_dtype` to `dtype`.

- `pipeline.py` and `server._run_events` swallow generation tracebacks (only `str(exc)` reaches the client); a failure like the FLUX→LTX one needed a temporary `traceback.print_exc()` to diagnose.
- Decided against `FLUX.2-klein-9b-kv` (2026-10-02): non-commercial, 9B only, needs a community SDNQ build (unverified, ~8 GB), and only speeds up multi-ref edits.

- Eyes Direction control (decided 2026-10-06, not started; hygiene pass first): sidebar accordion with a square pad and a draggable red dot (x, y in 0..1 of the canvas). On generate (klein-9B only, base image set) auto-render the author's 1024x1024 control image (white canvas, black 580 px square frame with 6 px border centered, red dot radius 85 at x,y; dot may sit outside the frame) as ref slot #2 at native size, prepend `change the eyes to match the reference dot direction`, and auto-attach `eric-venti-seeds/Eyes_Direction_Lora_Flux2Klein9B` (file `Eyes_direction_Lora_Flux2Klein_9B_v1.safetensors`, 265 MB, MIT, one-time HF download like the outpaint LoRA) at 1.0 (0.5-1 realistic, 1.25-1.5 anime/non-realistic). Spec source: github.com/eric-venti-seeds/Eyes_Direction_Lora_Control `nodes.py`. Mind CLAUDE.md rule 7.1/7.2: base stays slot #1, refs #2+ native size. Needs: pure TS/py renderer + tests, `GenerateRequest` field kept backward compatible.

- Gallery preview: still unexercised in a browser against real data: the "Upscaled from" link (no upscale-with-source output on disk; helper is unit-tested).

- CivitAI search in Model Sources (discussed 2026-10-06, not started): anonymous `GET civitai.com/api/v1/models?types=LORA&query=...&nsfw=false` works and `baseModel` (`Flux.2 Klein 9B`, `Flux.2 Klein 4B`, `LTXV`) maps onto the family folders; file downloads need no key. Open: link-only vs download into `lora_uploads/`; the Z-Image base-model label is unknown. Worth doing now that rows say what a LoRA does.
- Parked ideas: evaluate LTX-2.5 locally (diffusers 0.40 has `LTX2Pipeline`; needs a one-clip MPS time/memory test, ~174 GB repo) or drop local video entirely (frees ~45 GB, not 100; touches `app.py`, `server.py`, the Video accordion, tests and the `/generate` skill's local route).
- Model Sources: a few auto descriptions stay weak where the card has no good prose (e.g. `Flux2-Klein-9B-Migration` is empty); edit by hand, hand-written text is never overwritten.
