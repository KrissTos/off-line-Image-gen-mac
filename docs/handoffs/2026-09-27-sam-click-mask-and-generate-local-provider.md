# Handoff — SAM click-to-mask editor + `/generate` local provider

Date: 2026-09-27 · From: ai-generate session (Bagno 1 wall/floor swap) · Start with `cd ~/Projects/off-line-Image-gen-mac && claude`

Two tasks. Task A is in this repo. Task B lives in the `/generate` skill (`~/.claude/skills/generate/`, workspace `~/Projects/ai-generate`) and calls this repo's API. Do them in order A → B, or B alone. They are independent.

---

## Context: why

Job: replace bathroom wall tiles and floor in a photo (`~/Desktop/Bagno 1.jpg`) with two vinyl samples, FLUX.2-klein-9B multi-ref (room slot 1, wall sample slot 2, floor sample slot 3).

- Without a mask: ~80% good, but the model straightened the wide-angle-bowed right wall and spread the wall film onto the painted plaster above the tiles.
- With a mask: geometry and plaster held, but materials came out "very different". Root cause was this repo: Crop & Composite dropped ref slots 2+. Fixed and pushed (`b7aee0a`, `crop_flux_refs()`, `tests/test_mask_crop.py`).
- The mask itself was made by hand in a Claude session, outside this app. That is what Task A brings into the app.

### How the working mask was built (the recipe Task A must reproduce in UI)

Script: `~/Projects/ai-generate/work/bagno1-mask/build.py` (uses `~/Projects/ai-generate/tools/floor-swap/floor_swap/masks.py`: `sam_masks()`, `polygon_mask()`). Output `~/Desktop/Bagno 1_mask.png`.

1. **SAM ViT-L** (`facebook/sam-vit-large`, transformers, MPS), one **box** prompt per object to keep: toilet, cistern + pipe, radiator, dispensers, brush, door, window, shelf, valves, flush button, door handle. Clean edges in one pass.
2. **SAM fails on big plain surfaces**: a box around a whole tiled wall or the floor returned only a fragment (floor ~1/3, right wall one strip). So the surfaces were NOT segmented directly.
3. **Polygon** traced by hand for the plaster/tile boundary. No model sees that line.
4. `mask = ~(plaster polygon | SAM objects | door)` → **invert** is the key op. Door filled right of its per-row left edge (SAM left a sliver at the frame border).
5. Keep-region dilated 2 px (protect object edges), mask grown 3 px for blending, never onto plaster.
6. Needed 3 correction rounds (handle, flush button, lever tip). A UI must make corrections cheap.

MPS gotchas already solved in `masks.py`: processor returns float64 → cast to float32 before `.to("mps")`; keep `original_sizes` / `reshaped_input_sizes` on CPU for `post_process_masks`. `SamProcessor` needs torchvision.

---

## Task A — SAM click-to-mask in the mask editor (this repo)

**Goal:** the mask above in ~1 minute of clicking inside the app, then generate with Crop & Composite.

Current state:
- `MaskEditorModal` in `frontend/src/components/RefImagesRow.tsx` (line ~16): rectangle drag only, always shows slot #1.
- `EraseEditorModal.tsx` already has rect + brush (Shift = erase), an offscreen full-res `maskRef` + scaled display canvas, 45% red tint, and uploads via `POST /api/upload`. Reuse this pattern, don't reinvent it.
- Backend precedent for a local model endpoint: `/api/erase/detect` + `core/erase.py` (LaMa), `/api/depth-map` (DA3). Lazy-load, run in a thread executor.

Scope:
1. **Backend** `core/segment.py` + `POST /api/segment`: input temp image id + list of prompts (`{type: "box"|"point", coords, label: +1/-1}`), returns a mask PNG (temp id/url). Lazy-load SAM once, keep resident, free on model switch like other pipelines. Cache the image embedding per image id so each extra click is fast (SAM encoder is the slow part; decoder is ms).
2. **Mask editor tools**: SAM click (point) and SAM box, each result **added** or **subtracted** (modifier key) · brush / eraser · polygon (for boundaries like the plaster line) · **Invert** · **Grow/Shrink N px** · Clear · Undo.
3. Overlay preview (red tint) at full res, zoom for edges.
4. Output = same mask upload path the editor uses today → `mask_image_id`.
5. Tests: pure mask ops (combine/invert/grow) unit-tested; `segment` with SAM mocked (pattern: `tests/test_ltx.py` passes pipelines in).

Decisions still open (ask Cris):
- Model: `sam-vit-large` (~1.2 GB, known-good here) vs `sam-vit-base` (faster) vs SAM 2.x / SAM 3. Default suggestion: ViT-L, since it is proven on this exact job.
- **Text-to-mask** ("walls and floor") via SAM 3 or Grounded-SAM = phase 2. Expect the same weak spots (large plain surfaces partial, human boundaries missing): use it as a starting mask, then click-correct.

Verify on the real case: rebuild the Bagno 1 mask in the UI, compare with `~/Desktop/Bagno 1_mask.png` (IoU), then run a masked generation with the 3 refs and check the log says `img2img (3 ref)` + `(masked-crop)`.

Note: since `b7aee0a` another session added a full-frame composite for "Inpainting Pipeline" mode on FLUX (`app.py` ~1640, `(masked-composite)`). Earlier notes saying "Inpainting on FLUX = no composite" are outdated. Check which mode suits a full-frame wall+floor mask (the bbox is nearly the whole image, so crop mode gains no speed).

---

## Task B — `/generate` local provider via this app's HTTP API (the TODO saved earlier)

Also recorded in `~/Cris_Vault/02-Projects/Project-ai-generate.md` § TODO.

**Decision:** call the FastAPI server directly. Do NOT drive the React UI with agent-browser (slower, burns tokens, breaks on UI changes).

Endpoints (`server.py`):
- `POST /api/upload` → ref/mask temp file id
- `POST /api/generate` → SSE stream: `progress` / `image` / `video` / `done` / `error`. Params: `prompt, model_choice, width, height, steps, guidance, seed, img_strength, input_image_ids, mask_image_id, mask_mode, lora_files, upscale_enabled, num_frames, fps` (read `GenerateRequest` in `server.py` first; it may have changed)
- `/api/status`, `/api/ping`, `/api/models`, `/api/models/load`
- extras: `/api/erase`, `/api/depth-map`, `/api/upscale/single`, `/api/batch/generate`, and `/api/segment` once Task A lands

Gotchas:
- `/api/generate` returns **423** while busy → check `/api/status` first, wait/queue.
- Server runs `venv/bin/python3 server.py --port 7860`, no auto-reload.

Plan:
1. Small `uv` Python client in `~/.claude/skills/generate/lib/` (e.g. `local_run.py`): upload → generate → read SSE → fetch output from `/api/output/...`.
2. If the server is down, start it (or tell Cris) and poll `/api/ping`.
3. Copy the output into `~/Projects/ai-generate/generations/` with the skill's own sidecar `.json`, refresh the manifest so the gallery shows it.
4. Route on "offline" / "local"; offer it as the free draft tier before paid models. $0, so no cost gate, but keep arg parsing strict (global rule: an unknown flag must never fall through to a default costly path; here no cost, but same hygiene).
5. Recipe file `models/local-offline.md` like the other providers; update `SKILL.md` routing.

Read first: `~/.claude/skills/generate/SKILL.md`, an existing runner (`lib/fal_run.py`) for structure, `GenerateRequest` in `server.py`.

---

## Loose ends from the source session

- `~/Projects/ai-generate/work/bagno1-mask/` (mask script, SAM cache, previews): trash once the final Bagno 1 image is in `ai-generate/generations/` with its sidecar.
- Prompt that worked best (with mask): describe materials only, "one continuous seamless surface: no tiles, no grout lines, no joints, no black border stripe", name the right wall's lens curvature, crop each sample out of its photo before using it as a ref.
