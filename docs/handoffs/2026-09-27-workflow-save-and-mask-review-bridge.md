# Handoff: workflow save fix and mask-review bridge for `/generate`

Status: tasks 1-4 superseded by run folders (see docs/architecture.md); task 1 fixed in da004a4.

Date: 2026-09-27 · From: ai-generate session (`/generate` local provider) · Start with `cd ~/Projects/off-line-Image-gen-mac && claude`

## Why

`/generate` now drives this app over HTTP (`~/.claude/skills/generate/lib/local_run.py`, recipe
`models/local-offline.md`). Next on the `/generate` side is `lib/local_mask.py`: Claude builds a mask
with `/api/segment` (SAM boxes/points + polygons, then combine/invert/grow), and **Cris reviews and
fixes it in this app's mask editor** before any masked generation. Cris wants a human check between
the auto-built mask and the generation.

The bridge is a **workflow folder**:

1. `local_mask.py` writes `workflows/<yy-mm-dd>_<job>/` directly on disk: `workflow.json`,
   `slot_1_image.png` and `slot_1_mask.png` (plus more slots for material refs), with prompt, model
   and size filled in.
2. Cris loads it in the app. Slot 1 shows the image with the mask on it. He fixes it in the mask
   editor (SAM click, brush, invert, grow).
3. Back to `/generate`, one of two ways:
   - **A. Generate in the app.** `/generate` imports the output plus its companion folder
     (`<output stem>/mask.png`, `ref_slot_N.png`, `params.json`) into `ai-generate/generations/`.
   - **B. Save the workflow.** `/generate` reads the fixed `slot_1_mask.png` and runs `local_run.py`.
     **Blocked by Task 1.**

This session covers the app side only. `local_mask.py` is built afterwards in an ai-generate session,
against the contract you confirm here.

---

## Task 1: fix `/api/workflows/save` (bug, blocks path B)

`server.py:867` has `"timestamp": timestamp`. That name is defined only inside `api_save_log()`
(`server.py:1700`), so every workflow save raises `NameError` and returns 500. Evidence: the newest
folder in `workflows/` is `20260301_142246_…`, and none uses the current `yy-mm-dd_name` format.
Probably broken since `f9646a4` ("module-level datetime import"); check with `git log -L`.

- Write a failing test first (`tests/test_workflow_save.py`). Use FastAPI `TestClient`, point
  `app.WORKFLOWS_DIR` and `TEMP_DIR` at `tmp_path`, and put a temp image + mask in TEMP_DIR. Assert
  a 200 response, that `workflow.json` has `ref_slots[0].mask == "slot_1_mask.png"`, and that both
  PNGs were copied.
- Fix: `datetime.now().strftime(...)`. Match the format the rest of the file uses (`datetime` is
  imported at module level on line 26).
- Test the round trip too: save, then `GET /api/workflows/{name}` returns `ref_slots[0].maskUrl`.
- Restart the server afterwards. It has no auto-reload, and there is usually one running on :7860.

## Task 2: confirm the external-writer contract

`local_mask.py` will write `workflow.json` itself, without calling `/api/workflows/save`. Confirm
and document the minimum it needs:

- `GET /api/workflows/{name}` runs `app.load_workflow(name)` (an 18-tuple) before it reads
  `ref_slots`. Find out which keys `load_workflow` requires and which defaults it tolerates. Easiest
  check: a hand-written minimal `workflow.json` with the fields below loads cleanly in the UI.
  ```json
  {"name": "...", "prompt": "...", "width": 1328, "height": 784, "steps": 20, "seed": -1,
   "guidance": 0.0, "device": "mps", "model_choice": "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)",
   "model_source": "Local", "img_strength": 1.0, "repeat_count": 1, "lora_files": [],
   "num_frames": 25, "fps": 24, "mask_mode": "Crop & Composite (Fast)", "outpaint_align": "center",
   "ref_slots": [{"image": "slot_1_image.png", "mask": "slot_1_mask.png", "strength": 1.0},
                 {"image": "slot_2_image.png", "mask": null, "strength": 1.0}]}
  ```
- Mask semantics: L-mode PNG at slot 1's full resolution, white = regenerate. Confirm the editor
  and generation read it the same way. Check that a mask with soft (grown/feathered) edges survives
  editor load → apply without being thresholded.
- `GET /api/workflows` shows only the newest 15, sorted by name descending (`list_saved_workflows`).
  A `26-09-…` name sorts above the old `2026…` ones, so it will show. Note it anyway.
- Write the contract into `docs/architecture.md` as an external-writer section, and add a pointer
  line in CLAUDE.md next to the existing "External API client" line.

## Task 3: decide how a save lands (path B)

`api_save_workflow` writes to `<today>_<name>`. Saving under the same name on the same day
overwrites the folder, while a different day or name creates a new one. The rewrite also drops any
keys it doesn't know. For path B, `/generate` has to find the folder Cris saved. Pick one and
document it:

- Keep it as is. Cris saves under the same job name on the same day, and `/generate` takes the
  newest folder matching `*_<job>`.
- Or add "Save back to loaded workflow" (overwrite the folder that was loaded, keep its name). That
  makes a cleaner contract. Ask Cris first, since it's a UI change.

## Task 4: check the path A artifacts

Generate once with a masked slot 1. Confirm that the output's companion folder `<stem>/` holds
`mask.png` (the mask the editor applied, full resolution), `ref_slot_N.png` and `params.json`, and
that the sidecar `.json` next to the output has `has_mask: true`. `/generate` will import from these.
Note that the companion folder is written only when there are refs or a mask (`server.py:~667`).

## Related open items (do if cheap, else leave in `docs/TODO.md`)

- `_load_pil` (`server.py:343`) skips `exif_transpose`, while SAM and the editor work on the
  transposed image. A rotated phone JPEG can get a misaligned mask. Masks from `local_mask.py` will
  hit this too. Fixing it here is better than `/generate` writing upright copies.
- `/api/models` `available` is always empty: it scans repo `models/`, but weights live in the global
  HF cache (`server.py:409`). Point it at `get_local_models_dir()`.

## Verify before closing

- `uv run pytest tests/` passes, including the new test.
- In the UI: load a hand-written workflow (the Task 2 JSON plus real PNGs, e.g. a copy of
  `~/Projects/ai-generate/generations/refs/sala100_tavola_single.jpg` and a rough mask). The mask
  shows on slot 1. Edit it, save it, and confirm `slot_1_mask.png` on disk changed.
- Commit per task. Update CHANGELOG and `docs/TODO.md`.
