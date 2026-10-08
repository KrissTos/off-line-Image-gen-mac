# off-line-Image-gen-mac

Offline AI image/video generation for Apple Silicon (MPS): FLUX.2-klein, Z-Image Turbo, LTX-Video.
FastAPI backend + React frontend. UI brand "Local AI Image Gen".

## Pointers
- Backend/frontend files, components, API endpoints, SSE events → docs/architecture.md
- Models, VRAM, sizing/fit, masking, LoRA, LTX, depth-map internals → docs/models.md
- Open issues and ideas → docs/TODO.md
- History, past bugs, "why we did X" → CHANGELOG.md
- External API client: `/generate` skill `~/.claude/skills/generate/lib/local_run.py` (`/api/upload`, `/api/generate` SSE incl. event `path`/`info`, `/api/status`, `/api/ping`) → keep `GenerateRequest` fields and event shape backward compatible.

## 1. Maintaining this file
1. Keep it ≤ 200 lines, operational only; follow `~/.claude/docs/claudemd-hygiene.md`.
2. Route content: long reference → `docs/architecture.md` or `docs/models.md`; open items → `docs/TODO.md`; done/dated → `CHANGELOG.md`.
3. Update the matching doc in the same session as the code change.
4. Suggest `/ukn` after a non-obvious bug fix or a new architectural piece.

## 2. Workflow
1. `quick:` prefix → implement directly, no brainstorm/plan.
2. 1–2 files with clear requirements → implement directly.
3. 3+ files or unclear design → superpowers flow (brainstorm → spec → plan → subagents).
4. Plan first for any edit to `server.py` / `app.py` or touching 2+ files.
<!-- server.py was reworked 16× across sessions; large files punish blind edits -->
5. Read the target region before editing `server.py` / `app.py`; never edit from memory.
6. Write a failing test in `tests/` before new behavior, then make it pass.
7. Implement → verify → commit one step at a time.
8. Corrected twice on the same issue → write a progress note, `/clear`, restart (avoid `/compact`).

## 3. Folders
1. Put per-job scratch in `work/<job>/` (gitignored); trash it when the job is done.
2. Put project docs in `docs/` (max 3 topic files + `TODO.md`); never in `~/Downloads/Claude AI/`.
   `docs/superpowers/` (specs/plans) is gitignored: local only, don't `git add -f`.
3. Never commit runtime data: `models/`, `venv/`, `huggingface/`, `lora_uploads/`, `upscale_models/`, `workflows/`, `logs/`, `.tmp_uploads/`, `model_sources.json`.
4. `app_settings.json` is tracked but holds local runtime settings; leave its diffs uncommitted unless asked.

## 4. Run
```bash
./Launch.command               # production: builds frontend/dist, serves :7860, opens Chrome
./Launch.command --dev         # FastAPI :7861 + Vite HMR :5173
venv/bin/python server.py --port 7860 --no-auto-shutdown
cd frontend && npm run build   # after any frontend change
venv/bin/python -m pytest -q   # full suite
cd frontend && npm test        # pure-logic tests: mask editor + slots (node --test)
venv/bin/python -m core.run_store migrate <dir> [--apply]   # legacy flat outputs / v1 workflows → run folders (dry run default)
```
1. Server exits 60 s after the last browser ping; use `--no-auto-shutdown` for headless/API testing.
2. Restart the server after backend changes (no auto-reload).
3. `generate.py` CLI is Z-Image Turbo only.
4. One server only, on :7860 (no second server on :7861; that was dropped 2026-10-07 because the two processes share `model_sources.json`, `lora_uploads/` and the CivitAI registry). The user may be running it (`Launch.command`): say before stopping it (it may be mid-generation), kill only its PID (`lsof -nP -iTCP:7860 -sTCP:LISTEN -t`), never `pkill -f server.py`, run the check with `venv/bin/python server.py --port 7860 --no-auto-shutdown` via `run_in_background`, then restart it as in rule 5. The `/generate` runner's `--start` uses :7860.
5. `Launch.command` never closes its Terminal window (the old 'close front window' hit whichever window was frontmost); it exits to the shell. A server you stopped can be restarted from CC with `venv/bin/python server.py --port 7860` via `run_in_background`, and say so.

## 5. Environment
1. Use `uv`, never pip; sync with `UV_PROJECT_ENVIRONMENT=venv uv sync`.
<!-- plain `uv sync` targets .venv/, the wrong env; Launch.command sets the variable -->
2. Use `venv/bin/python` for everything (tests, server, scripts).
3. `diffusers` is pinned to git tag `v0.40.0` (pulls `transformers` 5 + `huggingface-hub` 1); `sdnq` comes from git main.
4. xformers is not needed on Apple Silicon (MPS has SDPA); don't try to install it.
5. HF token lives in `huggingface/token` (gitignored, Read type); gated models need terms accepted on their page.
6. Models live in the global HF cache `~/.cache/huggingface/hub`, not `./models/`.
7. Enable the terminal commit gate once per clone: `git config core.hooksPath .githooks`.
8. Commits are test-gated (`.claude/hooks/pre-commit-test-gate.sh` + `.githooks/pre-commit`); keep `tests/` green.
9. DA3 (`depth_anything_3`) is the `depth` dependency group in `pyproject.toml` (git source; its xformers/open3d/pycolmap/evo/e3nn/moviepy/gsplat deps are excluded, `core/depth_map.py` mocks the export modules). Plain `uv sync` keeps it; it was never declared before, so every earlier `uv sync` removed it.

## 6. Backend rules
1. Keep the SPA wildcard route `/{path:path}` LAST in `server.py`; routes after it are unreachable.
2. Make endpoints with blocking calls (HF `whoami()`, `os.walk`) sync `def`, not `async def`.
3. Guard every endpoint taking a temp file id: `path.resolve().is_relative_to(TEMP_DIR.resolve())`.
4. Use `manager.stop_requested`, never `_stop_event`, from `server.py`.
5. Use `core/lora_zimage.load_lora_for_pipeline()` for Z-Image LoRA; never `pipe.load_lora_weights()`.
6. Never use `FluxInpaintPipeline` with Flux2Klein (incompatible). Masked FLUX: crop mode = img2img on the crop + composite; "Inpainting Pipeline (Quality)" = `Flux2KleinInpaintPipeline` built from `pipe.components` (never `from_pipe`, it casts the quantized dtype; never `padding_mask_crop`, it returns a flat rectangle) + pixel composite; auto-outpaint stays on img2img (outpaint LoRA).
7. Model keys: internal `current_model` → `startswith("flux2")`; display `model_choice` → `startsWith('FLUX')`. Never mix.
8. Resize binary masks with `Image.NEAREST`, never LANCZOS.
9. Default model = `default_model` in `app_settings.json`, read in `App.tsx` bootstrap after `fetchSettings()`.
10. Every generation writes one run folder through `core/run_store.py` (`server._run_events`); never write flat outputs or sidecar JSONs. Client paths/names go through `run_store.safe_join` / `_guarded_temp`.

11. Model Sources: `core/model_sources.lora_family()` decides what the app can use (see architecture.md); update it when a model family is added (e.g. LTX-2), and keep manual "Add source" entries `custom: true` so they are never pruned. CivitAI rows are `provider: civitai`: the family comes from `baseModel` (`core/civitai.py`), never the name; the key lives in `civitai/token` (never log it).
12. Load every uploaded image through `server._load_pil` → `core/image_alpha.flatten_to_rgb()`; never `convert("RGB")` a ref (a logo PNG with same-colour pixels under its alpha became a solid block and the model never saw it). `_stored_sources()` maps a base row with empty `model_choice` to its `DEFAULT_SOURCES` row by URL (name + `model_choice`), or the card never shows installed.

## 7. Image pipeline rules
1. Never stretch slot #1: fit it with `fit_ref_to_canvas()` and give its mask the same transform.
2. Pass FLUX refs #2+ at native size (`prepare_flux_refs()`); the pipeline keeps their aspect.
3. In crop mode swap only slot #1 for its crop (`crop_flux_refs()`); keep material refs.
4. Keep `app._SNAP_REL` (3%) and the Sidebar size-note tolerance in sync.
5. Auto-size lives in the store reducer (`autoSizeParams()`), never in a `SizePanel` effect.
<!-- Accordion renders {open && children}: effects there miss collapsed state and re-fire on reopen -->
6. Mark slots restored from Load Params / workflows with `keepSize: true`.
7. All current models are distilled: guidance 0 (LTX 1.0), steps 20 FLUX / 4 Z-Image; guidance slider stays hidden.
8. FLUX 4B + padding → outpaint LoRA path (green pad, auto LoRA, trigger prompt); don't send it for 9B/Z-Image.
9. Load FLUX LoRAs only through `core/lora_flux2` (own BFL converter) and `sync_loras()`; never ignore a load status. `ensure_loras_loaded()` treats a status starting with `Loaded` as success, so a partial failure (Z-Image too) must never begin with it.
10. klein LoRAs are size-specific (4B hidden 3072, 9B 4096): classify with `lora_variant()` (header only), never hardcode block counts; `/api/lora/list` exposes `variant` and the UI greys mismatches (`src/loraCompat.ts`); keep `assert_lora_matches_model()` in both loaders.
11. Small mask (either mask mode): `plan_mask_crop()` gives a context crop upscaled toward 1 MP; generate on `ref_work`/`mask_work` (`gen_w/gen_h` differ from the output), then composite back via `mask_bbox` (`apply_mask_composite`). Large masks and auto-outpaint keep the old paths; never feed a small mask to a full-frame inpaint.

## 8. Video (LTX) rules
1. Use `LTXConditionPipeline` only.
2. Frames must be 8k+1, dims multiples of 32.
3. Keep `app.DOWNLOAD_IGNORE_PATTERNS`; never delete `vae/transformer` / `vae/text_encoder` blobs of an existing copy (deduped with the real weights).
4. Don't add FP8 LTX variants (unsuitable for Apple Silicon).

## 9. Frontend rules
1. Tailwind tokens: `bg:#0a0a0a` · `surface:#141414` · `card:#1c1c1c` · `border:#2a2a2a` · `accent:#7c3aed` · `muted/label:#6b7280`.
2. Never put `title` on a Gallery thumbnail's outer div (native tooltip).
3. Restore goes through `workflowToParams()` (`src/workflow.ts`); a new `GenerateParams` field that should survive reload goes in its key lists.
4. No UI test runner: verify pure-logic TS modules (`src/mask/*`, `src/slots.ts`) with `cd frontend && npm test` (node --test); verify UI in a real browser — claude-in-chrome is reliable for the mask editor, where agent-browser has frozen the tab on Invert/Grow after a polygon edit.
5. Mask editor keys: Alt = subtract (SAM/box/brush/polygon-close), Shift+click = SAM refine last object.
6. In claude-in-chrome UI checks, add ref slots by dispatching `dragover`+`drop` `DragEvent`s (a `DataTransfer` with `text/plain` = gallery image URL) on the "+ base img" / "+ ref img" button (or on a card to replace it; `application/x-ref-slot` = slotId on the base card swaps; for ref<->ref reorder fire `dragstart` on the source card, then `dragover`+`drop` on the target, with one shared `DataTransfer`) — no file dialog needed.
7. Ref row: a slot thumbnail click enlarges it (`SlotLightbox`); Ctrl/Cmd+click or drop replaces it. A ref dragged onto another ref reorders them (`swapRefs`, never touches the base). Base (slot #1) and References are separate; never shift a ref into the base. Base changes only via `REPLACE_SLOT_IMAGE` / `SWAP_WITH_BASE` (`src/slots.ts`); every mask is drawn on the base, so a base change clears all masks.
8. Show LoRA lists alphabetically by shown name (case-insensitive, `sortLoras()`; `display` from `/api/lora/list`, built by `core/lora_names.py`: no extension, `_` as spaces, CivitAI model name), never in server/mtime order; each selected LoRA shows its trigger hint (copy / insert at prompt start / edit) from `/api/lora/list` (`core/lora_triggers.py`: seed table by file name, user overrides in gitignored `lora_triggers.json`).
9. Anything that adds or removes a file in `lora_uploads/` from the UI (CivitAI download/delete) must `window.dispatchEvent(new Event('lora-library-changed'))`; `LoraPanel` reloads its list on it (the Accordion unmounts closed panels, so a mount-only load goes stale).
10. Iterate Masks pass inputs come from `iterateInputIds()` (`src/slots.ts`): base, the pass slot, then every unmasked ref. Never build `[base, slotN]` by hand; a ref without a mask has no pass of its own and was silently dropped (logo in slot #2 never reached the model).
11. Any workflow load (Workflows panel, or a gallery output via the modal Load button / Cmd-Ctrl-click) marks the canvas result stale (`MARK_RESULT_STALE`, dimmed + badge) so it reads as "settings loaded, tweak and Generate"; the gallery load still shows that run's image, just dimmed. Anything new that restores params should do the same.
