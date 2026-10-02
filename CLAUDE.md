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
4. Killing a `Launch.command` server also closes its Terminal window; restart from CC with `venv/bin/python server.py --port 7860` via `run_in_background`, and say so.

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

## 6. Backend rules
1. Keep the SPA wildcard route `/{path:path}` LAST in `server.py`; routes after it are unreachable.
2. Make endpoints with blocking calls (HF `whoami()`, `os.walk`) sync `def`, not `async def`.
3. Guard every endpoint taking a temp file id: `path.resolve().is_relative_to(TEMP_DIR.resolve())`.
4. Use `manager.stop_requested`, never `_stop_event`, from `server.py`.
5. Use `core/lora_zimage.load_lora_for_pipeline()` for Z-Image LoRA; never `pipe.load_lora_weights()`.
6. Never use `FluxInpaintPipeline` with Flux2Klein (incompatible); masked FLUX = img2img + composite.
7. Model keys: internal `current_model` → `startswith("flux2")`; display `model_choice` → `startsWith('FLUX')`. Never mix.
8. Resize binary masks with `Image.NEAREST`, never LANCZOS.
9. Default model = `default_model` in `app_settings.json`, read in `App.tsx` bootstrap after `fetchSettings()`.
10. Every generation writes one run folder through `core/run_store.py` (`server._run_events`); never write flat outputs or sidecar JSONs. Client paths/names go through `run_store.safe_join` / `_guarded_temp`.

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
9. Load FLUX LoRAs only through `core/lora_flux2` (own BFL converter) and `sync_loras()`; never ignore a load status.

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
6. In claude-in-chrome UI checks, add ref slots by dispatching `dragover`+`drop` `DragEvent`s (a `DataTransfer` with `text/plain` = gallery image URL) on the "+ base img" / "+ ref img" button (or on a card to replace it; `application/x-ref-slot` = slotId on the base card swaps) — no file dialog needed.
7. Ref row: Base (slot #1) and References are separate; never shift a ref into the base. Base changes only via `REPLACE_SLOT_IMAGE` / `SWAP_WITH_BASE` (`src/slots.ts`); every mask is drawn on the base, so a base change clears all masks.
