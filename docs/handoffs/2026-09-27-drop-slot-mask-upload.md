# Handoff: drop the slot mask file upload, make the in-app editor the entry point

Date: 2026-09-27 · Start with `cd ~/Projects/off-line-Image-gen-mac && claude`
Status: **implemented 2026-09-27** — pencil dropped, Import mask… added inside MaskEditor, #2+ hint added.

## Why

Masks are now built inside the app (MaskEditor: SAM click, box, brush, polygon, invert, grow).
Uploading a mask PNG per slot is a leftover from before the editor existed, and it is still the
**primary** UI: the empty mask box's only action is "Upload mask file", while drawing is a 9px pencil
on the image thumbnail that only shows on hover.

## Current state (`frontend/src/components/RefImagesRow.tsx`)

- `SlotCard` pencil button (~line 63) → `onDrawMask` → opens MaskEditor. Title is stale:
  "Draw mask by selecting a rectangle".
- Empty mask box (~line 96): `UploadCloud` button → hidden `<input type=file>` (~line 108) →
  `onUploadMask(file)`. Box title: "Upload mask or draw rectangle on image".
- Filled mask box: thumbnail + X (`onClearMask`). Clicking the thumbnail does nothing.
- MaskEditor is rendered once at ~line 290 with `baseImageUrl={slots[0].imageUrl}` (every slot's
  mask is drawn on slot #1's image) and its Apply calls **`onUploadMask(slotId, file)`** (~line 297).
  So `onUploadMask` / `App.tsx:285 handleUploadSlotMask` **must stay**; only the file-picker UI goes.
- MaskEditor already loads an existing `slot.maskUrl` as its starting mask (`MaskEditor.tsx:91-95`).

## Plan (one file, CLAUDE.md §2.2 → implement directly)

1. Empty mask box click → open MaskEditor for that slot (`setMaskEditorSlot(slot)`); icon/label
   "draw mask" instead of `UploadCloud` / upload.
2. Clicking an existing mask thumbnail → open MaskEditor to refine it. Keep the X to clear.
3. Remove the hidden file `<input>`, `maskRef`, and the `UploadCloud` import. Decide on the image
   thumbnail pencil: drop it as a duplicate, or keep with a corrected title.
4. Fix stale titles/tooltips (pencil title, box title, the Iterate/mask-mode HelpTip at ~line 260 if
   it mentions upload).

Unchanged on purpose: backend `/api/upload`, `mask_image_id`, workflow save/load of
`slot_N_mask.png`, the `/generate` skill (`local_run.py`) — external masks still arrive via API and
workflow folders.

Optional, only if Cris asks: "Import mask…" button **inside** MaskEditor to load a PNG as the
starting mask (for Photoshop-made masks), so import still goes through review.

## Verify

- `cd frontend && npm run build` and `npm test`; `venv/bin/python -m pytest -q` stays green (commit gate).
- Real browser via **claude-in-chrome** (not agent-browser — it froze the mask editor before):
  empty box opens editor on slot #1 and slot #2; Apply fills the mask thumb; thumb click reopens with
  the mask loaded; X clears; Load Params / workflow restore still shows masks.
- CHANGELOG entry; update `docs/architecture.md` if it describes slot mask upload.

## Related finding (not in scope, maybe a TODO)

Masks on slots #2+ are only used by **Iterate Masks** (`App.tsx:308`), which only appears when mask
mode = "Inpainting Pipeline (Quality)" (`Sidebar.tsx:1601`). Normal Generate sends only slot #1's
mask (`store.ts:113`), so a #2+ mask is silently ignored in every other mode. Worth either a UI hint
on #2+ mask boxes or a `docs/TODO.md` line — ask Cris.
