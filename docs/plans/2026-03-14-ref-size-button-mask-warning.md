# Ref Size Button + Mask Mode Warning — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Use ref size" button to SizePanel that snaps output dimensions to the uploaded reference image, and show amber warnings when "Inpainting Pipeline (Quality)" is selected with a FLUX.2 model.

**Architecture:** Capture ref image natural dimensions via `onLoad` on the thumbnail `<img>` in `SlotCard`, store them in `RefImageSlot` state, pass slot #1's dims to `SizePanel` as a new prop, and render the button + warnings using data already flowing through the component tree.

**Tech Stack:** React, TypeScript, Tailwind CSS v3, `useReducer` global store

---

## Task 1: Extend RefImageSlot type + store

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/store.ts`

- [ ] **Step 1: Add optional dims to RefImageSlot in types.ts**

In `frontend/src/types.ts`, add `w` and `h` to the `RefImageSlot` interface:

```typescript
export interface RefImageSlot {
  slotId:   number
  imageId:  string
  imageUrl: string
  maskId:   string | null
  maskUrl:  string | null
  strength: number
  w?:       number   // natural image width (populated on thumbnail load)
  h?:       number   // natural image height
}
```

- [ ] **Step 2: Add SET_SLOT_DIMS action to store.ts**

In `frontend/src/store.ts`, add to the `Action` union type (after `UPDATE_SLOT_STRENGTH`):

```typescript
| { type: 'SET_SLOT_DIMS'; slotId: number; w: number; h: number }
```

- [ ] **Step 3: Add SET_SLOT_DIMS case to reducer**

In the `reducer` function in `store.ts`, add after `UPDATE_SLOT_STRENGTH` case:

```typescript
case 'SET_SLOT_DIMS': {
  const slots = state.refSlots.map(s =>
    s.slotId === action.slotId ? { ...s, w: action.w, h: action.h } : s
  )
  return { ...state, refSlots: slots }
}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npx tsc --noEmit
```
Expected: no errors

- [ ] **Step 5: Commit**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac
git add frontend/src/types.ts frontend/src/store.ts
git commit -m "feat: add dims fields to RefImageSlot + SET_SLOT_DIMS action"
```

---

## Task 2: Capture dims in SlotCard on thumbnail load

**Files:**
- Modify: `frontend/src/components/RefImagesRow.tsx` (SlotCard only)

- [ ] **Step 1: Add onDimsLoaded prop to SlotCardProps**

In `RefImagesRow.tsx`, find `interface SlotCardProps` and add:

```typescript
onDimsLoaded?: (w: number, h: number) => void
```

- [ ] **Step 2: Destructure onDimsLoaded in SlotCard**

In the `SlotCard` function signature, add `onDimsLoaded` to destructured props:

```typescript
function SlotCard({ slot, isBase, thumbSize, onRemove, onUploadMask, onClearMask, onDrawMask, onStrengthChange, onDimsLoaded }: SlotCardProps) {
```

- [ ] **Step 3: Add onLoad to the reference image thumbnail**

Find the `<img>` tag inside SlotCard (the reference image thumbnail):
```tsx
<img src={slot.imageUrl} alt={`ref #${slot.slotId}`} className="w-full h-full object-cover" />
```

Replace with:
```tsx
<img
  src={slot.imageUrl}
  alt={`ref #${slot.slotId}`}
  className="w-full h-full object-cover"
  onLoad={e => {
    const img = e.currentTarget
    onDimsLoaded?.(img.naturalWidth, img.naturalHeight)
  }}
/>
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/RefImagesRow.tsx
git commit -m "feat: capture ref image natural dims on thumbnail load"
```

---

## Task 3: Wire dims through RefImagesRow + App.tsx

**Files:**
- Modify: `frontend/src/components/RefImagesRow.tsx` (RefImagesRow props + SlotCard usage)
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add onSlotDimsLoaded to RefImagesRow Props interface**

In `RefImagesRow.tsx`, find `interface Props` and add:

```typescript
onSlotDimsLoaded?: (slotId: number, w: number, h: number) => void
```

- [ ] **Step 2: Destructure and pass through in RefImagesRow**

In the `RefImagesRow` function signature, add `onSlotDimsLoaded` to destructured props.

In the `slots.map(slot => ...)` section, pass it to `SlotCard`:

```tsx
onDimsLoaded={(w, h) => onSlotDimsLoaded?.(slot.slotId, w, h)}
```

- [ ] **Step 3: Pass handler from App.tsx to RefImagesRow**

In `App.tsx`, find the `<RefImagesRow ...>` JSX and add:

```tsx
onSlotDimsLoaded={(slotId, w, h) =>
  dispatch({ type: 'SET_SLOT_DIMS', slotId, w, h })
}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/RefImagesRow.tsx frontend/src/App.tsx
git commit -m "feat: wire slot dims from thumbnail load through to store"
```

---

## Task 4: "Use ref size" button in SizePanel

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Add refImageSize prop to SizePanel**

Find the `SizePanel` function signature:

```typescript
function SizePanel({
  params, onChange, hasRefImage,
}: {
  params: GenerateParams
  onChange: (k: keyof GenerateParams, v: unknown) => void
  hasRefImage: boolean
})
```

Replace with:

```typescript
function SizePanel({
  params, onChange, hasRefImage, refImageSize,
}: {
  params: GenerateParams
  onChange: (k: keyof GenerateParams, v: unknown) => void
  hasRefImage: boolean
  refImageSize?: { w: number; h: number }
})
```

- [ ] **Step 2: Add snap-to-64 helper above SizePanel**

Add this function just above `SizePanel` (near the `presetsForModel` helper):

```typescript
function snapTo64(n: number): number {
  return Math.max(64, Math.round(n / 64) * 64)
}
```

- [ ] **Step 3: Add "Use ref size" button to SizePanel JSX**

Inside `SizePanel`, find the Width/Height inputs block:

```tsx
<div className="flex gap-2">
  <NumberInput label="Width"  value={params.width}  onChange={v => onChange('width', v)} />
  <NumberInput label="Height" value={params.height} onChange={v => onChange('height', v)} />
</div>
```

Replace with:

```tsx
<div className="flex gap-2 items-end">
  <NumberInput label="Width"  value={params.width}  onChange={v => onChange('width', v)} />
  <NumberInput label="Height" value={params.height} onChange={v => onChange('height', v)} />
  {refImageSize && (
    <button
      title={`Set to ref image size (${refImageSize.w}×${refImageSize.h} → snapped to 64)`}
      onClick={() => {
        onChange('width',  snapTo64(refImageSize.w))
        onChange('height', snapTo64(refImageSize.h))
      }}
      className="shrink-0 mb-[1px] px-2 py-1 rounded bg-card border border-border text-[10px] text-muted
                 hover:text-white hover:border-accent transition-colors whitespace-nowrap"
    >
      ↕ ref size
    </button>
  )}
</div>
```

- [ ] **Step 4: Pass refImageSize from Sidebar to SizePanel**

Find the `<SizePanel ...>` usage in the Sidebar's JSX (inside the "Output Size" Accordion):

```tsx
<SizePanel params={params} onChange={onParamChange} hasRefImage={hasRefImage} />
```

Add `refImageSize` to Sidebar's props interface and pass it through:

In `interface SidebarProps` (find the Sidebar's Props type), add:
```typescript
refImageSize?: { w: number; h: number }
```

Update `<SizePanel>` call:
```tsx
<SizePanel params={params} onChange={onParamChange} hasRefImage={hasRefImage} refImageSize={refImageSize} />
```

- [ ] **Step 5: Pass refImageSize from App.tsx to Sidebar**

In `App.tsx`, find `<Sidebar ...>` and add:

```tsx
refImageSize={
  state.refSlots[0]?.w && state.refSlots[0]?.h
    ? { w: state.refSlots[0].w, h: state.refSlots[0].h }
    : undefined
}
```

- [ ] **Step 6: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npx tsc --noEmit
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/Sidebar.tsx frontend/src/App.tsx
git commit -m "feat: add Use ref size button to SizePanel"
```

---

## Task 5: Mask mode warning in RefImagesRow

**Files:**
- Modify: `frontend/src/components/RefImagesRow.tsx`

- [ ] **Step 1: Add modelChoice prop to RefImagesRow Props**

In `interface Props`, add:

```typescript
modelChoice: string
```

- [ ] **Step 2: Destructure modelChoice in RefImagesRow**

Add `modelChoice` to the destructured props.

- [ ] **Step 3: Add inline warning below the mask mode select**

Find the mask mode dropdown section. After the `</select>` (and the existing multi-mask hint `<p>`), add:

```tsx
{modelChoice.startsWith('FLUX') && maskMode === 'Inpainting Pipeline (Quality)' && (
  <p className="text-[9px] text-amber-400/80 bg-amber-900/20 border border-amber-800/30
                rounded px-1.5 py-1 leading-tight mt-1">
    ⓘ FLUX.2-klein doesn't support inpainting — will use img2img instead
  </p>
)}
```

- [ ] **Step 4: Pass modelChoice from App.tsx to RefImagesRow**

In `App.tsx`, find `<RefImagesRow ...>` and add:

```tsx
modelChoice={state.params.model_choice}
```

- [ ] **Step 5: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npx tsc --noEmit
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/RefImagesRow.tsx frontend/src/App.tsx
git commit -m "feat: add FLUX inpainting fallback warning in RefImagesRow"
```

---

## Task 6: Mask mode warning in Sidebar near Generate button

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Locate the Generate/Iterate/Stop button area**

In `Sidebar.tsx`, find the bottom button section — it renders the Stop, Iterate Masks, or Generate button. It's near the bottom of the Sidebar JSX.

- [ ] **Step 2: Add amber warning above the button**

Just above the button group (the `<div>` containing the Stop/Iterate/Generate button), add:

```tsx
{params.model_choice.startsWith('FLUX') &&
 params.mask_mode === 'Inpainting Pipeline (Quality)' &&
 (
  <p className="text-[10px] text-amber-400/70 bg-amber-900/20 border border-amber-800/30
                rounded px-2 py-1 leading-tight">
    ⚠ Inpainting Pipeline unavailable for FLUX.2-klein — will use img2img
  </p>
)}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: add FLUX inpainting fallback warning near Generate button"
```

---

## Task 7: Build and verify

- [ ] **Step 1: Full frontend build**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend && npm run build
```
Expected: Build succeeds with no errors.

- [ ] **Step 2: Manual verification checklist**

Start the server and open the UI (`./Launch.command --dev`), then verify:

1. Upload a non-square ref image (e.g. 1200×800) → open Output Size accordion → "↕ ref size" button appears → click it → Width becomes 1152, Height becomes 768 (nearest 64 multiples of 1200/800 scaled, or 1200→1216 and 800→768 etc)
2. With no ref image → "↕ ref size" button is hidden
3. Select a FLUX.2 model + upload any ref image + draw a mask + set mask mode to "Inpainting Pipeline (Quality)" → amber warning appears below the dropdown in RefImagesRow AND above the Generate button in Sidebar
4. Switch mask mode to "Crop & Composite (Fast)" → warnings disappear
5. Switch to Z-Image model → warnings don't appear even with Inpainting mode set

- [ ] **Step 3: Push**

```bash
bash ~/.claude/skills/git-pushing/scripts/smart_commit.sh "feat: ref size button and FLUX inpainting warnings"
```
