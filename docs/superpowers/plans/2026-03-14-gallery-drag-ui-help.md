# Gallery Drag + UI Help System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make gallery thumbnails draggable into ref image slots and IP-Adapter slots, and add an inline `HelpTip` tooltip component to every non-obvious UI control.

**Architecture:** `HelpTip.tsx` is a tiny reusable component (icon + hover tooltip). `Gallery.tsx` gets `draggable` + `onDragStart`. `RefImagesRow.tsx` gets `onDragOver` + `onDrop`. `IpAdapterPanel.tsx` already has drop support (Plan A). `Sidebar.tsx` gets `HelpTip` scattered through all accordions.

**Tech Stack:** React, TypeScript, Tailwind CSS v3, existing `uploadFromUrl` / `uploadImage` API helpers.

---

## Chunk 1: HelpTip Component

### Task 1: Create `HelpTip.tsx`

**Files:**
- Create: `frontend/src/components/HelpTip.tsx`

- [ ] **Step 1: Create the component**

```tsx
// frontend/src/components/HelpTip.tsx
/**
 * Inline help tooltip. Renders a small ⓘ icon that shows a tooltip on hover.
 * Usage: <HelpTip text="Guidance scale controls…" />
 */
import { useState } from 'react'

interface Props {
  text: string
  position?: 'top' | 'bottom' | 'left' | 'right'
}

export default function HelpTip({ text, position = 'top' }: Props) {
  const [visible, setVisible] = useState(false)

  const posClass: Record<string, string> = {
    top:    'bottom-full left-1/2 -translate-x-1/2 mb-1.5',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-1.5',
    left:   'right-full top-1/2 -translate-y-1/2 mr-1.5',
    right:  'left-full top-1/2 -translate-y-1/2 ml-1.5',
  }

  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {/* ⓘ icon */}
      <span className="w-3.5 h-3.5 rounded-full border border-[var(--color-muted)] text-[var(--color-muted)] text-[9px] flex items-center justify-center cursor-help select-none hover:border-white hover:text-white transition-colors">
        i
      </span>

      {/* Tooltip */}
      {visible && (
        <span
          className={`absolute z-50 ${posClass[position]} w-56 p-2 rounded bg-[#2a2a2a] border border-[var(--color-border)] text-[11px] text-[var(--color-muted)] leading-relaxed shadow-lg pointer-events-none`}
        >
          {text}
        </span>
      )}
    </span>
  )
}
```

- [ ] **Step 2: Build and verify**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend
npm run build 2>&1 | tail -5
```
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/HelpTip.tsx
git commit -m "feat: HelpTip component — inline ⓘ icon with hover tooltip"
```

---

## Chunk 2: Gallery Drag Support

### Task 2: Make gallery thumbnails draggable

**Files:**
- Modify: `frontend/src/components/Gallery.tsx`

- [ ] **Step 1: Read the current Gallery.tsx to understand thumbnail rendering**

Look for how each `OutputItem` thumbnail is rendered — it will be an `<img>` or `<div>` inside a mapped list.

- [ ] **Step 2: Add `draggable` and `onDragStart` to each thumbnail**

Find the thumbnail element and add:
```tsx
draggable
onDragStart={e => {
  // Store the output URL so drop targets can fetch and re-upload it
  e.dataTransfer.setData('text/plain', item.url)
  e.dataTransfer.effectAllowed = 'copy'
}}
```

- [ ] **Step 3: Add visual drag affordance**

Add `cursor-grab active:cursor-grabbing` and `hover:opacity-80 transition-opacity` to the thumbnail className.

Optionally add a small drag indicator icon overlay on hover:
```tsx
// Inside the thumbnail wrapper, add:
<span className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
  <span className="text-white/70 text-xs bg-black/40 px-1 py-0.5 rounded">drag</span>
</span>
```
(Requires `group` class on the wrapper)

- [ ] **Step 4: Build and verify**

```bash
npm run build 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/Gallery.tsx
git commit -m "feat: Gallery — thumbnails are draggable, transfer output URL on drag"
```

---

### Task 3: Make RefImagesRow slots accept dropped gallery images

**Files:**
- Modify: `frontend/src/components/RefImagesRow.tsx`

- [ ] **Step 1: Read RefImagesRow.tsx to understand slot rendering and existing handlers**

Look for: the "add slot" / drop zone area and existing `onDrop` handling if any.

- [ ] **Step 2: Add drag-over state**

```tsx
const [dragOverNew, setDragOverNew] = useState(false)
```

- [ ] **Step 3: Add drop handler for the "add new slot" zone**

```tsx
async function handleRefDrop(e: React.DragEvent) {
  e.preventDefault()
  setDragOverNew(false)

  // File drop (direct file from OS)
  const file = e.dataTransfer.files[0]
  if (file) {
    const { id, url } = await uploadImage(file)
    dispatch({ type: 'ADD_REF_SLOT', imageId: id, imageUrl: url })
    return
  }

  // Gallery drag (URL string)
  const srcUrl = e.dataTransfer.getData('text/plain')
  if (srcUrl) {
    try {
      const { id, url } = await uploadFromUrl(srcUrl)
      dispatch({ type: 'ADD_REF_SLOT', imageId: id, imageUrl: url })
    } catch (e) {
      console.error('Drop upload failed', e)
    }
  }
}
```

- [ ] **Step 4: Wire drag events onto the "add" drop zone**

Find the "+" / drop zone element and add:
```tsx
onDragOver={e => { e.preventDefault(); setDragOverNew(true) }}
onDragLeave={() => setDragOverNew(false)}
onDrop={handleRefDrop}
className={`… ${dragOverNew ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : ''}`}
```

- [ ] **Step 5: Also support dropping onto existing slot thumbnails (to replace)**

For each existing slot thumbnail wrapper, add:
```tsx
onDragOver={e => e.preventDefault()}
onDrop={async e => {
  e.preventDefault()
  const srcUrl = e.dataTransfer.getData('text/plain')
  if (!srcUrl) return
  // Remove old slot, add new one at same position — simplest approach
  dispatch({ type: 'REMOVE_REF_SLOT', slotId: slot.slotId })
  const { id, url } = await uploadFromUrl(srcUrl)
  dispatch({ type: 'ADD_REF_SLOT', imageId: id, imageUrl: url })
}}
```

- [ ] **Step 6: Import helpers at top of file**

```tsx
import { uploadImage, uploadFromUrl } from '../api'
```

- [ ] **Step 7: Build and test manually**

```bash
npm run build && echo "Build OK"
```
Open app → generate an image → drag it from gallery into a ref slot → thumbnail should appear in slot.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/RefImagesRow.tsx
git commit -m "feat: RefImagesRow — accept gallery drag-and-drop into ref image slots"
```

---

## Chunk 3: UI Help Text — Sidebar Controls

### Task 4: Add `HelpTip` to all Sidebar controls

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`

Import `HelpTip` at the top:
```tsx
import HelpTip from './HelpTip'
```

Then add `<HelpTip text="…" />` after each label for the following controls. The exact JSX location depends on the current label structure — look for `<label>` or `<span>` text and add the HelpTip inline.

- [ ] **Step 1: Guidance scale**

```tsx
<HelpTip text="Controls how strictly the model follows your prompt. Lower = more creative freedom, higher = more literal. Z-Image Turbo always uses 0 (distilled model)." />
```

- [ ] **Step 2: Steps**

```tsx
<HelpTip text="Number of denoising steps. More steps = more detail and coherence but slower. FLUX.2: 20, Z-Image Turbo: 4, LTX-Video: 25. Going beyond recommended values rarely helps." />
```

- [ ] **Step 3: Seed**

```tsx
<HelpTip text="Random seed for reproducibility. -1 picks a new random seed each time. Set a fixed number to regenerate the same image with different settings." />
```

- [ ] **Step 4: Image strength (img_strength)**

```tsx
<HelpTip text="How much of the reference image to preserve in img2img mode. 0 = keep reference exactly, 1 = ignore reference and generate freely. 0.6–0.8 gives good results for style transfer." />
```

- [ ] **Step 5: Repeat count**

```tsx
<HelpTip text="Generate N images in sequence with the same settings (but different random seeds if seed is -1). Each result is saved individually." />
```

- [ ] **Step 6: LoRA strength**

```tsx
<HelpTip text="How strongly the LoRA style is applied. 0 = no effect, 1 = full LoRA strength. Values above 1 are possible but may cause artifacts." />
```

- [ ] **Step 7: Mask mode dropdown**

```tsx
<HelpTip text="Crop & Composite (Fast): fast inpainting — only the masked area is regenerated and composited back. Inpainting Pipeline (Quality): slower but more coherent, rerenders with full context awareness." />
```

- [ ] **Step 8: Outpaint align (9-position picker)**

```tsx
<HelpTip text="When your reference image is smaller than the output size, choose which corner or edge to anchor it to. The remaining area is auto-filled (outpainting)." />
```

- [ ] **Step 9: Model selector**

```tsx
<HelpTip text="FLUX.2-klein: best quality, slower. Z-Image Turbo: fastest (4 steps), good for iteration. LTX-Video: generates short video clips from text or image." />
```

- [ ] **Step 10: Build and verify**

```bash
npm run build 2>&1 | tail -5
```

- [ ] **Step 11: Commit**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: Sidebar — add HelpTip tooltips to all non-obvious controls"
```

---

### Task 5: Add `HelpTip` to remaining components

**Files:**
- Modify: `frontend/src/components/RefImagesRow.tsx`
- Modify: `frontend/src/components/IpAdapterPanel.tsx` (already has some from Plan A)

- [ ] **Step 1: RefImagesRow — mask mode label**

```tsx
<HelpTip text="When you have a mask drawn, this controls how inpainting is applied. Use Iterate Masks mode (bottom button) to apply different masks from different reference slots sequentially." />
```

- [ ] **Step 2: RefImagesRow — per-slot strength slider label**

```tsx
<HelpTip text="How strongly this reference image blends into the output. Slot #1 is the base image — lower values preserve more of the original. Slot #2+ are style references." />
```

- [ ] **Step 3: Build final verification**

```bash
npm run build 2>&1 | grep -E "error|✓|built" | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/RefImagesRow.tsx
git commit -m "feat: RefImagesRow — HelpTip tooltips on mask mode and slot strength"
```

---

## Chunk 4: Final build + rebuild dist

### Task 6: Final production build

- [ ] **Step 1: Build frontend**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac/frontend
npm run build
```

- [ ] **Step 2: Manual smoke test**

1. Open `http://localhost:7860`
2. Hover over any `ⓘ` icon — tooltip should appear
3. Generate an image → drag thumbnail from gallery → should land in ref slot
4. IP-Adapter accordion should be present
5. LoRA accordion should appear when a FLUX model is selected

- [ ] **Step 3: Final commit**

```bash
cd /Users/cris/Projects/off-line-Image-gen-mac
git add frontend/dist
git commit -m "chore: rebuild frontend/dist with gallery-drag + ui-help features"
```
