// Pure slot-list operations for the Base / References split.
// Slot #1 is the base; #2+ are references. Every mask is drawn on the base
// image (MaskEditor uses slots[0]), so a base change invalidates all masks,
// while a ref change keeps them.
import type { RefImageSlot } from './types'

const noMask = { maskId: null, maskUrl: null }

export const hasMasks = (slots: RefImageSlot[]) => slots.some(s => !!s.maskId)

/** Swap one slot's image in place; dims reset so the new image is re-measured (and auto-sized if base). */
export function replaceSlotImage(slots: RefImageSlot[], slotId: number, imageId: string, imageUrl: string): RefImageSlot[] {
  const isBase = slotId === 1
  return slots.map(s => {
    if (s.slotId === slotId) {
      return { ...s, imageId, imageUrl, w: undefined, h: undefined, keepSize: undefined, ...(isBase ? noMask : {}) }
    }
    return isBase ? { ...s, ...noMask } : s
  })
}

/** Remove a ref and renumber refs. The base is removable only when it is the last slot. */
export function removeSlot(slots: RefImageSlot[], slotId: number): RefImageSlot[] {
  if (slotId === 1 && slots.length > 1) return slots
  return slots
    .filter(s => s.slotId !== slotId)
    .map((s, i) => ({ ...s, slotId: i + 1 }))
}

/** Promote a ref to base: exchange the two images (strengths stay with their position); all masks cleared. */
export function swapWithBase(slots: RefImageSlot[], slotId: number): RefImageSlot[] {
  const base = slots[0]
  const ref  = slots.find(s => s.slotId === slotId)
  if (!base || !ref || slotId === 1) return slots
  const image = (s: RefImageSlot) => ({ imageId: s.imageId, imageUrl: s.imageUrl, w: s.w, h: s.h })
  return slots.map(s => {
    if (s.slotId === 1)      return { ...s, ...image(ref),  keepSize: undefined, ...noMask }
    if (s.slotId === slotId) return { ...s, ...image(base), keepSize: undefined, ...noMask }
    return { ...s, ...noMask }
  })
}

/** Reorder two references by exchanging whole cards (image, dims, mask, strength); slot ids stay positional.
 *  The base is never involved (use swapWithBase), so no mask is invalidated. */
export function swapRefs(slots: RefImageSlot[], a: number, b: number): RefImageSlot[] {
  const A = slots.find(s => s.slotId === a)
  const B = slots.find(s => s.slotId === b)
  if (!A || !B || a === b || a === 1 || b === 1) return slots
  return slots.map(s =>
    s.slotId === a ? { ...B, slotId: a } : s.slotId === b ? { ...A, slotId: b } : s)
}

/** The slot before (-1) or after (+1) `slotId` in display order; null at the ends or for an unknown id. */
export function neighborSlot(slots: RefImageSlot[], slotId: number, dir: -1 | 1): RefImageSlot | null {
  const i = slots.findIndex(s => s.slotId === slotId)
  return i < 0 ? null : (slots[i + dir] ?? null)
}

/** Ctrl/Cmd-click on a slot image replaces it (file picker); a plain click enlarges it. */
export const isReplaceClick = (e: { metaKey: boolean; ctrlKey: boolean }) => e.metaKey || e.ctrlKey
