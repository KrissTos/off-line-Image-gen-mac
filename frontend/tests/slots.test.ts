import { test } from 'node:test'
import assert from 'node:assert/strict'
import { replaceSlotImage, removeSlot, swapWithBase, hasMasks, iterateInputIds, neighborSlot, isReplaceClick, swapRefs } from '../src/slots.ts'
import type { RefImageSlot } from '../src/types.ts'

const slot = (slotId: number, extra: Partial<RefImageSlot> = {}): RefImageSlot => ({
  slotId, imageId: `img${slotId}`, imageUrl: `/u/img${slotId}`,
  maskId: null, maskUrl: null, strength: 0.5 + slotId / 10, w: 100 * slotId, h: 50 * slotId,
  ...extra,
})
const masked = (slotId: number) => slot(slotId, { maskId: `m${slotId}`, maskUrl: `/u/m${slotId}` })

test('replacing the base swaps its image, clears every mask (drawn on the old base), keeps strength', () => {
  const out = replaceSlotImage([masked(1), masked(2)], 1, 'new', '/u/new')
  assert.equal(out[0].imageId, 'new')
  assert.equal(out[0].imageUrl, '/u/new')
  assert.equal(out[0].strength, 0.6)
  assert.equal(out[0].w, undefined)      // re-measured on load → auto-size runs again
  assert.equal(out[0].keepSize, undefined)
  assert.equal(hasMasks(out), false)
  assert.equal(out[1].imageId, 'img2')
})

test('replacing a ref keeps masks (they are drawn on the base) and the slot order', () => {
  const out = replaceSlotImage([masked(1), masked(2), slot(3)], 2, 'new', '/u/new')
  assert.deepEqual(out.map(s => s.imageId), ['img1', 'new', 'img3'])
  assert.equal(out[1].maskId, 'm2')
  assert.equal(out[0].maskId, 'm1')
  assert.equal(out[1].w, undefined)
})

test('removing a ref renumbers refs only; the base never moves', () => {
  const out = removeSlot([slot(1), slot(2), slot(3)], 2)
  assert.deepEqual(out.map(s => [s.slotId, s.imageId]), [[1, 'img1'], [2, 'img3']])
})

test('the base cannot be removed while refs exist', () => {
  const slots = [slot(1), slot(2)]
  assert.equal(removeSlot(slots, 1), slots)
})

test('the base alone can be removed', () => {
  assert.deepEqual(removeSlot([slot(1)], 1), [])
})

test('swapWithBase exchanges images + dims, keeps per-position strength, clears all masks', () => {
  const out = swapWithBase([masked(1), slot(2), masked(3)], 3)
  assert.deepEqual(out.map(s => [s.slotId, s.imageId, s.w]), [[1, 'img3', 300], [2, 'img2', 200], [3, 'img1', 100]])
  assert.deepEqual(out.map(s => s.strength), [0.6, 0.7, 0.8])
  assert.equal(hasMasks(out), false)
})

test('swapWithBase with the base itself or a missing slot is a no-op', () => {
  const slots = [slot(1), slot(2)]
  assert.equal(swapWithBase(slots, 1), slots)
  assert.equal(swapWithBase(slots, 9), slots)
})

test('neighborSlot steps through slots in order and stops at both ends', () => {
  const s = [slot(1), slot(2), slot(3)]
  assert.equal(neighborSlot(s, 2, 1)?.slotId, 3)
  assert.equal(neighborSlot(s, 2, -1)?.slotId, 1)
  assert.equal(neighborSlot(s, 3, 1), null)
  assert.equal(neighborSlot(s, 1, -1), null)
  assert.equal(neighborSlot(s, 9, 1), null)      // unknown slot (removed while open)
  assert.equal(neighborSlot([], 1, 1), null)
})

test('a plain click enlarges; Ctrl or Cmd click replaces', () => {
  assert.equal(isReplaceClick({ metaKey: false, ctrlKey: false }), false)
  assert.equal(isReplaceClick({ metaKey: true,  ctrlKey: false }), true)
  assert.equal(isReplaceClick({ metaKey: false, ctrlKey: true  }), true)
})

test('swapRefs exchanges two reference cards whole (image, dims, mask, strength); ids stay positional', () => {
  const out = swapRefs([masked(1), masked(2), slot(3), slot(4)], 2, 4)
  assert.deepEqual(out.map(s => s.slotId), [1, 2, 3, 4])
  assert.deepEqual(out.map(s => s.imageId), ['img1', 'img4', 'img3', 'img2'])
  assert.deepEqual(out.map(s => s.w), [100, 400, 300, 200])
  assert.deepEqual(out.map(s => s.strength), [0.6, 0.9, 0.8, 0.7])
  assert.equal(out[3].maskId, 'm2')                 // the mask travels with its image
  assert.equal(out[0].maskId, 'm1')                 // the base and its mask are untouched
})

test('swapRefs never touches the base and ignores same / missing slots', () => {
  const slots = [masked(1), slot(2), slot(3)]
  assert.equal(swapRefs(slots, 1, 2), slots)
  assert.equal(swapRefs(slots, 2, 1), slots)
  assert.equal(swapRefs(slots, 2, 2), slots)
  assert.equal(swapRefs(slots, 2, 9), slots)
})

test('iterate pass for a masked base still sends the unmasked refs, in slot order (img 2, img 3)', () => {
  const slots = [masked(1), slot(2), slot(3)]
  assert.deepEqual(iterateInputIds(slots, slots[0], 'base'), ['base', 'img2', 'img3'])
})

test('iterate pass for a masked ref: base, that ref, then the other unmasked refs', () => {
  const slots = [slot(1), masked(2), slot(3), masked(4)]
  assert.deepEqual(iterateInputIds(slots, slots[1], 'prev'), ['prev', 'img2', 'img3'])
})

test('iterate pass with no unmasked refs is unchanged from before', () => {
  const slots = [slot(1), masked(2)]
  assert.deepEqual(iterateInputIds(slots, slots[0], 'b'), ['b'])
  assert.deepEqual(iterateInputIds(slots, slots[1], 'b'), ['b', 'img2'])
})
