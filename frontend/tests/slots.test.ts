import { test } from 'node:test'
import assert from 'node:assert/strict'
import { replaceSlotImage, removeSlot, swapWithBase, hasMasks } from '../src/slots.ts'
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
