import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createMask, type Mask } from '../src/mask/maskOps.ts'
import { MaskHistory } from '../src/mask/history.ts'

const mk = (v: number) => { const m = createMask(2, 2); m.data.fill(v); return m }
const val = (m: Mask | null) => m?.data[0]

test('undo/redo order', () => {
  const h = new MaskHistory()
  h.push(mk(1)); h.push(mk(2))            // states before two edits
  const cur = mk(3)
  const u1 = h.undo(cur); assert.equal(val(u1), 2)
  const u2 = h.undo(u1!); assert.equal(val(u2), 1)
  assert.equal(h.undo(u2!), null)
  const r1 = h.redo(u2!); assert.equal(val(r1), 2)
  const r2 = h.redo(r1!); assert.equal(val(r2), 3)
  assert.equal(h.canRedo, false)
})

test('a new push clears redo', () => {
  const h = new MaskHistory()
  h.push(mk(1)); h.undo(mk(2)); assert.equal(h.canRedo, true)
  h.push(mk(5)); assert.equal(h.canRedo, false)
})

test('depth cap', () => {
  const h = new MaskHistory(3)
  for (let i = 1; i <= 5; i++) h.push(mk(i))
  let cur: Mask | null = mk(9), n = 0, last = 0
  while ((cur = h.undo(cur!))) { n++; last = val(cur)! }
  assert.equal(n, 3); assert.equal(last, 3)             // oldest two dropped
})

test('byte cap limits snapshots of large masks', () => {
  const h = new MaskHistory(30, 10 * 4)                 // room for 10 masks of 4 bytes
  for (let i = 0; i < 25; i++) h.push(mk(i))
  let cur: Mask | null = mk(99), n = 0
  while ((cur = h.undo(cur!))) n++
  assert.equal(n, 10)
})

test('snapshots are copies', () => {
  const h = new MaskHistory(); const m = mk(1)
  h.push(m); m.data.fill(7)
  assert.equal(val(h.undo(mk(2))), 1)
})
