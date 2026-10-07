import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  createMask, cloneMask, union, subtract, invert, paintStroke, fillPolygon,
  grow, shrink, coverage, iou, fromRgba, toRgba, paintOverlayRect, paintOverlayFull, hexToRgb,
  strokeRect, type Mask,
} from '../src/mask/maskOps.ts'

const at = (m: Mask, x: number, y: number) => m.data[y * m.w + x]
const count = (m: Mask) => m.data.reduce((s, v) => s + (v ? 1 : 0), 0)

test('union / subtract / invert', () => {
  const a = createMask(4, 1), b = createMask(4, 1)
  a.data.set([255, 255, 0, 0]); b.data.set([0, 255, 255, 0])
  assert.deepEqual([...union(a, b).data], [255, 255, 255, 0])
  assert.deepEqual([...subtract(a, b).data], [255, 0, 0, 0])
  assert.deepEqual([...invert(a).data], [0, 0, 255, 255])
  assert.deepEqual([...a.data], [255, 255, 0, 0])            // inputs untouched
})

test('cloneMask is a deep copy', () => {
  const a = createMask(2, 2); const c = cloneMask(a); c.data[0] = 255
  assert.equal(a.data[0], 0)
})

test('paintStroke stamps circles along the path', () => {
  const m = createMask(20, 20)
  paintStroke(m, [{ x: 5, y: 5 }], 2, 255)
  assert.equal(at(m, 7, 5), 255)          // distance 2
  assert.equal(at(m, 8, 5), 0)
  assert.equal(at(m, 7, 6), 0)            // distance √5 > 2
  paintStroke(m, [{ x: 2, y: 15 }, { x: 17, y: 15 }], 1, 255)
  for (let x = 2; x <= 17; x++) assert.equal(at(m, x, 15), 255)   // no gaps
  paintStroke(m, [{ x: 5, y: 5 }], 3, 0)
  assert.equal(at(m, 5, 5), 0)            // erase
})

test('fillPolygon: square and concave L', () => {
  const m = createMask(10, 10)
  fillPolygon(m, [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 8, y: 8 }, { x: 2, y: 8 }], 255)
  assert.equal(count(m), 36)
  assert.equal(at(m, 2, 2), 255); assert.equal(at(m, 7, 7), 255); assert.equal(at(m, 8, 8), 0)
  const l = createMask(10, 10)
  fillPolygon(l, [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 2 }, { x: 2, y: 2 }, { x: 2, y: 4 }, { x: 0, y: 4 }], 255)
  assert.equal(count(l), 12)              // 4x4 minus the 2x2 notch
  assert.equal(at(l, 3, 3), 0)
})

test('grow and shrink by exact Euclidean radius', () => {
  const m = createMask(30, 30); m.data[10 * 30 + 10] = 255
  const g = grow(m, 3)
  assert.equal(at(g, 13, 10), 255); assert.equal(at(g, 14, 10), 0)
  assert.equal(at(g, 12, 12), 255)        // d² = 8 ≤ 9
  assert.equal(at(g, 13, 12), 0)          // d² = 13 > 9
  const sq = createMask(20, 20)
  fillPolygon(sq, [{ x: 5, y: 5 }, { x: 15, y: 5 }, { x: 15, y: 15 }, { x: 5, y: 15 }], 255)
  const s = shrink(sq, 2)
  assert.equal(count(s), 36)              // 10x10 → 6x6
  assert.equal(at(s, 7, 7), 255); assert.equal(at(s, 6, 7), 0)
  assert.equal(count(grow(createMask(5, 5), 3)), 0)       // empty stays empty
})

test('coverage and iou', () => {
  const a = createMask(4, 1), b = createMask(4, 1)
  a.data.set([255, 255, 0, 0]); b.data.set([0, 255, 255, 0])
  assert.equal(coverage(a), 0.5)
  assert.equal(iou(a, b), 1 / 3)
  assert.equal(iou(createMask(2, 2), createMask(2, 2)), 1)   // both empty = identical
})

test('fromRgba resizes nearest-neighbour and thresholds; toRgba round-trips', () => {
  // 2x1 source: white, black → 4x2 target
  const src = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255])
  const m = fromRgba(src, 2, 1, 4, 2)
  assert.deepEqual([...m.data], [255, 255, 0, 0, 255, 255, 0, 0])
  const back = fromRgba(toRgba(m), 4, 2, 4, 2)
  assert.deepEqual([...back.data], [...m.data])
  const grey = fromRgba(new Uint8ClampedArray([127, 127, 127, 255]), 1, 1, 1, 1)
  assert.equal(grey.data[0], 0)
})

test('paintOverlayRect over the full image matches paintOverlayFull', () => {
  const m = createMask(5, 4)
  m.data.set([255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0])
  const full = new Uint8ClampedArray(m.w * m.h * 4)
  paintOverlayFull(full, m)
  const viaRect = new Uint8ClampedArray(m.w * m.h * 4)
  paintOverlayRect(viaRect, m, 0, 0, m.w - 1, m.h - 1)
  assert.deepEqual([...viaRect], [...full])
  // on pixels are red and opaque (opacity is applied at draw time), off pixels are fully transparent
  assert.deepEqual([...full.slice(0, 4)], [239, 68, 68, 255])
  assert.deepEqual([...full.slice(4, 8)], [0, 0, 0, 0])
})

test('paintOverlayRect only writes pixels inside the given rect', () => {
  const m = createMask(4, 4); m.data.fill(255)
  const out = new Uint8ClampedArray(m.w * m.h * 4)
  paintOverlayRect(out, m, 1, 1, 2, 2)
  assert.deepEqual([...out.slice(0, 4)], [0, 0, 0, 0])              // (0,0) untouched
  const i = (1 * m.w + 1) * 4
  assert.deepEqual([...out.slice(i, i + 4)], [239, 68, 68, 255])    // (1,1) painted
})

test('paintOverlayRect uses the given color; off pixels stay transparent', () => {
  const m = createMask(2, 1); m.data.set([255, 0])
  const out = new Uint8ClampedArray(8)
  paintOverlayFull(out, m, [10, 200, 30])
  assert.deepEqual([...out], [10, 200, 30, 255, 0, 0, 0, 0])
})

test('hexToRgb parses #rrggbb and falls back to the default red on bad input', () => {
  assert.deepEqual(hexToRgb('#0a1b2c'), [10, 27, 44])
  assert.deepEqual(hexToRgb('nope'), [239, 68, 68])
})

test('strokeRect bounds a stroke by radius, clamped to the image', () => {
  assert.deepEqual(strokeRect([{ x: 5, y: 5 }], 2, 20, 20), { x0: 3, y0: 3, x1: 7, y1: 7 })
  assert.deepEqual(strokeRect([{ x: 0, y: 0 }], 3, 10, 10), { x0: 0, y0: 0, x1: 3, y1: 3 })
  assert.deepEqual(strokeRect([{ x: 2, y: 2 }, { x: 8, y: 8 }], 1, 20, 20), { x0: 1, y0: 1, x1: 9, y1: 9 })
})
