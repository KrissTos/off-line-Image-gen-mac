import { test } from 'node:test'
import assert from 'node:assert/strict'
import { fitView, zoomAt, screenToImage, imageToScreen, inImage } from '../src/mask/viewMath.ts'

test('fitView centres and fits the image', () => {
  const v = fitView(2000, 1000, 1000, 1000, 1)
  assert.equal(v.scale, 0.5); assert.equal(v.tx, 0); assert.equal(v.ty, 250)
})

test('screen ↔ image round trip', () => {
  const v = { scale: 0.5, tx: 10, ty: 20 }
  const p = screenToImage(v, 110, 70)
  assert.deepEqual(p, { x: 200, y: 100 })
  assert.deepEqual(imageToScreen(v, p.x, p.y), { x: 110, y: 70 })
})

test('zoomAt keeps the point under the cursor fixed and clamps', () => {
  const v = { scale: 1, tx: 0, ty: 0 }
  const z = zoomAt(v, 100, 50, 2)
  assert.equal(z.scale, 2)
  assert.deepEqual(screenToImage(z, 100, 50), screenToImage(v, 100, 50))
  assert.equal(zoomAt(v, 0, 0, 100).scale, 8)
  assert.equal(zoomAt(v, 0, 0, 0.001).scale, 0.1)
})

test('inImage', () => {
  assert.equal(inImage({ x: 0, y: 0 }, 10, 10), true)
  assert.equal(inImage({ x: 9.9, y: 5 }, 10, 10), true)
  assert.equal(inImage({ x: 10, y: 5 }, 10, 10), false)
  assert.equal(inImage({ x: -0.1, y: 5 }, 10, 10), false)
})
