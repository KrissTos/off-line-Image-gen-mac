import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loraName, paramRows, loraRows, neighbor, upscaleSource } from '../src/previewModel.ts'
import type { OutputItem } from '../src/types.ts'

const mk = (over: Partial<OutputItem>): OutputItem =>
  ({ name: 'r/outputs/a.png', url: '/api/output/r/outputs/a.png', mtime: 1, kind: 'image', run: 'r', file: 'outputs/a.png', ...over })

test('seed 0 and steps 0 still produce rows', () => {
  const rows = paramRows(mk({ seed: 0, steps: 0, model_choice: 'FLUX.2-klein-9B', width: 1024, height: 768 }), false)
  assert.deepEqual(rows, [
    { label: 'Model', value: 'FLUX.2-klein-9B' },
    { label: 'Size', value: '1024 × 768' },
    { label: 'Steps', value: '0' },
    { label: 'Seed', value: '0' },
  ])
})

test('missing fields are skipped, mask mode only with a mask', () => {
  assert.deepEqual(paramRows(mk({ mask_mode: 'Crop & Composite (Fast)' }), false), [])
  assert.deepEqual(paramRows(mk({ mask_mode: 'Crop & Composite (Fast)' }), true),
    [{ label: 'Mask mode', value: 'Crop & Composite (Fast)' }])
})

test('video adds frames and fps', () => {
  assert.deepEqual(paramRows(mk({ kind: 'video', num_frames: 25, fps: 24 }), false),
    [{ label: 'Frames', value: '25 @ 24 fps' }])
})

test('loraName strips folders and extension', () => {
  assert.equal(loraName('/a/b/My_Lora.safetensors'), 'My_Lora')
  assert.equal(loraName('plain'), 'plain')
})

test('loraRows formats strength, empty when none', () => {
  assert.deepEqual(loraRows(mk({})), [])
  assert.deepEqual(loraRows(mk({ lora_files: [{ path: '/x/Eyes.safetensors', strength: 1 }] })),
    [{ label: 'Eyes', value: '1.00' }])
})

test('neighbor stops at both ends and tolerates an unknown url', () => {
  const a = mk({ url: 'a' }), b = mk({ url: 'b' }), c = mk({ url: 'c' })
  const list = [a, b, c]
  assert.equal(neighbor(list, 'b', -1), a)
  assert.equal(neighbor(list, 'b', 1), c)
  assert.equal(neighbor(list, 'a', -1), null)
  assert.equal(neighbor(list, 'c', 1), null)
  assert.equal(neighbor(list, 'zzz', 1), null)
})

test('upscaleSource finds the sibling, null when deleted or not an upscale', () => {
  const src = mk({ url: 's', file: 'outputs/s.png' })
  const up  = mk({ url: 'u', file: 'outputs/u.png' })
  const wf = { outputs: [{ file: 'outputs/s.png' }, { file: 'outputs/u.png', upscaled_from: 'outputs/s.png' }] }
  assert.equal(upscaleSource([up, src], wf, up), src)
  assert.equal(upscaleSource([up], wf, up), null)
  assert.equal(upscaleSource([up, src], wf, src), null)
  assert.equal(upscaleSource([up, src], null, up), null)
})

test('keyAction maps Esc/arrows, ignores modifiers and typing/seek targets', async () => {
  const { keyAction } = await import('../src/previewModel.ts')
  const k = (key: string, over = {}) => ({ key, metaKey: false, ctrlKey: false, altKey: false, targetTag: 'DIV', targetEditable: false, ...over })
  assert.equal(keyAction(k('Escape')), 'close')
  assert.equal(keyAction(k('ArrowLeft')), 'prev')
  assert.equal(keyAction(k('ArrowRight')), 'next')
  assert.equal(keyAction(k('ArrowLeft', { metaKey: true })), null)
  assert.equal(keyAction(k('ArrowRight', { altKey: true })), null)
  assert.equal(keyAction(k('ArrowRight', { targetTag: 'VIDEO' })), null)
  assert.equal(keyAction(k('ArrowLeft', { targetTag: 'TEXTAREA' })), null)
  assert.equal(keyAction(k('ArrowLeft', { targetEditable: true })), null)
  assert.equal(keyAction(k('Escape', { targetTag: 'VIDEO' })), 'close')
  assert.equal(keyAction(k('a')), null)
})
