import { test } from 'node:test'
import assert from 'node:assert/strict'
import { workflowToParams } from '../src/workflow.ts'

test('every scalar survives, including seed 0 and guidance 0', () => {
  const p = workflowToParams({
    prompt: '', seed: 0, width: 1328, height: 784, steps: 20, guidance: 0, img_strength: 0.8,
    repeat_count: 2, num_frames: 25, fps: 24, model_choice: 'FLUX.2-klein-9B', model_source: 'Local',
    device: 'mps', upscale_enabled: false, upscale_model_path: '', fast_preview: true,
    mask_mode: 'Crop & Composite (Fast)', outpaint_align: 'center',
  })
  assert.deepEqual(p, {
    prompt: '', seed: 0, width: 1328, height: 784, steps: 20, guidance: 0, img_strength: 0.8,
    repeat_count: 2, num_frames: 25, fps: 24, model_choice: 'FLUX.2-klein-9B', model_source: 'Local',
    device: 'mps', upscale_enabled: false, fast_preview: true,
    mask_mode: 'Crop & Composite (Fast)', outpaint_align: 'center',
  })
})

test('missing, null, empty and wrong-typed fields are left out', () => {
  assert.deepEqual(workflowToParams({ mask_mode: '', seed: null, width: 'x', ref_slots: [] }), {})
})

test('seed override wins over the stored seed', () => {
  assert.equal(workflowToParams({ seed: -1 }, { seed: 812345 }).seed, 812345)
  assert.equal(workflowToParams({ seed: 7 }, { seed: null }).seed, 7)
})

test('lora_files keeps only entries with a path', () => {
  const p = workflowToParams({ lora_files: [{ path: '/a.safetensors', strength: 0.5 }, { strength: 1 }, null] })
  assert.deepEqual(p.lora_files, [{ path: '/a.safetensors', strength: 0.5 }])
})
