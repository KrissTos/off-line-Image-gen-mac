import { test } from 'node:test'
import assert from 'node:assert/strict'
import { modelVariant, loraDisabledReason } from '../src/loraCompat.ts'

test('modelVariant reads the klein size from the display model choice', () => {
  assert.equal(modelVariant('FLUX.2-klein-4B (Int8)'), '4b')
  assert.equal(modelVariant('FLUX.2-klein-4B (4bit SDNQ - Low VRAM)'), '4b')
  assert.equal(modelVariant('FLUX.2-klein-9B (4bit SDNQ - Higher Quality)'), '9b')
  assert.equal(modelVariant('Z-Image Turbo (Full)'), null)
})

test('mismatched klein size is disabled with a reason', () => {
  assert.match(loraDisabledReason({ variant: '9b' }, '4b')!, /9B/)
  assert.match(loraDisabledReason({ variant: '4b' }, '9b')!, /4B/)
})

test('matching, unclassified or non-klein model stays enabled', () => {
  assert.equal(loraDisabledReason({ variant: '9b' }, '9b'), null)
  assert.equal(loraDisabledReason({ variant: null }, '4b'), null)
  assert.equal(loraDisabledReason({ variant: undefined }, '4b'), null)
  assert.equal(loraDisabledReason({ variant: '9b' }, null), null)
})
