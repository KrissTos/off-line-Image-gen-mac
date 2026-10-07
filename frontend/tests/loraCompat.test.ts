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

import { sortLoras, insertTrigger } from '../src/loraCompat.ts'

test('sortLoras orders by name, case-insensitive, without mutating the input', () => {
  const input = [{ name: 'realistic.safetensors' }, { name: 'bfs_b' }, { name: 'Klein-consistency' }, { name: 'Bfs_a' }]
  const out = sortLoras(input)
  assert.deepEqual(out.map(l => l.name), ['Bfs_a', 'bfs_b', 'Klein-consistency', 'realistic.safetensors'])
  assert.equal(input[0].name, 'realistic.safetensors')
})

test('insertTrigger puts the trigger first, once', () => {
  assert.equal(insertTrigger('', 'dmc_style'), 'dmc_style')
  assert.equal(insertTrigger('a cat on a roof', 'dmc_style'), 'dmc_style, a cat on a roof')
  assert.equal(insertTrigger('dmc_style, a cat', 'dmc_style'), 'dmc_style, a cat')
  assert.equal(insertTrigger('x DMC_STYLE y', 'dmc_style'), 'x DMC_STYLE y')
})

test('insertTrigger joins a sentence trigger with a space and ignores surrounding blanks', () => {
  assert.equal(insertTrigger('  keep the pose  ', 'Transform into dmc_style.'), 'Transform into dmc_style. keep the pose')
  assert.equal(insertTrigger('anything', ''), 'anything')
})

import { cleanLoraName, loraLabel } from '../src/loraCompat.ts'

test('cleanLoraName drops the extension and turns underscores into spaces', () => {
  assert.equal(cleanLoraName('pasta_IL_v3.safetensors'), 'pasta IL v3')
  assert.equal(cleanLoraName('Klein-consistency.safetensors'), 'Klein-consistency')
  assert.equal(cleanLoraName('x.SAFETENSORS'), 'x')
  assert.equal(cleanLoraName('_.safetensors'), '_.safetensors')
})

test('loraLabel shows the friendly name plus the klein size, always when known', () => {
  assert.equal(loraLabel({ name: 'a.safetensors', display: '70s Sci-Fi Movie', variant: '4b' }), '70s Sci-Fi Movie · 4B')
  assert.equal(loraLabel({ name: 'realistic.safetensors', variant: '9b' }), 'realistic · 9B')
  assert.equal(loraLabel({ name: 'pasta_IL_v3.safetensors', variant: null }), 'pasta IL v3')
  assert.equal(loraLabel({ name: 'z.safetensors' }), 'z')
})

test('sortLoras sorts by the shown name, not the file name', () => {
  const out = sortLoras([
    { name: 'zzz.safetensors', display: 'Alpha style' },
    { name: 'aaa.safetensors', display: 'Beta style' },
    { name: 'm_file.safetensors' },
  ])
  assert.deepEqual(out.map(l => l.name), ['zzz.safetensors', 'aaa.safetensors', 'm_file.safetensors'])
})
