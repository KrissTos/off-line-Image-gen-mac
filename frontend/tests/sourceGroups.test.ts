import { test } from 'node:test'
import assert from 'node:assert/strict'
import { groupSources, functionCounts, filterByFunction, familyLabel, parseOpenState } from '../src/sourceGroups.ts'
import type { ModelSource } from '../src/api.ts'

const src = (name: string, type: ModelSource['type'], over: Partial<ModelSource> = {}): ModelSource =>
  ({ id: name, name, url: `https://huggingface.co/o/${name}`, type, description: '', ...over })

test('categories come out as Models, LoRAs, Upscalers and empty ones are omitted', () => {
  const g = groupSources([src('u', 'upscaler'), src('m', 'base')])
  assert.deepEqual(g.map(c => [c.type, c.label, c.items.length]), [['base', 'Models', 1], ['upscaler', 'Upscalers', 1]])
})

test('LoRAs are split into family folders in a fixed order, unknown family last', () => {
  const g = groupSources([
    src('a', 'lora', { family: 'LTX-Video' }),
    src('b', 'lora', { family: 'klein-4B' }),
    src('c', 'lora', { family: 'klein-9B' }),
    src('d', 'lora'),                                   // custom LoRA without a family
    src('e', 'lora', { family: 'Z-Image' }),
    src('f', 'lora', { family: 'klein' }),
  ])
  const folders = g.find(c => c.type === 'lora')!.folders!
  assert.deepEqual(folders.map(f => f.key), ['klein-9B', 'klein-4B', 'klein', 'Z-Image', 'LTX-Video', 'other'])
  assert.deepEqual(folders.map(f => f.label), [
    'FLUX.2-klein 9B', 'FLUX.2-klein 4B', 'FLUX.2-klein (any size)', 'Z-Image', 'LTX-Video 0.9', 'Other'])
})

test('items inside a folder are alphabetical, case-insensitive', () => {
  const g = groupSources([src('beta', 'lora', { family: 'klein' }), src('Alpha', 'lora', { family: 'klein' }),
                          src('alpine', 'lora', { family: 'klein' })])
  assert.deepEqual(g[0].folders![0].items.map(s => s.name), ['Alpha', 'alpine', 'beta'])
})

test('only folders with items appear; non-LoRA categories have no folders', () => {
  const g = groupSources([src('a', 'lora', { family: 'Z-Image' }), src('m', 'base')])
  assert.deepEqual(g.find(c => c.type === 'lora')!.folders!.map(f => f.key), ['Z-Image'])
  assert.equal(g.find(c => c.type === 'base')!.folders, undefined)
})

test('familyLabel falls back to Other', () => {
  assert.equal(familyLabel(undefined), 'Other')
  assert.equal(familyLabel('nope'), 'Other')
})

test('functionCounts sorts by count then name and ignores missing functions', () => {
  const items = [src('a', 'lora', { function: 'style' }), src('b', 'lora', { function: 'edit' }),
                 src('c', 'lora', { function: 'style' }), src('d', 'lora')]
  assert.deepEqual(functionCounts(items), [{ fn: 'style', count: 2 }, { fn: 'edit', count: 1 }])
})

test('filterByFunction: null keeps everything', () => {
  const items = [src('a', 'lora', { function: 'style' }), src('b', 'lora', { function: 'edit' })]
  assert.equal(filterByFunction(items, null).length, 2)
  assert.deepEqual(filterByFunction(items, 'edit').map(s => s.name), ['b'])
})

test('parseOpenState survives garbage', () => {
  assert.deepEqual(parseOpenState(null), {})
  assert.deepEqual(parseOpenState('not json'), {})
  assert.deepEqual(parseOpenState('[1,2]'), {})
  assert.deepEqual(parseOpenState('{"lora":true,"x":"y"}'), { lora: true })
})
