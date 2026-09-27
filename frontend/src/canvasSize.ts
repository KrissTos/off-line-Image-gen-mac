// Output size that matches a ref image's aspect at each model family's best
// resolution. Matching aspect = no padding/outpaint and no stretch; the pixel
// budget keeps VRAM/time sane (a 4000×3000 photo → 1184×880, not 4000×3008).
//   flux / zimage: ~1 MP (training area; FLUX.2 also caps refs at 1 MP), dims /16
//   ltx:           768×512 budget, dims /32 (latent stride)

export type SizeFamily = 'flux' | 'zimage' | 'ltx'

const RULES: Record<SizeFamily, { budget: number; snap: number }> = {
  flux:   { budget: 1024 * 1024, snap: 16 },
  zimage: { budget: 1024 * 1024, snap: 16 },
  ltx:    { budget: 768 * 512,   snap: 32 },
}
const MIN_SIDE = 256

export function sizeFamily(model: string): SizeFamily {
  if (model.includes('LTX'))     return 'ltx'
  if (model.includes('Z-Image')) return 'zimage'
  return 'flux'
}

export function canvasForRef(w: number, h: number, family: SizeFamily) {
  const { budget, snap } = RULES[family]
  const ratio = w / h
  const s = (n: number) => Math.max(MIN_SIDE, Math.round(n / snap) * snap)
  return { w: s(Math.sqrt(budget * ratio)), h: s(Math.sqrt(budget / ratio)) }
}
