// Pure mask operations for the mask editor. A mask is one byte per image pixel:
// 0 = keep, 255 = regenerate (same meaning as the mask PNGs the backend reads).
// No DOM here — tested with `npm test` (node --test).

export type Mask = { w: number; h: number; data: Uint8Array }
export type Pt = { x: number; y: number }

export function createMask(w: number, h: number): Mask {
  return { w, h, data: new Uint8Array(w * h) }
}

export function cloneMask(m: Mask): Mask {
  return { w: m.w, h: m.h, data: m.data.slice() }
}

function combine(a: Mask, b: Mask, f: (x: number, y: number) => number): Mask {
  const out = createMask(a.w, a.h)
  for (let i = 0; i < a.data.length; i++) out.data[i] = f(a.data[i], b.data[i])
  return out
}

export const union    = (a: Mask, b: Mask) => combine(a, b, (x, y) => (x || y ? 255 : 0))
export const subtract = (a: Mask, b: Mask) => combine(a, b, (x, y) => (x && !y ? 255 : 0))

export function invert(m: Mask): Mask {
  const out = createMask(m.w, m.h)
  for (let i = 0; i < m.data.length; i++) out.data[i] = m.data[i] ? 0 : 255
  return out
}

function stamp(m: Mask, cx: number, cy: number, r: number, value: number) {
  const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(m.w - 1, Math.ceil(cx + r))
  const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(m.h - 1, Math.ceil(cy + r))
  const r2 = r * r
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx
      if (dx * dx + dy * dy <= r2) m.data[y * m.w + x] = value
    }
  }
}

/** Circles of radius r along the polyline (in place). */
export function paintStroke(m: Mask, pts: Pt[], r: number, value: 0 | 255): void {
  if (pts.length === 0) return
  stamp(m, pts[0].x, pts[0].y, r, value)
  const step = Math.max(0.5, r / 2)
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i]
    const len = Math.hypot(b.x - a.x, b.y - a.y)
    const n = Math.ceil(len / step)
    for (let k = 1; k <= n; k++) {
      const t = k / n
      stamp(m, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, r, value)
    }
  }
}

/** Even-odd scanline fill, sampling pixel centres (in place). */
export function fillPolygon(m: Mask, pts: Pt[], value: 0 | 255): void {
  if (pts.length < 3) return
  const ys = pts.map(p => p.y)
  const yMin = Math.max(0, Math.floor(Math.min(...ys)))
  const yMax = Math.min(m.h - 1, Math.ceil(Math.max(...ys)))
  for (let y = yMin; y <= yMax; y++) {
    const cy = y + 0.5
    const xs: number[] = []
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length]
      if ((a.y <= cy && b.y > cy) || (b.y <= cy && a.y > cy)) {
        xs.push(a.x + ((cy - a.y) / (b.y - a.y)) * (b.x - a.x))
      }
    }
    xs.sort((p, q) => p - q)
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const from = Math.max(0, Math.ceil(xs[k] - 0.5))
      const to = Math.min(m.w - 1, Math.floor(xs[k + 1] - 0.5))
      for (let x = from; x <= to; x++) m.data[y * m.w + x] = value
    }
  }
}

// Squared Euclidean distance transform (Felzenszwalb & Huttenlocher), O(pixels).
const INF = 1e20
function edt1d(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array) {
  let k = 0
  v[0] = 0; z[0] = -INF; z[1] = INF
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    while (s <= z[k]) {
      k--
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    }
    k++; v[k] = q; z[k] = s; z[k + 1] = INF
  }
  k = 0
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++
    const dq = q - v[k]
    d[q] = dq * dq + f[v[k]]
  }
}

function sqDistToSet(m: Mask): Float64Array {
  const { w, h } = m
  const n = Math.max(w, h)
  const grid = new Float64Array(w * h)
  for (let i = 0; i < grid.length; i++) grid[i] = m.data[i] ? 0 : INF
  const f = new Float64Array(n), d = new Float64Array(n)
  const v = new Int32Array(n), z = new Float64Array(n + 1)
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = grid[y * w + x]
    edt1d(f, h, d, v, z)
    for (let y = 0; y < h; y++) grid[y * w + x] = d[y]
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = grid[y * w + x]
    edt1d(f, w, d, v, z)
    for (let x = 0; x < w; x++) grid[y * w + x] = d[x]
  }
  return grid
}

/** Every pixel within px (Euclidean) of the mask becomes 255. */
export function grow(m: Mask, px: number): Mask {
  const d = sqDistToSet(m)
  const out = createMask(m.w, m.h)
  const r2 = px * px
  for (let i = 0; i < d.length; i++) out.data[i] = d[i] <= r2 ? 255 : 0
  return out
}

/** Removes every mask pixel within px of the outside. */
export function shrink(m: Mask, px: number): Mask {
  return invert(grow(invert(m), px))
}

export function coverage(m: Mask): number {
  let n = 0
  for (let i = 0; i < m.data.length; i++) if (m.data[i]) n++
  return m.data.length ? n / m.data.length : 0
}

export function iou(a: Mask, b: Mask): number {
  let inter = 0, uni = 0
  for (let i = 0; i < a.data.length; i++) {
    const x = a.data[i] !== 0, y = b.data[i] !== 0
    if (x && y) inter++
    if (x || y) uni++
  }
  return uni === 0 ? 1 : inter / uni
}

/** Any RGBA mask image (e.g. an older upload of another size) → Mask of w×h. */
export function fromRgba(rgba: Uint8ClampedArray, srcW: number, srcH: number, w: number, h: number): Mask {
  const m = createMask(w, h)
  for (let y = 0; y < h; y++) {
    const sy = Math.min(srcH - 1, Math.floor(((y + 0.5) * srcH) / h))
    for (let x = 0; x < w; x++) {
      const sx = Math.min(srcW - 1, Math.floor(((x + 0.5) * srcW) / w))
      m.data[y * w + x] = rgba[(sy * srcW + sx) * 4] >= 128 ? 255 : 0
    }
  }
  return m
}

export function toRgba(m: Mask): Uint8ClampedArray {
  const out = new Uint8ClampedArray(m.w * m.h * 4)
  for (let i = 0; i < m.data.length; i++) {
    const v = m.data[i]
    out[i * 4] = v; out[i * 4 + 1] = v; out[i * 4 + 2] = v; out[i * 4 + 3] = 255
  }
  return out
}
