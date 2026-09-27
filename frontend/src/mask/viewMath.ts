// Image ↔ screen mapping for the mask editor (CSS pixels; the canvas multiplies by
// devicePixelRatio when drawing). screen = image * scale + t
export type View = { scale: number; tx: number; ty: number }

export function fitView(imgW: number, imgH: number, boxW: number, boxH: number, margin = 0.96): View {
  const scale = Math.min(boxW / imgW, boxH / imgH) * margin
  return { scale, tx: (boxW - imgW * scale) / 2, ty: (boxH - imgH * scale) / 2 }
}

export function zoomAt(v: View, sx: number, sy: number, factor: number, minScale = 0.1, maxScale = 8): View {
  const scale = Math.min(maxScale, Math.max(minScale, v.scale * factor))
  const k = scale / v.scale
  return { scale, tx: sx - (sx - v.tx) * k, ty: sy - (sy - v.ty) * k }
}

export function screenToImage(v: View, sx: number, sy: number) {
  return { x: (sx - v.tx) / v.scale, y: (sy - v.ty) / v.scale }
}

export function imageToScreen(v: View, x: number, y: number) {
  return { x: x * v.scale + v.tx, y: y * v.scale + v.ty }
}

export function inImage(p: { x: number; y: number }, w: number, h: number) {
  return p.x >= 0 && p.y >= 0 && p.x < w && p.y < h
}
