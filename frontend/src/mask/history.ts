import { cloneMask, type Mask } from './maskOps.ts'

// Undo/redo snapshots of the whole mask. Depth is capped by count AND bytes so a
// 12 MP photo keeps ~25 steps instead of 30 × 12 MB.
export class MaskHistory {
  private undoStack: Mask[] = []
  private redoStack: Mask[] = []
  private maxDepth: number
  private maxBytes: number

  constructor(maxDepth = 30, maxBytes = 300_000_000) {
    this.maxDepth = maxDepth
    this.maxBytes = maxBytes
  }

  private limit(bytesPer: number) {
    return Math.max(1, Math.min(this.maxDepth, Math.floor(this.maxBytes / Math.max(1, bytesPer))))
  }

  /** Call with the mask as it is BEFORE an edit. */
  push(m: Mask) {
    this.undoStack.push(cloneMask(m))
    const lim = this.limit(m.data.length)
    while (this.undoStack.length > lim) this.undoStack.shift()
    this.redoStack = []
  }

  undo(current: Mask): Mask | null {
    const prev = this.undoStack.pop()
    if (!prev) return null
    this.redoStack.push(cloneMask(current))
    return prev
  }

  redo(current: Mask): Mask | null {
    const next = this.redoStack.pop()
    if (!next) return null
    this.undoStack.push(cloneMask(current))
    return next
  }

  clear() { this.undoStack = []; this.redoStack = [] }
  get canUndo() { return this.undoStack.length > 0 }
  get canRedo() { return this.redoStack.length > 0 }
}
