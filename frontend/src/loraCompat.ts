// FLUX.2-klein LoRAs are size-specific (4B hidden 3072, 9B hidden 4096): a LoRA trained for
// one size cannot load into the other. The server reports each LoRA's `variant`; the UI greys
// out mismatches instead of letting generation fail.

export type KleinVariant = '4b' | '9b'

/** Klein size of the selected display model (`model_choice`), or null for non-klein models. */
export function modelVariant(modelChoice: string): KleinVariant | null {
  if (!modelChoice.startsWith('FLUX')) return null
  if (modelChoice.includes('9B')) return '9b'
  if (modelChoice.includes('4B')) return '4b'
  return null
}

/** Why this LoRA can't be used with the model, or null when it can (unclassified = allowed). */
export function loraDisabledReason(
  lora: { variant?: string | null },
  model: KleinVariant | null,
): string | null {
  if (!model || !lora.variant || lora.variant === model) return null
  return `${lora.variant.toUpperCase()} LoRA — needs klein-${lora.variant.toUpperCase()}`
}

/** LoRA lists are shown alphabetically (case-insensitive), whatever order the server sends. */
export function sortLoras<T extends { name: string }>(loras: T[]): T[] {
  return [...loras].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }))
}

/** Prepend a LoRA trigger to the prompt, unless it is already there (case-insensitive). */
export function insertTrigger(prompt: string, trigger: string): string {
  const t = trigger.trim()
  const p = prompt.trim()
  if (!t) return prompt
  if (!p) return t
  if (p.toLowerCase().includes(t.toLowerCase())) return prompt
  return `${t}${/[.!?:]$/.test(t) ? ' ' : ', '}${p}`
}
