# core/lora_flux2.py
"""
LoRA loading for Flux2KleinPipeline.

The upgraded diffusers (post-Feb 2026) handles ai-toolkit and diffusers-native
LoRA formats automatically. This module adds:
  - PEFT/fal format pre-processing (base_model.model. prefix strip)
  - Friendly error messages when LoRA is truly incompatible
  - Unload helper
"""
from __future__ import annotations


def load_lora(pipe, lora_path: str, strength: float) -> str:
    """
    Load a LoRA into Flux2KleinPipeline, handling all known key formats:
      - diffusers-native  (transformer. prefix)
      - ai-toolkit        (diffusion_model. prefix + lora_A/B or lora_down/up)
      - PEFT/fal trainer  (base_model.model.diffusion_model. prefix)
      - CivitAI           (any of the above depending on trainer used)

    Returns a status string for display in the UI.
    Raises RuntimeError with a user-friendly message if incompatible.
    """
    from safetensors.torch import load_file

    state_dict = load_file(lora_path)

    # Pre-process PEFT/fal format: strip base_model.model. prefix
    # (diffusers upgraded main handles the rest automatically)
    if any(k.startswith("base_model.model.") for k in state_dict):
        state_dict = {
            k.replace("base_model.model.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    # Unload any existing LoRA first
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    try:
        pipe.load_lora_weights(state_dict, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[strength])
    except Exception as e:
        err_str = str(e)
        if "No LoRA keys" in err_str or "KeyError" in err_str:
            raise RuntimeError(
                "LoRA not compatible with FLUX.2-klein. "
                "Try a LoRA trained for FLUX.2-klein or standard FLUX.1. "
                f"(Detail: {err_str[:120]})"
            )
        raise

    lora_name = lora_path.split("/")[-1]
    return f"Loaded LoRA: {lora_name} (strength {strength:.2f})"


def unload_lora(pipe) -> str:
    """Unload LoRA from pipeline. Safe to call even if none loaded."""
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    return "LoRA unloaded"
