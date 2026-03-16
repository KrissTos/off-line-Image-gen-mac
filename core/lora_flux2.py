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


def check_lora_compatibility(path: str) -> None:
    """
    Validate that a LoRA file is compatible with FLUX.2-klein before saving.
    Reads only the safetensors header (no tensor data loaded).

    Raises RuntimeError with a user-facing message if incompatible.
    """
    from safetensors import safe_open
    import re

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception as e:
        raise RuntimeError(f"Could not read LoRA file: {e}")

    if not keys:
        raise RuntimeError("LoRA file appears to be empty.")

    # Normalise all key prefixes to bare block names for uniform checking
    # Handles: diffusion_model.*, base_model.model.diffusion_model.*, transformer.*
    def _normalise(k: str) -> str:
        k = re.sub(r'^base_model\.model\.', '', k)
        k = re.sub(r'^diffusion_model\.', '', k)
        k = re.sub(r'^transformer\.single_transformer_blocks\.', 'single_blocks.', k)
        k = re.sub(r'^transformer\.transformer_blocks\.', 'double_blocks.', k)
        return k

    normalised = [_normalise(k) for k in keys]

    # Detect FLUX keys at all
    flux_keys = [k for k in normalised if k.startswith(('single_blocks.', 'double_blocks.'))]
    if not flux_keys:
        raise RuntimeError(
            "No FLUX LoRA keys found — this may be a Stable Diffusion or other format LoRA."
        )

    # Check block index bounds for FLUX.2-klein
    single_re = re.compile(r'^single_blocks\.(\d+)\.')
    double_re = re.compile(r'^double_blocks\.(\d+)\.')

    for k in normalised:
        m = single_re.match(k)
        if m and int(m.group(1)) >= 20:
            raise RuntimeError(
                f"LoRA not compatible with FLUX.2-klein — trained for a larger model "
                f"(found single_blocks.{m.group(1)}, klein has 20). "
                f"Use a LoRA trained for FLUX.2-klein 4B or 9B."
            )
        m = double_re.match(k)
        if m and int(m.group(1)) >= 19:
            raise RuntimeError(
                f"LoRA not compatible with FLUX.2-klein — trained for a larger model "
                f"(found double_blocks.{m.group(1)}, klein has 19). "
                f"Use a LoRA trained for FLUX.2-klein 4B or 9B."
            )


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
        if "No LoRA keys" in err_str or isinstance(e, KeyError):
            raise RuntimeError(
                "LoRA not compatible with FLUX.2-klein. "
                "Try a LoRA trained for FLUX.2-klein or standard FLUX.1. "
                f"(Detail: {err_str[:120]})"
            )
        raise

    lora_name = lora_path.split("/")[-1]
    return f"Loaded LoRA: {lora_name} (strength {strength:.2f})"


def load_loras(pipe, loras: list) -> str:
    """Load multiple LoRAs into Flux2KleinPipeline.
    loras = [{path: str, strength: float}, ...]
    """
    import os
    from safetensors.torch import load_file

    # Unload previous adapters
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    adapter_names   = []
    adapter_weights = []

    try:
        for i, lora in enumerate(loras):
            lora_path = lora["path"]
            strength  = float(lora.get("strength", 1.0))
            adapter_name = f"lora_{i}"

            state_dict = load_file(lora_path)

            # Pre-process PEFT/fal format
            if any(k.startswith("base_model.model.") for k in state_dict):
                state_dict = {
                    k.replace("base_model.model.", "diffusion_model."): v
                    for k, v in state_dict.items()
                }

            try:
                pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
            except Exception as e:
                err_str = str(e)
                if "No LoRA keys" in err_str or isinstance(e, KeyError):
                    raise RuntimeError(
                        f"LoRA '{os.path.basename(lora_path)}' not compatible with FLUX.2-klein. "
                        f"(Detail: {err_str[:120]})"
                    )
                raise

            adapter_names.append(adapter_name)
            adapter_weights.append(strength)

    except Exception:
        # Partial load — clean up any adapters already loaded so pipeline stays in known state
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        raise

    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    names = [os.path.basename(l["path"]) for l in loras]
    return f"Loaded {len(loras)} LoRA(s): {', '.join(names)}"


def unload_lora(pipe) -> str:
    """Unload LoRA from pipeline. Safe to call even if none loaded."""
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    return "LoRA unloaded"
