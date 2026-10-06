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

import re


# ── BFL-native FLUX.2 LoRA → diffusers ─────────────────────────────────────────
# diffusers' _convert_non_diffusers_flux2_lora_to_diffusers hardcodes FLUX.2-dev
# (8 double / 48 single blocks), requires MLP keys and rejects embedder/modulation
# keys — a klein LoRA (5/20 or 8/24 blocks) raises KeyError. Convert ourselves,
# taking block counts from the file.

_BFL_PREFIXES = ("base_model.model.diffusion_model.", "base_model.model.", "diffusion_model.")

_BFL_EXTRA = {
    "img_in":                            "x_embedder",
    "txt_in":                            "context_embedder",
    "time_in.in_layer":                  "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer":                 "time_guidance_embed.timestep_embedder.linear_2",
    "double_stream_modulation_img.lin":  "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin":  "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin":      "single_stream_modulation.linear",
    "final_layer.linear":                "proj_out",
    "final_layer.adaLN_modulation.1":    "norm_out.linear",
}

_BFL_DOUBLE = {
    "img_attn.proj": "attn.to_out.0",
    "txt_attn.proj": "attn.to_add_out",
    "img_mlp.0":     "ff.linear_in",
    "img_mlp.2":     "ff.linear_out",
    "txt_mlp.0":     "ff_context.linear_in",
    "txt_mlp.2":     "ff_context.linear_out",
}


def _strip_bfl_prefix(k: str) -> str:
    for p in _BFL_PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def is_bfl_flux2_lora(state_dict: dict) -> bool:
    """True for BFL-native keys (double_blocks.N.img_attn.qkv / single_blocks.N.linear1)."""
    return any(_strip_bfl_prefix(k).startswith(("double_blocks.", "single_blocks."))
               for k in state_dict)


def convert_bfl_flux2_lora(state_dict: dict) -> dict:
    """Convert a BFL-native FLUX.2 LoRA (lora_A/lora_B) to diffusers `transformer.*` keys.
    Fused qkv: lora_A is shared by q/k/v, lora_B is split in 3. Raises ValueError on
    any key it can't place, so a LoRA is never half-applied silently."""
    import re
    import torch

    sd = {_strip_bfl_prefix(k): v for k, v in state_dict.items()}
    out: dict = {}
    unknown = []
    for key, w in sd.items():
        m = re.match(r"^(.*)\.(lora_A|lora_B)\.weight$", key)
        if not m:
            unknown.append(key)
            continue
        mod, ab = m.groups()

        if (d := re.match(r"^double_blocks\.(\d+)\.(img_attn|txt_attn)\.qkv$", mod)):
            n, attn = d.groups()
            names = ("to_q", "to_k", "to_v") if attn == "img_attn" else ("add_q_proj", "add_k_proj", "add_v_proj")
            parts = [w] * 3 if ab == "lora_A" else list(torch.chunk(w, 3, dim=0))
            for name, part in zip(names, parts):
                out[f"transformer.transformer_blocks.{n}.attn.{name}.{ab}.weight"] = part
        elif (d := re.match(r"^double_blocks\.(\d+)\.(.+)$", mod)) and d.group(2) in _BFL_DOUBLE:
            out[f"transformer.transformer_blocks.{d.group(1)}.{_BFL_DOUBLE[d.group(2)]}.{ab}.weight"] = w
        elif (s_ := re.match(r"^single_blocks\.(\d+)\.linear([12])$", mod)):
            n, lin = s_.groups()
            tgt = "to_qkv_mlp_proj" if lin == "1" else "to_out"
            out[f"transformer.single_transformer_blocks.{n}.attn.{tgt}.{ab}.weight"] = w
        elif mod in _BFL_EXTRA:
            out[f"transformer.{_BFL_EXTRA[mod]}.{ab}.weight"] = w
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(f"Unmapped FLUX.2 LoRA keys: {sorted(unknown)[:5]}")
    return out


def _prepare_state_dict(state_dict: dict) -> dict:
    """BFL-native → our converter; PEFT/fal prefixed diffusers-style → prefix swap."""
    if is_bfl_flux2_lora(state_dict):
        return convert_bfl_flux2_lora(state_dict)
    if any(k.startswith("base_model.model.") for k in state_dict):
        return {k.replace("base_model.model.", "diffusion_model."): v for k, v in state_dict.items()}
    return state_dict


# klein-4B: 5 double / 20 single, hidden 3072. klein-9B: 8 double / 24 single, hidden 4096.
# FLUX.2-dev (8 / 48, hidden 6144) is the larger model a LoRA must not be trained for.
_KLEIN_MAX_SINGLE = 24
_KLEIN_MAX_DOUBLE = 8
_VARIANT_BY_HIDDEN = {3072: "4b", 4096: "9b"}
# Modules whose lora_A input / lora_B output dim equals the transformer hidden size.
_HIDDEN_IN_MODULES = re.compile(
    r"(?:^|\.)(linear1|to_qkv_mlp_proj|qkv|to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)$")
_HIDDEN_OUT_MODULES = re.compile(r"(?:^|\.)(linear2|to_out|to_out\.0|proj|to_add_out)$")


def lora_variant(path: str):
    """'4b' | '9b' | None for a FLUX.2-klein LoRA file, from tensor shapes in the header only.
    None = hidden size unreadable or not a klein size (don't block it, just can't classify)."""
    from safetensors import safe_open

    dims = set()
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                m = re.match(r"^(.*)\.(lora_A|lora_B|lora_down|lora_up)(?:\.[^.]+)?\.weight$", key)
                if not m:
                    continue
                mod, kind = m.groups()
                mod = re.sub(r"^.*(?:single_blocks|double_blocks|transformer_blocks|"
                             r"single_transformer_blocks)\.\d+\.", "", mod)
                shape = f.get_slice(key).get_shape()
                if kind in ("lora_A", "lora_down") and _HIDDEN_IN_MODULES.search(mod):
                    dims.add(shape[1])
                elif kind in ("lora_B", "lora_up") and _HIDDEN_OUT_MODULES.search(mod):
                    dims.add(shape[0])
    except Exception:
        return None
    variants = {_VARIANT_BY_HIDDEN.get(d) for d in dims}
    return variants.pop() if len(variants) == 1 else None


def pipe_variant(pipe):
    """'4b' | '9b' | None for a loaded Flux2KleinPipeline (hidden = heads * head_dim)."""
    try:
        cfg = pipe.transformer.config
        return _VARIANT_BY_HIDDEN.get(cfg.num_attention_heads * cfg.attention_head_dim)
    except AttributeError:
        return None  # unknown pipeline shape: skip the guard, loading still validates


def assert_lora_matches_model(pipe, lora_path: str) -> None:
    """Raise a one-line RuntimeError when a klein LoRA was trained for the other klein size.
    Unclassifiable LoRAs (variant None) pass through; load itself will still validate."""
    import os

    lv, mv = lora_variant(lora_path), pipe_variant(pipe)
    if lv and mv and lv != mv:
        raise RuntimeError(
            f"LoRA '{os.path.basename(lora_path)}' is for FLUX.2-klein-{lv.upper()} but the "
            f"loaded model is klein-{mv.upper()} — switch model or pick a {mv.upper()} LoRA.")


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
        if m and int(m.group(1)) >= _KLEIN_MAX_SINGLE:
            raise RuntimeError(
                f"LoRA not compatible with FLUX.2-klein — trained for a larger model "
                f"(found single_blocks.{m.group(1)}, klein has at most {_KLEIN_MAX_SINGLE}). "
                f"Use a LoRA trained for FLUX.2-klein 4B or 9B."
            )
        m = double_re.match(k)
        if m and int(m.group(1)) >= _KLEIN_MAX_DOUBLE:
            raise RuntimeError(
                f"LoRA not compatible with FLUX.2-klein — trained for a larger model "
                f"(found double_blocks.{m.group(1)}, klein has at most {_KLEIN_MAX_DOUBLE}). "
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

    assert_lora_matches_model(pipe, lora_path)
    state_dict = load_file(lora_path)

    try:
        state_dict = _prepare_state_dict(state_dict)
    except ValueError as e:
        raise RuntimeError(f"LoRA '{lora_path.split('/')[-1]}' not compatible with FLUX.2-klein: {e}")

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
            assert_lora_matches_model(pipe, lora_path)

            state_dict = load_file(lora_path)

            try:
                state_dict = _prepare_state_dict(state_dict)
            except ValueError as e:
                raise RuntimeError(
                    f"LoRA '{os.path.basename(lora_path)}' not compatible with FLUX.2-klein: {e}")

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
