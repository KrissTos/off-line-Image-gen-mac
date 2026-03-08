"""
Workflow utilities for ultra-fast-image-gen.

Supports two formats:
  - Native app  workflow.json  (saved/loaded by app.py)
  - ComfyUI     workflow.json  (node-graph format)

Usage:
    from workflow_utils import load_any_workflow, is_comfyui_workflow
    wf = load_any_workflow("path/to/workflow.json")
    # wf["prompt"], wf["height"], wf["steps"], ... always present
    # wf["_source"] == "native" | "comfyui"
    # wf["_unknown_nodes"] == list of custom/unrecognised class_type strings (comfyui only)
    # wf["_comfyui_ckpt_name"] == raw checkpoint filename from workflow (comfyui only)
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

# ── Model name fragments → app MODEL_CHOICES ────────────────────────────────
_COMFYUI_MODEL_MAP: list[tuple[str, str]] = [
    ("flux.2-klein-9b",   "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)"),
    ("flux.2-klein-4b",   "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)"),
    ("flux2-klein-9b",    "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)"),
    ("flux2-klein-4b",    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)"),
    ("flux.2-klein",      "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)"),
    ("flux2-klein",       "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)"),
    ("z-image-turbo",     "Z-Image Turbo (Full - LoRA support)"),
    ("zimage-turbo",      "Z-Image Turbo (Full - LoRA support)"),
    ("z_image_turbo",     "Z-Image Turbo (Full - LoRA support)"),
    ("ltx-video",         "LTX-Video  (txt2video · img2video with ref)"),
    ("ltxvideo",          "LTX-Video  (txt2video · img2video with ref)"),
]

# ── App MODEL_CHOICES → HuggingFace repo IDs (mirrors app.py's dict) ────────
_APP_MODEL_REPOS: dict[str, str] = {
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)":       "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
    "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)":  "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
    "FLUX.2-klein-4B (Int8)":                         "aydin99/FLUX.2-klein-4B-int8",
    "Z-Image Turbo (Quantized - Fast)":               "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
    "Z-Image Turbo (Full - LoRA support)":            "Tongyi-MAI/Z-Image-Turbo",
    "LTX-Video  (txt2video · img2video with ref)":    "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
}

# ── Standard built-in ComfyUI node types we recognise ───────────────────────
# Everything NOT in this set is considered a custom / third-party node.
_KNOWN_CORE_NODES: frozenset[str] = frozenset({
    # Samplers & schedulers
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
    "KSamplerSelect", "SamplerDPMPP2M", "SamplerDPMPP2SANCESTRAL",
    "SamplerDPMPP3MSDE", "SamplerEulerAncestral", "SamplerLMS",
    "BasicScheduler", "KarrasScheduler", "PolyexponentialScheduler",
    "SDTurboScheduler", "VPScheduler", "ExponentialScheduler",
    # Conditioning / CLIP
    "CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeSDXLRefiner",
    "CLIPTextEncodeFlux", "CLIPLoader", "DualCLIPLoader", "CLIPSetLastLayer",
    "ConditioningCombine", "ConditioningConcat", "ConditioningZeroOut",
    "ConditioningSetArea", "ConditioningSetMask", "ConditioningAverage",
    "ConditioningSetAreaPercentage", "ConditioningSetTimestepRange",
    # Loaders
    "CheckpointLoaderSimple", "CheckpointLoader", "unCLIPCheckpointLoader",
    "UNETLoader", "DiffusionModelLoader", "FluxLoader", "ModelLoader",
    "VAELoader", "CLIPVisionLoader", "StyleModelLoader",
    "LoraLoader", "LoraLoaderModelOnly",
    "ControlNetLoader", "ControlNetApply", "ControlNetApplyAdvanced",
    "ControlNetApplySD3",
    # Latent
    "EmptyLatentImage", "EmptySD3LatentImage",
    "EmptyHunyuanLatentVideo", "EmptyMochiLatentVideo",
    "LatentUpscale", "LatentUpscaleBy", "LatentCrop", "LatentFlip",
    "LatentRotate", "LatentComposite", "LatentBlend", "LatentFromBatch",
    "RepeatLatentBatch", "LatentBatch",
    # VAE
    "VAEDecode", "VAEEncode", "VAEDecodeTiled", "VAEEncodeTiled",
    "VAEEncodeForInpaint",
    # Image I/O
    "LoadImage", "LoadImageMask", "ImageLoader",
    "SaveImage", "PreviewImage",
    # Image ops
    "ImageScale", "ImageScaleBy", "ImageScaleToTotalPixels", "ImageCrop",
    "ImageInvert", "ImagePadForOutpaint", "ImageBlend", "ImageComposite",
    "ImageBatch", "RepeatImageBatch", "ImageFromBatch",
    "ImageFlip", "ImageRotate", "ImageToMask", "MaskToImage",
    "ImageColorToMask", "SolidMask", "CropMask",
    "FeatherMask", "GrowMask", "InvertMask", "MaskComposite",
    # Model ops
    "ModelMergeSimple", "ModelMergeBlocks", "CLIPMergeSimple",
    "UpscaleModelLoader", "ImageUpscaleWithModel",
    # Misc / utility
    "PrimitiveNode", "Note", "Reroute",
    "FreeU", "FreeU_V2",
    "unCLIPConditioning", "GLIGENLoader", "GLIGENTextBoxApply",
    "HyperTile", "PatchModelAddDownscale",
    "IPAdapterModelLoader", "IPAdapterApply",
})


# ── Detection ────────────────────────────────────────────────────────────────

def is_comfyui_workflow(data: dict) -> bool:
    """
    Return True if *data* looks like a ComfyUI node-graph workflow.

    ComfyUI format: keys are integer-string node IDs, values are dicts
    with at least a ``class_type`` key.
    """
    if not isinstance(data, dict) or len(data) == 0:
        return False
    sample = list(data.values())[:10]
    hits = sum(1 for v in sample if isinstance(v, dict) and "class_type" in v)
    return hits >= min(3, len(data))


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_input(data: dict, ref: Any) -> Optional[dict]:
    """
    Resolve a ComfyUI link reference ``["node_id", output_slot]``
    to the referenced node dict, or *None* if unresolvable.
    """
    if isinstance(ref, list) and len(ref) >= 2 and isinstance(ref[0], str):
        node = data.get(str(ref[0]))
        return node if isinstance(node, dict) else None
    return None


def _nodes_by_type(data: dict) -> dict[str, list[tuple[str, dict]]]:
    """Return a mapping of ``class_type → [(node_id, node), …]``."""
    by_type: dict[str, list[tuple[str, dict]]] = {}
    for nid, node in data.items():
        if isinstance(node, dict):
            ct = node.get("class_type", "")
            by_type.setdefault(ct, []).append((nid, node))
    return by_type


# ── ComfyUI parser ───────────────────────────────────────────────────────────

def parse_comfyui_workflow(data: dict) -> dict:
    """
    Parse a ComfyUI workflow JSON and return a dict in the app's
    workflow format (same keys as ``workflow.json`` saved by app.py).

    Unknown / unsupported parameters are set to sensible defaults.

    Extra keys set on the returned dict:
        _unknown_nodes      – sorted list of custom/unrecognised class_type strings
        _comfyui_ckpt_name  – raw checkpoint filename from the workflow (or "")
    """
    result: dict[str, Any] = {
        "prompt":             "",
        "height":             512,
        "width":              512,
        "steps":              20,
        "seed":               -1,
        "guidance":           7.0,
        "device":             "mps",
        "model_choice":       None,
        "model_source":       "Local",
        "lora_file":          "",
        "lora_strength":      1.0,
        "img_strength":       1.0,
        "repeat_count":       1,
        "upscale_enabled":    False,
        "upscale_model_path": "",
        "num_frames":         25,
        "fps":                24,
        "reference_images":   [],
        "_comfyui":           True,
        "_comfyui_ckpt_name": "",
    }

    bt = _nodes_by_type(data)

    # ── Sampler (steps / cfg / seed / denoise / positive prompt) ────────────
    for ct in ("KSampler", "KSamplerAdvanced", "SamplerCustom",
               "KSamplerSelect", "SamplerCustomAdvanced"):
        for nid, node in bt.get(ct, []):
            inp = node.get("inputs", {})
            if "steps"       in inp: result["steps"]        = int(inp["steps"])
            if "cfg"         in inp: result["guidance"]     = float(inp["cfg"])
            if "seed"        in inp: result["seed"]         = int(inp["seed"])
            if "noise_seed"  in inp: result["seed"]         = int(inp["noise_seed"])
            if "denoise"     in inp: result["img_strength"] = float(inp["denoise"])

            # Positive conditioning → resolve to CLIPTextEncode text
            pos_node = _resolve_input(data, inp.get("positive"))
            if pos_node:
                txt = pos_node.get("inputs", {}).get("text", "")
                if txt:
                    result["prompt"] = txt
            break  # use the first sampler found

    # ── Prompt fallback (any CLIPTextEncode with meaningful text) ────────────
    if not result["prompt"]:
        for nid, node in bt.get("CLIPTextEncode", []):
            txt = node.get("inputs", {}).get("text", "")
            if txt and len(txt.strip()) > 5:
                result["prompt"] = txt.strip()
                break

    # ── Latent dimensions ────────────────────────────────────────────────────
    for ct in ("EmptyLatentImage", "EmptySD3LatentImage",
               "EmptyHunyuanLatentVideo", "EmptyMochiLatentVideo"):
        for nid, node in bt.get(ct, []):
            inp = node.get("inputs", {})
            if "width"  in inp: result["width"]  = int(inp["width"])
            if "height" in inp: result["height"] = int(inp["height"])
            break

    # ── Model / checkpoint ───────────────────────────────────────────────────
    for ct in ("CheckpointLoaderSimple", "CheckpointLoader",
               "UNETLoader", "FluxLoader", "ModelLoader",
               "DiffusionModelLoader"):
        for nid, node in bt.get(ct, []):
            inp  = node.get("inputs", {})
            name = (inp.get("ckpt_name") or inp.get("unet_name")
                    or inp.get("model_name") or "")
            result["_comfyui_ckpt_name"] = name   # preserve raw name
            name_l = name.lower()
            for key, choice in _COMFYUI_MODEL_MAP:
                if key in name_l:
                    result["model_choice"] = choice
                    break
            break

    # ── LoRA ─────────────────────────────────────────────────────────────────
    for nid, node in bt.get("LoraLoader", []):
        inp   = node.get("inputs", {})
        lname = inp.get("lora_name", "")
        if lname:
            result["lora_file"]     = lname
            result["lora_strength"] = float(inp.get("strength_model", 1.0))
        break

    # ── Reference images (LoadImage nodes) ───────────────────────────────────
    for ct in ("LoadImage", "LoadImageMask", "ImageLoader"):
        for nid, node in bt.get(ct, []):
            fname = node.get("inputs", {}).get("image", "")
            if fname:
                result["reference_images"].append(fname)
        if result["reference_images"]:
            break

    # ── Video parameters (LTX / Mochi / etc.) ───────────────────────────────
    for ct in ("EmptyHunyuanLatentVideo", "EmptyMochiLatentVideo"):
        for nid, node in bt.get(ct, []):
            inp = node.get("inputs", {})
            if "num_frames" in inp: result["num_frames"] = int(inp["num_frames"])
            break

    # ── Collect unknown / custom node types ──────────────────────────────────
    all_types = {ct for ct in bt.keys() if ct}   # skip empty-string key
    result["_unknown_nodes"] = sorted(all_types - _KNOWN_CORE_NODES)

    return result


# ── Local model availability ─────────────────────────────────────────────────

def get_locally_available_models(models_dir: str) -> list[str]:
    """
    Return a list of MODEL_CHOICES strings whose HuggingFace cache
    directories are present under *models_dir*.

    The HuggingFace cache directory naming convention is:
        models--{org}--{repo_name}
    e.g. "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic"
         → "models--Disty0--FLUX.2-klein-4B-SDNQ-4bit-dynamic"
    """
    available: list[str] = []
    for choice, repo_id in _APP_MODEL_REPOS.items():
        cache_name = "models--" + repo_id.replace("/", "--")
        cache_path = os.path.join(models_dir, cache_name)
        if os.path.isdir(cache_path):
            available.append(choice)
    return available


# ── Public loader ────────────────────────────────────────────────────────────

def load_any_workflow(path: str) -> dict:
    """
    Load a workflow from *path*, auto-detecting the format.

    Returns a dict in the native app workflow format with two extra keys:
        _source  – "native" | "comfyui"
        _file    – absolute path of the loaded file

    ComfyUI-format dicts also carry:
        _unknown_nodes      – sorted list of custom/unrecognised class_type strings
        _comfyui_ckpt_name  – raw checkpoint filename (or "")

    Raises:
        FileNotFoundError  – if the file does not exist
        json.JSONDecodeError – if the file is not valid JSON
        ValueError         – if the file cannot be parsed as either format
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workflow file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if is_comfyui_workflow(data):
        parsed            = parse_comfyui_workflow(data)
        parsed["_source"] = "comfyui"
        parsed["_file"]   = path
        return parsed
    else:
        # Native app format — just add meta keys
        data["_source"] = "native"
        data["_file"]   = path
        return data


# ── Convenience summary ───────────────────────────────────────────────────────

def workflow_summary(wf: dict) -> str:
    """Return a one-line human-readable summary of a loaded workflow dict."""
    src    = wf.get("_source", "?")
    p      = (wf.get("prompt") or "")[:60].strip()
    model  = wf.get("model_choice") or "?"
    dims   = f"{wf.get('width', '?')}×{wf.get('height', '?')}"
    steps  = wf.get("steps", "?")
    seed   = wf.get("seed", -1)
    seed_s = "random" if seed == -1 else str(seed)
    unknown = wf.get("_unknown_nodes", [])
    suffix = f" | {len(unknown)} custom nodes" if unknown else ""
    return (f"[{src}] prompt='{p}' | model={model} | "
            f"{dims} | steps={steps} | seed={seed_s}{suffix}")
