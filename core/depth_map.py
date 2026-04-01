"""Depth map generation using Depth Anything 3 (or compatible HuggingFace models).

Output convention: white = near, black = far (Blender/standard 3D convention).
DA3 predicts depth directly (larger = farther), so we invert before saving.
DA2 predicted disparity (larger = nearer) — no inversion was needed there.
"""
from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Module-level cache: repo_id -> loaded pipeline
_model_cache: dict[str, object] = {}


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(repo_id: str):
    """Load and cache a depth estimation pipeline by HuggingFace repo ID."""
    if repo_id in _model_cache:
        return _model_cache[repo_id]

    from transformers import pipeline as hf_pipeline

    device = _get_device()
    print(f"[depth_map] Loading {repo_id} on {device} …")
    pipe = hf_pipeline(
        task="depth-estimation",
        model=repo_id,
        device=device,
    )
    _model_cache[repo_id] = pipe
    print(f"[depth_map] {repo_id} loaded.")
    return pipe


def generate_depth_map(
    image_path: str,
    repo_id: str = "depth-anything/DA3MONO-LARGE",
    invert: bool = True,
) -> bytes:
    """Run depth estimation on an image file and return 16-bit grayscale PNG bytes.

    Args:
        image_path: Absolute path to source image.
        repo_id:    HuggingFace model repo ID.
        invert:     If True (default), invert depth so white=near, black=far.
                    Required for DA3 (direct depth). Not needed for DA2 (disparity).

    Returns:
        PNG bytes of a 16-bit grayscale depth map.
    """
    pipe = _load_model(repo_id)

    img = Image.open(image_path).convert("RGB")
    result = pipe(img)

    # result["depth"] is a PIL Image (grayscale) or numpy array depending on model
    depth = result["depth"]
    if not isinstance(depth, np.ndarray):
        depth = np.array(depth, dtype=np.float32)
    else:
        depth = depth.astype(np.float32)

    # Normalise to [0, 1]
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    # Invert: DA3 outputs depth (larger=farther) → flip so white=near
    if invert:
        depth = 1.0 - depth

    # Scale to 16-bit
    depth_16 = (depth * 65535).astype(np.uint16)

    # Save as PNG via Pillow
    out_img = Image.fromarray(depth_16, mode="I;16")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()
