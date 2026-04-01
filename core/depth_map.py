"""Depth map generation — Depth Anything 3 (DA3) or Depth Anything 2 (DA2).

Output convention: white = near, black = far (Blender/standard 3D convention).
DA3 predicts depth directly (larger = farther) → invert before saving.
DA2 predicts disparity (larger = nearer) → no inversion needed.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

# Module-level cache: repo_id -> loaded model
_model_cache: dict[str, object] = {}

# Local model directory for DA3 (user-downloaded weights)
_DA3_LOCAL_DIR = Path(__file__).parent.parent / "models" / "da3mono-large"


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_da3(repo_id: str) -> bool:
    return "DA3" in repo_id or "da3" in repo_id.lower()


def _load_da3(repo_id: str):
    """Load DA3 model from local dir (preferred) or HF Hub."""
    # Mock the optional 3DGS/video export deps that we don't need
    for mod in [
        "depth_anything_3.utils.export",
        "depth_anything_3.utils.export.gs",
        "depth_anything_3.utils.pose_align",
        "evo",
        "evo.core",
        "evo.core.trajectory",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    from depth_anything_3.api import DepthAnything3

    device = _get_device()
    print(f"[depth_map] Loading DA3 on {device} …")

    # Use local dir if it has the weights, otherwise fall back to HF Hub
    model_source = str(_DA3_LOCAL_DIR) if (_DA3_LOCAL_DIR / "model.safetensors").exists() else repo_id
    print(f"[depth_map] Source: {model_source}")

    model = DepthAnything3.from_pretrained(model_source)
    model = model.to(device)
    model.eval()
    print(f"[depth_map] DA3 loaded.")
    return model


def _load_da2(repo_id: str):
    """Load DA2 model via transformers pipeline."""
    from transformers import pipeline as hf_pipeline

    device = _get_device()
    print(f"[depth_map] Loading DA2 {repo_id} on {device} …")
    try:
        pipe = hf_pipeline(
            task="depth-estimation",
            model=repo_id,
            device=device,
            trust_remote_code=True,
        )
    except Exception as e:
        if device != "cpu":
            print(f"[depth_map] {device} failed ({e}), retrying on CPU …")
            pipe = hf_pipeline(
                task="depth-estimation",
                model=repo_id,
                device="cpu",
                trust_remote_code=True,
            )
        else:
            raise
    print(f"[depth_map] DA2 loaded.")
    return pipe


def _load_model(repo_id: str):
    if repo_id not in _model_cache:
        if _is_da3(repo_id):
            _model_cache[repo_id] = _load_da3(repo_id)
        else:
            _model_cache[repo_id] = _load_da2(repo_id)
    return _model_cache[repo_id]


def generate_depth_map(
    image_path: str,
    repo_id: str = "istiakiat/DA3MONO-LARGE",
    invert: bool = True,
) -> bytes:
    """Run depth estimation and return 16-bit grayscale PNG bytes.

    Args:
        image_path: Absolute path to source image.
        repo_id:    HuggingFace repo ID (or local path hint for DA3).
        invert:     Flip depth so white=near. Required for DA3; not for DA2.

    Returns:
        PNG bytes of a 16-bit grayscale depth map.
    """
    model = _load_model(repo_id)

    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size

    if _is_da3(repo_id):
        # DA3: inference() accepts file paths directly; returns Prediction with .depth [N,H,W]
        prediction = model.inference([image_path])
        depth = prediction.depth[0]  # (H, W) float32, larger = farther
        if not isinstance(depth, np.ndarray):
            depth = np.array(depth, dtype=np.float32)
        else:
            depth = depth.astype(np.float32)
    else:
        # DA2: transformers pipeline
        result = model(orig)
        depth = result["depth"]
        if not isinstance(depth, np.ndarray):
            depth = np.array(depth, dtype=np.float32)
        else:
            depth = depth.astype(np.float32)

    # Restore original resolution if model resized the input
    if depth.shape != (orig_h, orig_w):
        depth_img = Image.fromarray(depth).resize((orig_w, orig_h), Image.LANCZOS)
        depth = np.array(depth_img, dtype=np.float32)

    # Normalise to [0, 1]
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    if invert:
        depth = 1.0 - depth

    # Scale to 16-bit and encode as PNG
    depth_16 = (depth * 65535).astype(np.uint16)
    out_img = Image.fromarray(depth_16, mode="I;16")
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()
