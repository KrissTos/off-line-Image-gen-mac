"""Watermark detection and removal.

Detection: heuristic — FFT high-pass + local brightness anomaly.
Removal: LaMa inpainting via simple-lama-inpainting.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

# Module-level LaMa model cache (lazy-loaded on first remove call)
_lama_cache: dict[str, object] = {}


def detect_watermark(path: Path) -> bytes:
    """Heuristic watermark detection — FFT + brightness anomaly.

    Returns a grayscale PNG (white = suspected watermark area).
    The result is intentionally conservative; the user is expected
    to refine it with the mask editor.
    """
    from scipy.ndimage import (
        binary_dilation,
        label,
        laplace,
        uniform_filter,
    )

    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = np.mean(arr, axis=2)

    # --- High-frequency content via Laplacian ---
    edges = np.abs(laplace(gray))
    e_min, e_max = float(edges.min()), float(edges.max())
    if e_max > e_min:
        edges = (edges - e_min) / (e_max - e_min)

    # --- Local brightness anomaly ---
    blur_size = max(min(h, w) // 20, 3)
    local_mean = uniform_filter(gray, size=blur_size)
    anomaly = np.abs(gray - local_mean)
    a_min, a_max = float(anomaly.min()), float(anomaly.max())
    if a_max > a_min:
        anomaly = (anomaly - a_min) / (a_max - a_min)

    # --- Combined score ---
    score = 0.5 * edges + 0.5 * anomaly

    # If the image is perfectly flat there is no activity — return blank mask
    if float(score.max()) == 0.0:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_img = Image.fromarray(mask, mode="L")
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        return buf.getvalue()

    # Threshold at 90th percentile (top 10% of activity)
    threshold = float(np.percentile(score, 90))
    binary = (score >= threshold).astype(np.uint8)

    # Dilate to connect nearby regions
    dilation_iters = max(max(h, w) // 80, 2)
    dilated = binary_dilation(binary, iterations=dilation_iters)

    # Keep only the largest connected component
    labeled, num_features = label(dilated)
    if num_features > 0:
        sizes = np.bincount(labeled.ravel())[1:]
        largest_idx = int(np.argmax(sizes)) + 1
        mask = (labeled == largest_idx).astype(np.uint8) * 255
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    mask_img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return buf.getvalue()


def remove_watermark(image_path: Path, mask_bytes: bytes) -> bytes:
    """Fill the masked region using LaMa inpainting.

    Downloads ~60 MB of model weights on first call.
    Result is always LANCZOS-resized back to source dimensions.
    """
    if "lama" not in _lama_cache:
        print("[erase] Loading LaMa model …")
        from simple_lama_inpainting import SimpleLama
        _lama_cache["lama"] = SimpleLama()
        print("[erase] LaMa loaded.")

    lama = _lama_cache["lama"]

    orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig.size

    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if mask.size != (orig_w, orig_h):
        mask = mask.resize((orig_w, orig_h), Image.NEAREST)

    result = lama(orig, mask)

    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.array(result, dtype=np.uint8))

    result = result.convert("RGB")
    if result.size != (orig_w, orig_h):
        result = result.resize((orig_w, orig_h), Image.LANCZOS)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
