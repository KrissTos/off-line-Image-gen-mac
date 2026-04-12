"""Unit tests for core/erase.py — watermark detection and removal."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_test_image(w: int = 256, h: int = 256) -> Path:
    """Create a temporary solid-colour PNG for testing."""
    import tempfile
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add a fake watermark: bright rectangle in top-right quadrant
    arr[10:50, w - 80:w - 10, :] = 240
    img = Image.fromarray(arr, mode="RGB")
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    return Path(f.name)


def test_detect_watermark_returns_png_bytes():
    from core.erase import detect_watermark
    path = _make_test_image()
    result = detect_watermark(path)
    assert isinstance(result, bytes)
    assert len(result) > 0
    # Should be a valid PNG
    img = Image.open(io.BytesIO(result))
    assert img.mode == "L"


def test_detect_watermark_same_size_as_input():
    from core.erase import detect_watermark
    path = _make_test_image(400, 300)
    result = detect_watermark(path)
    mask = Image.open(io.BytesIO(result))
    assert mask.size == (400, 300)


def test_detect_watermark_blank_image_returns_mask():
    """Blank image has no watermark — should return a blank (all-black) mask."""
    from core.erase import detect_watermark
    import tempfile
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    result = detect_watermark(Path(f.name))
    mask = Image.open(io.BytesIO(result))
    arr_mask = np.array(mask)
    # Blank image → no high-frequency content → mask should be mostly dark
    assert arr_mask.mean() < 50


def test_remove_watermark_returns_png_bytes():
    from core.erase import remove_watermark
    path = _make_test_image()
    # Simple all-white mask (erase everything)
    mask_img = Image.fromarray(np.full((256, 256), 255, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    result = remove_watermark(path, mask_bytes)
    assert isinstance(result, bytes)
    assert len(result) > 0
    out = Image.open(io.BytesIO(result))
    assert out.mode in ("RGB", "RGBA")


def test_remove_watermark_output_matches_source_size():
    from core.erase import remove_watermark
    path = _make_test_image(512, 384)
    mask_img = Image.fromarray(np.zeros((384, 512), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    result = remove_watermark(path, buf.getvalue())
    out = Image.open(io.BytesIO(result))
    assert out.size == (512, 384)
