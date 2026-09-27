"""FLUX 4B outpaint LoRA path (fal/flux-2-klein-4B-outpaint-lora).

The LoRA fills pure-green (0,255,0) borders; verified on our distilled 4B SDNQ
(seamless fill vs. green left untouched without it). 9B / Z-Image keep the
blurred-image pad (no compatible LoRA).
"""
import numpy as np
from PIL import Image


def _solid(w, h, color="red"):
    return Image.new("RGB", (w, h), color)


def test_green_pad_fill():
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300), 512, 512, pad_fill="green")
    assert fit.padded
    assert fit.canvas.getpixel((256, 5)) == (0, 255, 0)
    assert fit.canvas.getpixel((256, 506)) == (0, 255, 0)
    assert fit.canvas.getpixel((256, 256)) == (255, 0, 0)


def test_default_pad_is_still_blur():
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300), 512, 512)
    assert fit.canvas.getpixel((256, 5)) != (0, 255, 0)


def test_lora_used_only_for_4b_when_padding():
    import app
    wide, square = (768, 512), (512, 512)
    for model in ("flux2-klein-sdnq", "flux2-klein-int8"):
        assert app.wants_outpaint_lora(model, False, wide, 512, 512, "center")
        assert not app.wants_outpaint_lora(model, False, square, 512, 512, "center")  # no padding
        assert not app.wants_outpaint_lora(model, True, wide, 512, 512, "center")     # video
    for model in ("flux2-klein-9b-sdnq", "zimage-full", "zimage-quant", None):
        assert not app.wants_outpaint_lora(model, False, wide, 512, 512, "center")
    # ≤3% drift fills instead of padding → no LoRA
    assert not app.wants_outpaint_lora("flux2-klein-sdnq", False, (4000, 3000), 1184, 880, "center")


def test_outpaint_prompt_prepends_trigger():
    import app
    assert app.outpaint_prompt("") == app.OUTPAINT_LORA_TRIGGER
    assert app.outpaint_prompt("  ") == app.OUTPAINT_LORA_TRIGGER
    p = app.outpaint_prompt("a train in a station")
    assert p.startswith(app.OUTPAINT_LORA_TRIGGER) and p.endswith("a train in a station")


def test_lora_download_failure_returns_none(monkeypatch):
    import huggingface_hub
    import app

    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    assert app.get_outpaint_lora_path() is None


def test_composite_edge_never_shows_the_pad_fill():
    """The composite's soft edge must fall on the ORIGINAL side of the mask: pixels
    inside the (green) pad come 100% from the generated image, or a green line shows."""
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300), 512, 512, pad_fill="green")
    generated = _solid(512, 512, "blue")
    out = app.apply_mask_composite(fit.canvas, generated, fit.mask, (0, 0, 512, 512))
    a = np.array(out).astype(int)
    pad_rows = np.array(fit.mask)[:, 256] == 255
    assert (a[pad_rows, 256] == [0, 0, 255]).all()          # every pad pixel fully generated
    assert tuple(a[256, 256]) == (255, 0, 0)                # deep inside: original kept
