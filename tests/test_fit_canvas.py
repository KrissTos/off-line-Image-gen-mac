"""Unit tests for fitting slot #1 (+ its mask) into the output canvas (app.fit_ref_to_canvas).

Regression: a ref whose aspect differed from the output size was stretched to the
output dims whenever a mask was present, and auto-outpaint padded with black then
fed FLUX plain img2img (mask ignored) — so no real extension ever happened.
"""
import numpy as np
from PIL import Image


def _solid(w, h, color="red"):
    return Image.new("RGB", (w, h), color)


def _half_mask(w, h):
    """White (edit) on the left half, black on the right."""
    m = Image.new("L", (w, h), 0)
    m.paste(255, (0, 0, w // 2, h))
    return m


def test_same_aspect_is_plain_resize_no_padding():
    import app
    fit = app.fit_ref_to_canvas(_solid(1024, 768), 512, 384)
    assert fit.canvas.size == (512, 384)
    assert not fit.padded
    assert fit.mask is None
    assert fit.warning is None


def test_near_same_aspect_rounding_does_not_pad():
    import app
    fit = app.fit_ref_to_canvas(_solid(1000, 667), 768, 512)
    assert fit.canvas.size == (768, 512)
    assert not fit.padded
    assert fit.mask is None


def test_different_aspect_fits_without_stretch_and_pads_center():
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300), 512, 512)
    assert fit.canvas.size == (512, 512)
    assert fit.padded
    m = np.array(fit.mask)
    assert m.shape == (512, 512)
    # ref fitted to 512x384, centered → rows 64..447 preserved, top/bottom 64 rows generated
    assert m[10, 256] == 255 and m[500, 256] == 255
    assert m[256, 256] == 0 and m[64, 10] == 0 and m[447, 500] == 0
    assert m[63, 256] == 255 and m[448, 256] == 255


def test_padding_is_filled_from_image_not_black():
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300, "red"), 512, 512)
    r, g, b = fit.canvas.getpixel((256, 5))
    assert r > 200 and g < 40 and b < 40  # blurred edge-fill, not (0,0,0)
    assert fit.canvas.getpixel((256, 256)) == (255, 0, 0)


def test_align_top_places_ref_at_top():
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300), 512, 512, align="top")
    m = np.array(fit.mask)
    assert m[0, 256] == 0 and m[383, 256] == 0
    assert m[384, 256] == 255 and m[511, 256] == 255


def test_small_ref_is_scaled_up_to_fit():
    import app
    fit = app.fit_ref_to_canvas(_solid(200, 150), 512, 512)
    m = np.array(fit.mask)
    # fitted to 512x384 (not left at 200x150 in the middle)
    assert m[256, 0] == 0 and m[256, 511] == 0
    assert m[64, 256] == 0 and m[63, 256] == 255


def test_user_mask_follows_ref_transform_and_unions_padding():
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300), 512, 512, mask=_half_mask(400, 300))
    assert fit.warning is None
    m = np.array(fit.mask)
    assert m[256, 100] == 255   # user-masked left half, inside the fitted ref
    assert m[256, 400] == 0     # right half preserved
    assert m[10, 400] == 255    # padding still generated


def test_user_mask_same_aspect_no_padding():
    import app
    fit = app.fit_ref_to_canvas(_solid(1024, 768), 512, 384, mask=_half_mask(1024, 768))
    assert not fit.padded
    m = np.array(fit.mask)
    assert m.shape == (384, 512)
    assert m[200, 100] == 255 and m[200, 400] == 0


def test_mismatched_uploaded_mask_warns_and_stretches_to_ref():
    import app
    fit = app.fit_ref_to_canvas(_solid(400, 300), 512, 512, mask=_half_mask(300, 300))
    assert fit.warning is not None
    m = np.array(fit.mask)
    assert m[256, 100] == 255 and m[256, 400] == 0


def test_iterate_pass_mask_fits_onto_prior_output():
    """Pass 2+: base is the previous output (already canvas size); the mask was drawn
    on slot #1's original aspect → must land where pass 1 placed slot #1, no warning."""
    import app
    fit = app.fit_ref_to_canvas(_solid(512, 512), 512, 512, mask=_half_mask(400, 300))
    assert fit.warning is None
    assert not fit.padded
    m = np.array(fit.mask)
    assert m[256, 100] == 255 and m[256, 400] == 0
    assert m[10, 100] == 0      # letterbox area of pass 1 is NOT regenerated again


def test_empty_user_mask_without_padding_gives_no_mask():
    import app
    fit = app.fit_ref_to_canvas(_solid(512, 384), 512, 384, mask=Image.new("L", (512, 384), 0))
    assert fit.mask is None


def test_flux_extra_refs_keep_native_size():
    import app
    canvas = _solid(512, 512)
    tile = _solid(300, 900, "blue")
    refs = app.prepare_flux_refs(canvas, [_solid(400, 300), tile])
    assert refs[0] is canvas
    assert refs[1].size == (300, 900)
    assert refs[1].mode == "RGB"
