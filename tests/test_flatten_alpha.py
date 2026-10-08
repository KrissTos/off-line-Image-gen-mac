"""Transparent PNGs (logos) must keep their shape when converted to RGB.

Regression: a dark logo PNG whose transparent pixels hold the logo's own colour
(35,35,45,0) became a solid dark rectangle under `convert("RGB")`, so FLUX never
saw the logo.
"""
from PIL import Image, ImageDraw


def _logo(fg, transparent_rgb):
    im = Image.new("RGBA", (200, 60), transparent_rgb + (0,))
    ImageDraw.Draw(im).rectangle((20, 10, 90, 50), fill=fg + (255,))
    return im


def test_dark_logo_lands_on_light_background():
    from core.image_alpha import flatten_to_rgb
    out = flatten_to_rgb(_logo((35, 35, 45), (35, 35, 45)))
    assert out.mode == "RGB"
    assert out.getpixel((2, 2)) == (255, 255, 255)        # was (35, 35, 45): logo invisible
    assert out.getpixel((50, 30)) == (35, 35, 45)         # logo intact


def test_light_logo_lands_on_dark_background():
    from core.image_alpha import flatten_to_rgb
    out = flatten_to_rgb(_logo((240, 240, 240), (240, 240, 240)))
    assert out.getpixel((2, 2)) == (20, 20, 20)
    assert out.getpixel((50, 30)) == (240, 240, 240)


def test_opaque_rgba_and_rgb_are_plain_conversions():
    from core.image_alpha import flatten_to_rgb
    rgba = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    assert flatten_to_rgb(rgba).getpixel((1, 1)) == (10, 20, 30)
    rgb = Image.new("RGB", (8, 8), (1, 2, 3))
    assert flatten_to_rgb(rgb).getpixel((1, 1)) == (1, 2, 3)


def test_palette_png_with_transparency_is_flattened():
    from core.image_alpha import flatten_to_rgb
    p = _logo((35, 35, 45), (35, 35, 45)).convert("P")
    p.info["transparency"] = 0
    out = flatten_to_rgb(p)
    assert out.mode == "RGB"
    assert out.getpixel((2, 2)) == (255, 255, 255)


def test_fully_transparent_image_does_not_crash():
    from core.image_alpha import flatten_to_rgb
    out = flatten_to_rgb(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))
    assert out.size == (4, 4) and out.mode == "RGB"
