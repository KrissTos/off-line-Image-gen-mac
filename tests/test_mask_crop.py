"""Unit tests for mask crop mode reference handling (app.crop_flux_refs).

Regression: crop mode used to replace ALL FLUX refs with the crop of slot #1,
so material/style refs in slots #2+ never reached the model.
"""
from PIL import Image


def test_crop_replaces_slot_one_and_keeps_extra_refs():
    import app
    room, wall, floor = (Image.new("RGB", (8, 8), c) for c in ("red", "green", "blue"))
    crop = Image.new("RGB", (4, 4), "white")
    out = app.crop_flux_refs([room, wall, floor], crop)
    assert out == [crop, wall, floor]


def test_crop_single_ref():
    import app
    crop = Image.new("RGB", (4, 4))
    assert app.crop_flux_refs([Image.new("RGB", (8, 8))], crop) == [crop]


def test_crop_no_refs():
    import app
    crop = Image.new("RGB", (4, 4))
    assert app.crop_flux_refs(None, crop) == [crop]
