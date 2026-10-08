"""Small masks get a context crop upscaled toward ~1 MP (app.plan_mask_crop).

Regression: a 117x83 px mask on a 1376x768 canvas gave the model about 7x5 latent
tokens (full-frame inpaint) or a native 192x192 crop (crop mode), so a wide logo
came out as a smeared blob. Small masks must be generated on an upscaled crop.
"""
from PIL import Image, ImageDraw


def _mask(size, box):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m


def test_small_mask_is_planned_and_upscaled():
    import app
    plan = app.plan_mask_crop(_mask((1376, 768), (626, 312, 743, 395)), (1376, 768))
    assert plan is not None
    x0, y0, x1, y1 = plan.bbox
    assert x0 <= 626 and y0 <= 312 and x1 >= 744 and y1 >= 396   # mask inside the crop
    cw, ch = x1 - x0, y1 - y0
    ww, wh = plan.work_size
    assert ww % 16 == 0 and wh % 16 == 0
    assert ww * wh > 4 * cw * ch                  # clearly more pixels than the native crop
    assert ww * wh <= 1_100_000                   # about 1 MP, not more


def test_context_margin_is_proportional():
    import app
    small = app.plan_mask_crop(_mask((2000, 2000), (900, 900, 960, 960)), (2000, 2000))
    big = app.plan_mask_crop(_mask((2000, 2000), (700, 700, 1000, 1000)), (2000, 2000))
    # margin around the mask grows with the mask, never below the floor
    assert (small.bbox[2] - small.bbox[0]) - 61 >= 2 * 64
    assert (big.bbox[2] - big.bbox[0]) - 301 > (small.bbox[2] - small.bbox[0]) - 61


def test_large_mask_keeps_legacy_path():
    import app
    assert app.plan_mask_crop(_mask((1376, 768), (200, 100, 1100, 700)), (1376, 768)) is None


def test_empty_mask_returns_none():
    import app
    assert app.plan_mask_crop(Image.new("L", (512, 512), 0), (512, 512)) is None


def test_crop_already_near_target_is_not_upscaled():
    import app
    # crop with context is already about 1 MP: scale < min_scale, nothing to gain
    assert app.plan_mask_crop(_mask((4096, 4096), (1000, 1000, 1500, 1500)), (4096, 4096)) is None


def test_upscale_is_capped_for_tiny_masks():
    import app
    plan = app.plan_mask_crop(_mask((2000, 2000), (1000, 1000, 1010, 1010)), (2000, 2000))
    cw, ch = plan.bbox[2] - plan.bbox[0], plan.bbox[3] - plan.bbox[1]
    assert plan.work_size[0] <= 4 * cw + 16 and plan.work_size[1] <= 4 * ch + 16


def test_wide_crop_long_side_is_capped():
    import app
    plan = app.plan_mask_crop(_mask((6000, 3000), (100, 1000, 700, 1060)), (6000, 3000))
    assert plan is not None
    assert max(plan.work_size) <= 1536


def test_crop_work_keeps_mask_binary_and_sizes_match():
    import app
    ref = Image.new("RGB", (1376, 768), "red")
    mask = _mask((1376, 768), (626, 312, 743, 395))
    plan = app.plan_mask_crop(mask, (1376, 768))
    ref_w, mask_w = app.crop_mask_work(ref, mask, plan)
    assert ref_w.size == plan.work_size and mask_w.size == plan.work_size
    assert ref_w.mode == "RGB" and mask_w.mode == "L"
    assert set(mask_w.getdata()) <= {0, 255}      # NEAREST: no grey ramp
