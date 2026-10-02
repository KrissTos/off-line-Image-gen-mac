"""FLUX.2-klein "Inpainting Pipeline (Quality)" uses diffusers' Flux2KleinInpaintPipeline.

The mode used to fall through to img2img. Auto-outpaint also sets this mode, but its
4B outpaint LoRA is trained for the img2img path, so it must stay on img2img.
Pipelines are faked: no model download.
"""
from unittest.mock import MagicMock

from PIL import Image


class _FakeInpaint:
    instances = []

    def __init__(self, **components):
        self.components = components
        self.calls = []
        _FakeInpaint.instances.append(self)

    def __call__(self, **kw):
        self.calls.append(kw)
        out = MagicMock()
        out.images = [Image.new("RGB", (kw["width"], kw["height"]), "blue")]
        return out


def _setup(monkeypatch):
    import diffusers
    import app
    _FakeInpaint.instances.clear()
    img2img = MagicMock(name="klein_pipe")
    img2img.components = {"transformer": object(), "vae": MagicMock()}
    img2img.return_value.images = [Image.new("RGB", (512, 512), "green")]
    monkeypatch.setattr(diffusers, "Flux2KleinInpaintPipeline", _FakeInpaint, raising=False)
    monkeypatch.setattr(app, "pipe", img2img)
    monkeypatch.setattr(app, "inpaint_pipe", None)
    monkeypatch.setattr(app, "img2img_pipe", None)
    monkeypatch.setattr(app, "current_model", "flux2-klein-9b-sdnq")
    monkeypatch.setattr(app, "current_device", "cpu")
    monkeypatch.setattr(app, "current_lora_paths", [])
    monkeypatch.setattr(app, "load_pipeline", lambda choice, device="mps": app.pipe)
    return img2img


def _generate(app, tmp_path, input_images, mask, mask_mode, width=512, height=512, strength=1.0):
    return list(app.generate_image(
        prompt="blue mug", height=height, width=width, steps=4, seed=1, guidance=0.0, device="cpu",
        model_choice="FLUX.2-klein-9B (4bit SDNQ - Higher Quality)", model_source_choice="Local",
        input_images=input_images, lora_file=None, lora_strength=1.0, img_strength=strength,
        repeat_count=1, auto_save=False, output_dir=str(tmp_path), upscale_enabled=False,
        upscale_model_path="", num_frames=25, fps_val=24, mask_image=mask, mask_mode=mask_mode,
    ))


def _mask():
    m = Image.new("L", (512, 512), 0)
    m.paste(255, (128, 128, 384, 384))
    return m


def test_inpainting_mode_uses_native_pipeline(monkeypatch, tmp_path):
    import app
    img2img = _setup(monkeypatch)
    base = Image.new("RGB", (512, 512), "red")
    ref2 = Image.new("RGB", (100, 80), "white")
    res = _generate(app, tmp_path, [base, ref2], _mask(), "Inpainting Pipeline (Quality)", strength=0.9)

    assert len(_FakeInpaint.instances) == 1
    fake = _FakeInpaint.instances[0]
    assert fake.components is not None and "transformer" in fake.components
    kw = fake.calls[0]
    assert kw["image"].size == (512, 512)
    assert kw["mask_image"].size == (512, 512)
    assert kw["strength"] == 0.9
    assert kw["width"] == 512 and kw["height"] == 512
    assert kw["image_reference"] == [ref2]
    assert "padding_mask_crop" not in kw          # produces a flat rectangle on klein
    img2img.assert_not_called()
    info = res[-1][2]
    assert "inpainting" in info and "masked-composite" in info   # outside the mask stays original
    final = res[-1][0]
    assert final.getpixel((10, 10)) == (255, 0, 0)               # untouched pixel is the original


def test_inpaint_pipeline_cached_and_single_ref_has_no_reference(monkeypatch, tmp_path):
    import app
    _setup(monkeypatch)
    base = Image.new("RGB", (512, 512), "red")
    _generate(app, tmp_path, [base], _mask(), "Inpainting Pipeline (Quality)")
    _generate(app, tmp_path, [base], _mask(), "Inpainting Pipeline (Quality)")
    assert len(_FakeInpaint.instances) == 1       # built once, reused
    assert _FakeInpaint.instances[0].calls[0]["image_reference"] is None


def test_crop_mode_still_uses_img2img(monkeypatch, tmp_path):
    import app
    img2img = _setup(monkeypatch)
    _generate(app, tmp_path, [Image.new("RGB", (512, 512), "red")], _mask(), "Crop & Composite (Fast)")
    assert _FakeInpaint.instances == []
    assert img2img.call_count == 1


def test_auto_outpaint_stays_on_img2img(monkeypatch, tmp_path):
    """Slot #1 needs padding -> mode forced to Inpainting, but the outpaint LoRA path is img2img."""
    import app
    img2img = _setup(monkeypatch)
    _generate(app, tmp_path, [Image.new("RGB", (256, 256), "red")], None, "Crop & Composite (Fast)",
              width=512, height=256)
    assert _FakeInpaint.instances == []
    assert img2img.call_count == 1


def test_inpaint_failure_falls_back_to_img2img(monkeypatch, tmp_path):
    import app

    class _Broken(_FakeInpaint):
        def __call__(self, **kw):
            raise RuntimeError("boom")

    img2img = _setup(monkeypatch)
    import diffusers
    monkeypatch.setattr(diffusers, "Flux2KleinInpaintPipeline", _Broken, raising=False)
    res = _generate(app, tmp_path, [Image.new("RGB", (512, 512), "red")], _mask(), "Inpainting Pipeline (Quality)")
    assert img2img.call_count == 1
    assert "inpainting" not in res[-1][2].split("Mode:")[1].split("|")[0]


def test_inpaint_pipeline_rebuilt_when_klein_pipe_replaced(monkeypatch, tmp_path):
    """delete_model()/reload leaves a stale cached inpaint pipe wrapping the old components."""
    import app
    _setup(monkeypatch)
    base = Image.new("RGB", (512, 512), "red")
    _generate(app, tmp_path, [base], _mask(), "Inpainting Pipeline (Quality)")
    new_pipe = MagicMock(name="reloaded_klein")
    new_pipe.components = {"transformer": object(), "vae": MagicMock()}
    monkeypatch.setattr(app, "pipe", new_pipe)
    _generate(app, tmp_path, [base], _mask(), "Inpainting Pipeline (Quality)")
    assert len(_FakeInpaint.instances) == 2
    assert _FakeInpaint.instances[1].components == new_pipe.components
