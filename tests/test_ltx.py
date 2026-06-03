"""Unit tests for LTX-Video 0.9.8-13B-distilled render helper (app.render_ltx_video).

Pipelines are passed in and mocked — no model download. Verifies the multiscale
vs fast-preview branching, distilled timesteps, image conditioning, frame/dim
normalization, and progress-callback mapping.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image


def _frames_out(n=1, size=(64, 64)):
    """A pipeline return object whose .frames[0] is a list of n PIL frames."""
    return SimpleNamespace(frames=[[Image.new("RGB", size) for _ in range(n)]])


def test_fast_preview_single_pass():
    import app
    vp = MagicMock(return_value=_frames_out(2))
    up = MagicMock()
    frames = app.render_ltx_video(
        vp, up, prompt="x", ref_image=None, num_frames=25,
        width=832, height=480, generator=None, fast_preview=True,
    )
    assert vp.call_count == 1
    assert up.call_count == 0  # upsampler untouched in fast mode
    kw = vp.call_args.kwargs
    assert kw["output_type"] == "pil"
    assert kw["timesteps"] == app.LTX_BASE_TIMESTEPS
    assert kw["guidance_scale"] == 1.0
    assert len(frames) == 2


def test_multiscale_three_stage():
    import app
    gen_out = SimpleNamespace(frames="LATENTS")          # stage 1 (latent)
    denoise_out = _frames_out(3, size=(64, 64))           # stage 3 (pil)
    vp = MagicMock(side_effect=[gen_out, denoise_out])
    up = MagicMock(return_value=SimpleNamespace(frames="UP_LATENTS"))

    frames = app.render_ltx_video(
        vp, up, prompt="x", ref_image=None, num_frames=25,
        width=832, height=480, generator=None, fast_preview=False,
    )

    assert vp.call_count == 2        # base gen + final denoise
    assert up.call_count == 1        # one upsample pass
    base, denoise = vp.call_args_list
    assert base.kwargs["output_type"] == "latent"
    assert base.kwargs["timesteps"] == app.LTX_BASE_TIMESTEPS
    assert denoise.kwargs["output_type"] == "pil"
    assert denoise.kwargs["timesteps"] == app.LTX_DENOISE_TIMESTEPS
    assert denoise.kwargs["latents"] == "UP_LATENTS"
    assert up.call_args.kwargs["latents"] == "LATENTS"
    assert up.call_args.kwargs["tone_map_compression_ratio"] == 0.6
    # final frames resized down to requested target (832x480, both /32)
    assert frames[0].size == (832, 480)
    assert len(frames) == 3


def test_i2v_builds_condition(monkeypatch):
    import app
    cond_sentinel = object()
    fake_cond = MagicMock(return_value=cond_sentinel)
    monkeypatch.setattr(
        "diffusers.pipelines.ltx.pipeline_ltx_condition.LTXVideoCondition",
        fake_cond,
    )
    vp = MagicMock(return_value=_frames_out(1))
    ref = Image.new("RGB", (100, 100))
    app.render_ltx_video(
        vp, None, prompt="x", ref_image=ref, num_frames=9,
        width=512, height=512, generator=None, fast_preview=True,
    )
    fake_cond.assert_called_once()
    assert fake_cond.call_args.kwargs.get("frame_index") == 0
    assert vp.call_args.kwargs["conditions"] == [cond_sentinel]


def test_multi_ref_builds_keyframes(monkeypatch):
    """Multiple ref images -> one LTXVideoCondition each, frame indices spread
    across the timeline, latent-aligned (multiple of 8), strictly increasing."""
    import app
    calls = []
    def _fake_cond(**kw):
        calls.append(kw)
        return ("cond", kw["frame_index"])
    monkeypatch.setattr(
        "diffusers.pipelines.ltx.pipeline_ltx_condition.LTXVideoCondition",
        _fake_cond,
    )
    vp = MagicMock(return_value=_frames_out(1))
    refs = [Image.new("RGB", (64, 64)) for _ in range(3)]
    app.render_ltx_video(
        vp, None, prompt="x", ref_image=refs, num_frames=25,  # last frame = 24
        width=512, height=512, generator=None, fast_preview=True,
    )
    idxs = [c["frame_index"] for c in calls]
    assert len(idxs) == 3
    assert idxs[0] == 0 and idxs[-1] == 24          # span whole clip
    assert idxs == sorted(set(idxs))                # strictly increasing
    assert all(i % 8 == 0 for i in idxs)            # latent stride
    assert len(vp.call_args.kwargs["conditions"]) == 3


def test_multi_ref_capped_to_available_slots(monkeypatch):
    """More refs than latent slots -> conditions capped, no duplicate frames."""
    import app
    monkeypatch.setattr(
        "diffusers.pipelines.ltx.pipeline_ltx_condition.LTXVideoCondition",
        lambda **kw: kw["frame_index"],
    )
    vp = MagicMock(return_value=_frames_out(1))
    refs = [Image.new("RGB", (64, 64)) for _ in range(6)]
    app.render_ltx_video(
        vp, None, prompt="x", ref_image=refs, num_frames=9,  # last=8 -> slots {0,8}
        width=512, height=512, generator=None, fast_preview=True,
    )
    conds = vp.call_args.kwargs["conditions"]
    assert conds == sorted(set(conds))   # no dup frame indices, within {0,8}
    assert len(conds) <= 2


def test_txt2video_passes_no_conditions():
    import app
    vp = MagicMock(return_value=_frames_out(1))
    app.render_ltx_video(
        vp, None, prompt="x", ref_image=None, num_frames=9,
        width=512, height=512, generator=None, fast_preview=True,
    )
    assert vp.call_args.kwargs["conditions"] is None


def test_num_frames_and_dims_normalized():
    import app
    vp = MagicMock(return_value=_frames_out(1))
    app.render_ltx_video(
        vp, None, prompt="x", ref_image=None, num_frames=20,
        width=500, height=300, generator=None, fast_preview=True,
    )
    kw = vp.call_args.kwargs
    assert kw["num_frames"] == 17          # (20-1)//8*8+1
    assert kw["width"] % 32 == 0
    assert kw["height"] % 32 == 0


def test_progress_callback_maps_to_percent():
    import app
    vp = MagicMock(return_value=_frames_out(1))
    calls = []
    app.render_ltx_video(
        vp, None, prompt="x", ref_image=None, num_frames=9,
        width=512, height=512, generator=None, fast_preview=True,
        step_callback=lambda current, total: calls.append((current, total)),
    )
    cb = vp.call_args.kwargs["callback_on_step_end"]
    cb(None, 0, None, {})
    assert calls and calls[-1][1] == 100
    assert 0 <= calls[-1][0] <= 100
