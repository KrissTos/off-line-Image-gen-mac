"""Unit tests for model-source filtering.

The model loader only handles repos in app.KNOWN_MODELS. Base-type sources whose
repo isn't loadable must never reach the UI (they can't download or load).
LoRAs/upscalers are link-only by design and are always kept.
"""


def test_drop_unusable_base(monkeypatch):
    import app
    import server

    monkeypatch.setattr(app, "KNOWN_MODELS", {
        "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic": "FLUX.2-klein-4B (4bit SDNQ)",
        "Tongyi-MAI/Z-Image-Turbo": "Z-Image Turbo (Full)",
    })

    sources = [
        {"id": "a", "type": "base",     "url": "https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic"},
        {"id": "b", "type": "base",     "url": "https://huggingface.co/Disty0/Qwen-Image-SDNQ-4bit"},  # unsupported
        {"id": "c", "type": "lora",     "url": "https://huggingface.co/fal/flux-2-klein-4B-zoom-lora"},
        {"id": "d", "type": "upscaler", "url": "https://huggingface.co/Phips/4xNomosWebPhoto_atd"},
    ]

    out = server._drop_unusable_base(sources)
    assert {s["id"] for s in out} == {"a", "c", "d"}  # unsupported base 'b' dropped, others kept


def test_total_memory_positive():
    import app
    # On any real host (MPS recommended-max or sysconf physical RAM) this is > 0.
    assert app.get_total_memory_gb() > 0
