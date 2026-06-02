"""Download filtering — the LTX 0.9.8-13B-distilled repo nests duplicate transformer +
text_encoder weights under vae/ (45 GB) that diffusers never loads. snapshot_download must
skip them; other repos must download unfiltered.
"""
import queue


def test_ltx_download_skips_duplicate_weights(monkeypatch):
    import huggingface_hub
    import app

    calls = {}

    def fake_snapshot(repo_id, **kw):
        calls["repo_id"] = repo_id
        calls["ignore"] = kw.get("ignore_patterns")
        return "/tmp/fake"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    app.download_model_update_stream("LTX-Video", queue.Queue())

    assert calls["repo_id"] == "Lightricks/LTX-Video-0.9.8-13B-distilled"
    assert "vae/transformer/*" in calls["ignore"]
    assert "vae/text_encoder/*" in calls["ignore"]


def test_other_repo_download_unfiltered(monkeypatch):
    import huggingface_hub
    import app

    calls = {}

    def fake_snapshot(repo_id, **kw):
        calls["ignore"] = kw.get("ignore_patterns")
        return "/tmp/fake"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    app.download_model_update_stream("Z-Image Turbo (Quantized)", queue.Queue())

    assert calls["ignore"] is None  # no junk to skip → full download
