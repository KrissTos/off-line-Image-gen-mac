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


# ── Model Sources cleanup: list read, save and Update wiring ─────────────────────

import json


def _lora(name, **kw):
    return {"id": name, "name": name, "url": f"https://huggingface.co/org/{name}",
            "type": "lora", "description": "", "model_choice": "", **kw}


def _use_file(monkeypatch, tmp_path, sources, ignored=None):
    import server
    f = tmp_path / "model_sources.json"
    data = {"version": 1, "sources": sources}
    if ignored is not None:
        data["ignored"] = ignored
    f.write_text(json.dumps(data))
    monkeypatch.setattr(server, "MODEL_SOURCES_FILE", f)
    return f


def test_get_prunes_unsupported_loras_but_keeps_custom(monkeypatch, tmp_path):
    import server
    _use_file(monkeypatch, tmp_path, [_lora("Flux2-Klein-9B-Consistency"), _lora("huggy_v17"),
                                      _lora("my-own", custom=True)])
    names = [s["name"] for s in server.api_get_model_sources()["sources"]]
    assert names == ["Flux2-Klein-9B-Consistency", "my-own"]


def test_save_keeps_the_ignored_list(monkeypatch, tmp_path):
    import server
    f = _use_file(monkeypatch, tmp_path, [], ignored=["https://huggingface.co/org/huggy_v17"])
    server.api_save_model_sources({"sources": [_lora("Flux2-Klein-9B-Consistency", custom=True)]})
    saved = json.loads(f.read_text())
    assert saved["ignored"] == ["https://huggingface.co/org/huggy_v17"]
    assert saved["sources"][0]["custom"] is True


def test_discover_screens_describes_and_remembers(monkeypatch, tmp_path):
    import server
    from core import model_sources as ms
    f = _use_file(monkeypatch, tmp_path, [])
    cands = [_lora("Flux2-Klein-9B-Enhanced-Details"), _lora("huggy_v17")]
    monkeypatch.setattr(server, "_discover_candidates", lambda existing, next_id: cands)
    monkeypatch.setattr(ms, "hf_fetch_files", lambda repo: ["w.safetensors"])
    monkeypatch.setattr(ms, "hf_fetch_card", lambda repo: "Sharpens fine detail in photos.")
    from core import civitai
    monkeypatch.setattr(civitai, "http_fetch_models", lambda base, limit: [])   # Update also asks CivitAI
    monkeypatch.setattr(server, "_show_nsfw", lambda: False)
    out = server.api_discover_model_sources()
    assert out["added"] == 1 and out["skipped"] == 1 and out["described"] == 1
    saved = json.loads(f.read_text())
    assert [s["name"] for s in saved["sources"]] == ["Flux2-Klein-9B-Enhanced-Details"]
    assert saved["sources"][0]["description"] == "Sharpens fine detail in photos."
    assert saved["ignored"] == ["https://huggingface.co/org/huggy_v17"]


def test_discovered_base_row_gets_model_choice_from_defaults_by_url(monkeypatch, tmp_path):
    """Regression: a base row for a known repo added by Update (new id, empty model_choice)
    never matched the available list, so the LTX card had no green frame and no Download."""
    import server
    ltx = next(s for s in server.DEFAULT_SOURCES if s["name"] == "LTX-Video")
    _use_file(monkeypatch, tmp_path, [
        {"id": "src-116", "name": "LTX-Video-0.9.8-13B-distilled", "url": ltx["url"],
         "type": "base", "description": "", "model_choice": ""}])
    rows = server.api_get_model_sources()["sources"]
    assert rows[0]["model_choice"] == ltx["model_choice"]
    assert rows[0]["name"] == "LTX-Video"   # the download endpoint and the Download gate key on it


def test_base_row_with_its_own_model_choice_is_left_alone(monkeypatch, tmp_path):
    import server
    ltx = next(s for s in server.DEFAULT_SOURCES if s["name"] == "LTX-Video")
    _use_file(monkeypatch, tmp_path, [
        {"id": "src-9", "name": "mine", "url": ltx["url"], "type": "base",
         "description": "", "model_choice": "custom choice"}])
    assert server.api_get_model_sources()["sources"][0]["model_choice"] == "custom choice"
