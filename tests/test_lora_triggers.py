"""LoRA trigger hints (core.lora_triggers).

Triggers are not stored in the safetensors file (only a lone `realistic` tag in two ai-toolkit
files), so the app ships a seed table of known LoRAs keyed by file name and lets the user override
or add one per file (saved in lora_triggers.json, runtime data).
"""
import json

import pytest
from safetensors.torch import save_file
import torch


@pytest.fixture
def store(tmp_path, monkeypatch):
    from core import lora_triggers
    monkeypatch.setattr(lora_triggers, "STORE_PATH", tmp_path / "lora_triggers.json")
    return lora_triggers


def _lora(tmp_path, name, metadata=None):
    p = tmp_path / name
    save_file({"x": torch.zeros(1)}, str(p), metadata=metadata)
    return str(p)


def test_seeded_trigger_by_file_name(store, tmp_path):
    p = _lora(tmp_path, "dever_devil_may_cry_flux2_klein_9b.safetensors")
    info = store.get_trigger_info(p)
    assert info["trigger"] == "dmc_style"
    assert info["source"].startswith("https://huggingface.co/")


def test_seed_without_trigger_keeps_the_note(store, tmp_path):
    info = store.get_trigger_info(_lora(tmp_path, "Klein-consistency.safetensors"))
    assert info["trigger"] is None and "no trigger" in info["note"].lower()


def test_both_bfs_files_share_the_head_swap_prompt(store, tmp_path):
    for n in ("bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors",
              "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors"):
        assert store.get_trigger_info(_lora(tmp_path, n))["trigger"].startswith("head_swap:")


def test_unknown_file_has_no_trigger(store, tmp_path):
    info = store.get_trigger_info(_lora(tmp_path, "mystery.safetensors"))
    assert info == {"trigger": None, "note": "", "source": "", "origin": "none"}


def test_single_metadata_tag_is_used_when_not_seeded(store, tmp_path):
    p = _lora(tmp_path, "mystery.safetensors",
              {"ss_tag_frequency": json.dumps({"1_realistic": {"realistic": 1}})})
    info = store.get_trigger_info(p)
    assert info["trigger"] == "realistic" and info["origin"] == "metadata"


def test_many_metadata_tags_are_not_a_trigger(store, tmp_path):
    p = _lora(tmp_path, "mystery.safetensors",
              {"ss_tag_frequency": json.dumps({"d": {"1girl": 3, "solo": 2}})})
    assert store.get_trigger_info(p)["trigger"] is None


def test_user_override_wins_and_persists(store, tmp_path):
    p = _lora(tmp_path, "dever_devil_may_cry_flux2_klein_9b.safetensors")
    store.set_trigger("dever_devil_may_cry_flux2_klein_9b.safetensors", "my_trigger")
    info = store.get_trigger_info(p)
    assert info["trigger"] == "my_trigger" and info["origin"] == "user"
    assert json.loads((tmp_path / "lora_triggers.json").read_text())[
        "dever_devil_may_cry_flux2_klein_9b.safetensors"] == "my_trigger"


def test_clearing_an_override_falls_back_to_the_seed(store, tmp_path):
    n = "dever_devil_may_cry_flux2_klein_9b.safetensors"
    store.set_trigger(n, "my_trigger")
    store.set_trigger(n, "")
    assert store.get_trigger_info(_lora(tmp_path, n))["trigger"] == "dmc_style"


def test_set_trigger_rejects_path_names(store):
    with pytest.raises(ValueError):
        store.set_trigger("../evil.safetensors", "x")


def test_refcontrol_depth_seeded_under_both_file_names(store, tmp_path):
    for n in ("refcontrol_depth_klein9b.safetensors", "flux2_klein_9b_refcontrol_depth.safetensors"):
        info = store.get_trigger_info(_lora(tmp_path, n))
        assert info["trigger"] == "refcontrol"
        assert "depth" in info["note"].lower() and info["source"].endswith("reference-depth-lora")


def test_civitai_trained_words_fill_trigger_after_seed_and_user(store, tmp_path, monkeypatch):
    from core import civitai_install as ci
    monkeypatch.setattr(ci, "BASE_DIR", tmp_path)
    (tmp_path / "civitai_installed.json").write_text(
        '{"FComic.safetensors": {"modelId": 1, "versionId": 2, "family": "klein-9B", "trained": ["ComSpa"]}}')
    path = str(tmp_path / "FComic.safetensors")
    info = store.get_trigger_info(path)
    assert info["trigger"] == "ComSpa" and info["origin"] == "civitai"
    store.set_trigger("FComic.safetensors", "my own")
    assert store.get_trigger_info(path)["origin"] == "user"
