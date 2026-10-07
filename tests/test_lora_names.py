"""Friendly LoRA names for the dropdown: no extension, underscores as spaces, CivitAI name when known."""
import json

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from core import civitai_install as ci
from core import lora_names


@pytest.mark.parametrize("raw, shown", [
    ("Klein-consistency.safetensors", "Klein-consistency"),
    ("pasta_IL_v3.safetensors", "pasta IL v3"),
    ("a__b.pt", "a b"),
    ("x.SAFETENSORS", "x"),
    ("70sSciFiKlein4B.safetensors", "70sSciFiKlein4B"),
    ("_.safetensors", "_.safetensors"),                 # nothing left: keep the file name
])
def test_clean_name(raw, shown):
    assert lora_names.clean_name(raw) == shown


def test_display_prefers_the_civitai_model_name(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "BASE_DIR", tmp_path)
    (tmp_path / "civitai_installed.json").write_text(json.dumps({
        "70sSciFiKlein4B.safetensors": {"modelId": 1, "versionId": 2, "family": "klein-4B", "name": "70s Sci-Fi Movie"},
        "old_entry.safetensors": {"modelId": 3, "versionId": 4, "family": "klein-9B"},       # no name stored
    }))
    assert lora_names.display_name("70sSciFiKlein4B.safetensors") == "70s Sci-Fi Movie"
    assert lora_names.display_name("old_entry.safetensors") == "old entry"
    assert lora_names.display_name("mine_v2.safetensors") == "mine v2"


def test_lora_list_returns_display_for_every_file(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(ci, "BASE_DIR", tmp_path)
    (tmp_path / "lora_uploads").mkdir()
    for n in ("pasta_IL_v3.safetensors", "70sSciFiKlein4B.safetensors"):
        save_file({"x": torch.zeros(1)}, str(tmp_path / "lora_uploads" / n))
    (tmp_path / "civitai_installed.json").write_text(json.dumps(
        {"70sSciFiKlein4B.safetensors": {"modelId": 1, "versionId": 2, "family": "klein-4B", "name": "70s Sci-Fi Movie"}}))
    files = TestClient(server.app).get("/api/lora/list").json()["files"]
    assert {f["name"]: f["display"] for f in files} == {
        "pasta_IL_v3.safetensors": "pasta IL v3", "70sSciFiKlein4B.safetensors": "70s Sci-Fi Movie"}


def test_download_stores_the_model_name_in_the_registry(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(ci, "BASE_DIR", tmp_path)
    (tmp_path / "lora_uploads").mkdir()
    body = b"abcdefgh"
    row = {"id": "civ-5-klein-9B", "name": "Comic Style", "family": "klein-9B", "civitai": {
        "modelId": 5, "versionId": 11, "file": "Comic.safetensors", "sizeKB": 1,
        "sha256": hashlib.sha256(body).hexdigest(), "trained": []}}

    class R:
        status_code, headers = 200, {}
        def iter_content(self, n): yield body
        def close(self): pass
    ci.start_download(row, open_stream=lambda u, k: R(), verify=lambda p, f: None, threaded=False)
    assert lora_names.display_name("Comic.safetensors") == "Comic Style"
