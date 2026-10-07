"""CivitAI endpoints: key, download, delete, list/save/discover with hidden NSFW rows."""
import json

import pytest
from fastapi.testclient import TestClient

import server
from core import civitai, civitai_install as ci
from tests.test_civitai import model, version


def crow(mid, fam, vid, nsfw=False, file="a.safetensors"):
    return {"id": f"civ-{mid}-{fam}", "name": f"m{mid}", "type": "lora", "provider": "civitai",
            "url": f"https://civitai.com/models/{mid}", "family": fam, "nsfw": nsfw, "described": True,
            "description": "", "civitai": {"modelId": mid, "versionId": vid, "file": file,
                                           "sha256": "", "sizeKB": 1, "trained": []}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "BASE_DIR", tmp_path)
    monkeypatch.setattr(server, "MODEL_SOURCES_FILE", tmp_path / "model_sources.json")
    monkeypatch.setattr(server, "_drop_unusable_base", lambda s: s)
    state = {"nsfw": False}
    monkeypatch.setattr(server, "_show_nsfw", lambda: state["nsfw"])
    (tmp_path / "lora_uploads").mkdir()
    ci._jobs.clear()
    return tmp_path, state, TestClient(server.app)


def write_sources(path, rows, ignored=()):
    data = {"version": 1, "sources": rows}
    if ignored:
        data["ignored"] = list(ignored)
    path.write_text(json.dumps(data))


def ids(resp):
    return [s["id"] for s in resp.json()["sources"]]


def test_key_endpoints_never_return_the_key(env):
    _, _, c = env
    assert c.get("/api/civitai/status").json() == {"has_key": False, "show_nsfw": False}
    r = c.post("/api/civitai/key", json={"key": "SECRET"})
    assert r.json() == {"has_key": True} and "SECRET" not in r.text
    assert "SECRET" not in c.get("/api/civitai/status").text
    assert c.post("/api/civitai/key", json={"key": "  "}).status_code == 422
    assert c.delete("/api/civitai/key").json() == {"has_key": False}


def test_list_hides_nsfw_but_keeps_installed_and_annotates(env):
    tmp, state, c = env
    write_sources(tmp / "model_sources.json", [
        crow(1, "klein-9B", 5), crow(2, "klein-9B", 1, nsfw=True), crow(3, "klein-9B", 1, nsfw=True)])
    (tmp / "civitai_installed.json").write_text(json.dumps(
        {"a.safetensors": {"modelId": 3, "versionId": 1, "family": "klein-9B", "trained": []}}))
    r = c.get("/api/model-sources")
    assert sorted(ids(r)) == ["civ-1-klein-9B", "civ-3-klein-9B"]
    assert next(s for s in r.json()["sources"] if s["id"] == "civ-3-klein-9B")["installed"] is True
    state["nsfw"] = True
    assert len(ids(c.get("/api/model-sources"))) == 3


def test_save_from_ui_keeps_hidden_nsfw_rows_and_strips_annotations(env):
    tmp, _, c = env
    write_sources(tmp / "model_sources.json", [crow(1, "klein-9B", 5), crow(2, "klein-9B", 1, nsfw=True)])
    visible = c.get("/api/model-sources").json()["sources"]
    assert [s["id"] for s in visible] == ["civ-1-klein-9B"]
    assert c.post("/api/model-sources", json={"sources": visible}).status_code == 200
    stored = json.loads((tmp / "model_sources.json").read_text())["sources"]
    assert sorted(s["id"] for s in stored) == ["civ-1-klein-9B", "civ-2-klein-9B"]
    assert all("installed" not in s and "update" not in s for s in stored)


def test_removing_a_visible_row_from_the_list_really_removes_it(env):
    tmp, _, c = env
    write_sources(tmp / "model_sources.json", [crow(1, "klein-9B", 5), crow(4, "klein-9B", 1)])
    c.post("/api/model-sources", json={"sources": [crow(4, "klein-9B", 1)]})
    stored = json.loads((tmp / "model_sources.json").read_text())["sources"]
    assert [s["id"] for s in stored] == ["civ-4-klein-9B"]


def test_discover_merges_civitai_keeps_hidden_nsfw_and_survives_a_failed_family(env, monkeypatch):
    tmp, state, c = env
    write_sources(tmp / "model_sources.json", [crow(9, "klein-4B", 1), crow(8, "klein-9B", 1, nsfw=True)])
    monkeypatch.setattr(server, "_discover_candidates", lambda urls, nid: [])
    monkeypatch.setattr(server.model_sources, "hf_fetch_files", lambda r: [])
    monkeypatch.setattr(server.model_sources, "hf_fetch_card", lambda r: "")

    def fetch(base, limit):
        if base.startswith("Flux.2 Klein 4B"):
            raise RuntimeError("HTTP 429")
        if base == "ZImageTurbo":
            return [model(1, "Z comic", [version(10, "ZImageTurbo")])]
        return []
    monkeypatch.setattr(civitai, "http_fetch_models", fetch)
    # hidden NSFW row 8 is not "installed" and not in a failed family: it is dropped by merge (fresh set wins)
    r = c.get("/api/model-sources/discover")
    assert r.status_code == 200
    body = r.json()
    assert body["civitai"]["added"] == 1 and body["civitai"]["failed"] == ["klein-4B"]
    assert sorted(ids(r)) == ["civ-1-Z-Image", "civ-9-klein-4B"]          # 9 kept: its family failed
    stored = json.loads((tmp / "model_sources.json").read_text())["sources"]
    assert sorted(s["id"] for s in stored) == ["civ-1-Z-Image", "civ-9-klein-4B"]


def test_discover_when_civitai_is_down_keeps_everything(env, monkeypatch):
    tmp, _, c = env
    write_sources(tmp / "model_sources.json", [crow(9, "klein-4B", 1), crow(1, "Z-Image", 1)])
    monkeypatch.setattr(server, "_discover_candidates", lambda urls, nid: [])
    monkeypatch.setattr(server.model_sources, "hf_fetch_files", lambda r: [])
    monkeypatch.setattr(server.model_sources, "hf_fetch_card", lambda r: "")
    monkeypatch.setattr(civitai, "http_fetch_models", lambda b, l: (_ for _ in ()).throw(RuntimeError("down")))
    r = c.get("/api/model-sources/discover")
    assert sorted(ids(r)) == ["civ-1-Z-Image", "civ-9-klein-4B"]
    assert sorted(r.json()["civitai"]["failed"]) == ["Z-Image", "klein-4B", "klein-9B"]


def test_download_endpoint_runs_job_and_delete_endpoint_removes(env, monkeypatch):
    tmp, _, c = env
    write_sources(tmp / "model_sources.json", [crow(1, "klein-9B", 5)])
    body = b"abcdefgh"

    class R:
        status_code, headers = 200, {}
        def iter_content(self, n): yield body
        def close(self): pass
    real_start = ci.start_download
    monkeypatch.setattr(ci, "start_download", lambda row, **kw: real_start(
        row, open_stream=lambda u, k: R(), verify=lambda p, f: None, threaded=False))
    r = c.post("/api/civitai/download", json={"version_id": 5})
    assert r.status_code == 200 and r.json()["state"] == "done"
    assert c.get("/api/civitai/download/5").json()["state"] == "done"
    assert (tmp / "lora_uploads" / "a.safetensors").exists()
    assert c.post("/api/civitai/download", json={"version_id": 404}).status_code == 404

    monkeypatch.setattr(server, "_loaded_lora_paths", lambda: [str(tmp / "lora_uploads" / "a.safetensors")])
    assert c.delete("/api/civitai/5").status_code == 409
    monkeypatch.setattr(server, "_loaded_lora_paths", lambda: [])
    assert c.delete("/api/civitai/5").json() == {"deleted": "a.safetensors"}
    assert c.delete("/api/civitai/5").status_code == 404
    assert not (tmp / "lora_uploads" / "a.safetensors").exists()
