"""POST /api/workflows/save writes a workflow folder that GET /api/workflows/{name} loads back."""
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def api(monkeypatch, tmp_path):
    import app
    import server
    wf_dir, tmp_dir = tmp_path / "workflows", tmp_path / "uploads"
    tmp_dir.mkdir()
    monkeypatch.setattr(app, "WORKFLOWS_DIR", str(wf_dir))
    monkeypatch.setattr(server, "TEMP_DIR", tmp_dir)
    Image.new("RGB", (40, 30), "white").save(tmp_dir / "img.png")
    Image.new("L", (40, 30), 255).save(tmp_dir / "mask.png")
    yield TestClient(server.app), wf_dir    # no `with`: startup (heartbeat) not run



def test_save_writes_v2_folder_with_image_and_mask(api):
    client, wf_dir = api
    r = client.post("/api/workflows/save", json={
        "name": "bagno", "prompt": "tiles",
        "ref_slots": [{"imageId": "img.png", "maskId": "mask.png", "strength": 0.8}],
    })
    assert r.status_code == 200, r.text
    folder = wf_dir / r.json()["name"]
    assert folder.name.endswith("_bagno")
    data = json.loads((folder / "workflow.json").read_text())
    assert data["version"] == 2 and data["prompt"] == "tiles"
    assert data["ref_slots"][0] == {"image": "refs/slot_1.png", "mask": "masks/slot_1.png", "strength": 0.8}
    assert (folder / "refs" / "slot_1.png").exists()
    assert (folder / "masks" / "slot_1.png").exists()


def test_saved_workflow_loads_back_with_mask(api):
    client, _ = api
    name = client.post("/api/workflows/save", json={
        "name": "roundtrip", "prompt": "p",
        "ref_slots": [{"imageId": "img.png", "maskId": "mask.png"}],
    }).json()["name"]
    r = client.get(f"/api/workflows/{name}")
    assert r.status_code == 200, r.text
    slot = r.json()["ref_slots"][0]
    assert slot["maskUrl"] == f"/api/workflow-assets/{name}/masks/slot_1.png"
    assert client.get(slot["maskUrl"]).status_code == 200
    assert client.get(slot["imageUrl"]).status_code == 200


def test_save_overwrite_keeps_folder_name(api):
    client, wf_dir = api
    name = client.post("/api/workflows/save", json={
        "name": "job", "prompt": "a", "ref_slots": [{"imageId": "img.png"}]}).json()["name"]
    r = client.post("/api/workflows/save", json={"name": "job", "prompt": "b", "overwrite": name, "ref_slots": []})
    assert r.status_code == 200 and r.json()["name"] == name
    assert json.loads((wf_dir / name / "workflow.json").read_text())["prompt"] == "b"
    assert [p.name for p in wf_dir.iterdir()] == [name]


def test_save_overwrite_outside_workflows_dir_is_400(api):
    client, _ = api
    r = client.post("/api/workflows/save", json={"name": "x", "overwrite": "../../etc"})
    assert r.status_code == 400
