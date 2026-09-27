"""Output listing, serving, run loading, delete and upscale on run folders."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def api(monkeypatch, tmp_path):
    import server
    from core import run_store
    out = tmp_path / "out"
    monkeypatch.setattr(server, "_output_dir", lambda: str(out))
    trashed = []
    monkeypatch.setattr(run_store, "trash", lambda p: trashed.append(Path(p).name))
    run = run_store.create_run(out, {"prompt": "hello", "seed": 0}, [])
    files = []
    for s in (1, 2):
        f = run_store.output_filename(run, s, ".png")
        Image.new("RGB", (8, 6)).save(f)
        run_store.add_output(run, f, "image", seed=s)
        files.append(f"{run.name}/outputs/{f.name}")
    return TestClient(server.app), run, files, trashed


def test_list_outputs_endpoint(api):
    client, run, files, _ = api
    items = client.get("/api/outputs?limit=10").json()["files"]
    assert sorted(i["name"] for i in items) == sorted(files)
    assert all(i["run"] == run.name and i["prompt"] == "hello" for i in items)


def test_serve_output_and_load_run(api):
    client, run, files, _ = api
    assert client.get(f"/api/output/{files[0]}").status_code == 200
    wf = client.get(f"/api/runs/{run.name}").json()
    assert wf["seed"] == 0 and len(wf["outputs"]) == 2
    assert wf["outputs"][0]["url"].startswith(f"/api/output/{run.name}/outputs/")


@pytest.mark.parametrize("url", [
    "/api/output/..%2F..%2Fetc%2Fhosts", "/api/runs/..%2F..", "/api/runs/nope"])
def test_bad_paths_rejected(api, url):
    client, *_ = api
    assert client.get(url).status_code in (400, 404)


def test_delete_output_then_last_trashes_run(api):
    client, run, files, trashed = api
    assert client.delete(f"/api/output/{files[0]}").status_code == 200
    assert not (run.parent / files[0]).exists() and trashed == []
    assert client.delete(f"/api/output/{files[1]}").status_code == 200
    assert trashed == [run.name]


def test_upscale_gallery_output_lands_in_run(api, monkeypatch):
    client, run, files, _ = api
    import app as app_mod
    monkeypatch.setattr(app_mod, "get_available_devices", lambda: ["cpu"])
    monkeypatch.setattr(app_mod, "upscale_image",
                        lambda img, path, dev: img.resize((img.width * 4, img.height * 4)))
    r = client.post("/api/upscale/single", json={"source": "gallery", "filename": files[0], "model_path": "/m.pth"})
    assert r.status_code == 200, r.text
    up = json.loads((run / "workflow.json").read_text())["outputs"][-1]
    assert up["upscaled_from"] == files[0].split("/", 1)[1]
    assert up["file"].endswith("_32x24.png") and (run / up["file"]).is_file()
