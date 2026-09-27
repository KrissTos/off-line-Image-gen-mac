"""/api/generate and /api/batch/generate write one run folder per generation."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


class FakeManager:
    is_busy = False
    stop_requested = False
    is_batch_running = False

    def __init__(self, seeds=(111, 222)):
        self.seeds = seeds

    async def generate(self, params):
        out = Path(params["output_dir"])
        for s in self.seeds:
            p = out / f"20260927_tmp_{s}.png"
            Image.new("RGB", (8, 6)).save(p)
            yield {"type": "image", "url": f"/api/output/{p.name}", "path": str(p),
                   "info": f"Seed: {s} | Model: x"}


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


@pytest.fixture
def api(monkeypatch, tmp_path):
    import server
    from core import run_store
    out, uploads = tmp_path / "out", tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(server, "TEMP_DIR", uploads)
    monkeypatch.setattr(server, "_output_dir", lambda: str(out))
    trashed = []
    monkeypatch.setattr(run_store, "trash", lambda p: trashed.append(Path(p).name))
    Image.new("RGB", (8, 6)).save(uploads / "img.png")
    Image.new("L", (8, 6), 255).save(uploads / "mask.png")

    def client_with(mgr):
        monkeypatch.setattr(server, "_mgr", lambda: mgr)
        return TestClient(server.app)    # no `with`: startup (heartbeat) not run

    return client_with, out, trashed


def _only_run(out: Path) -> Path:
    runs = [p for p in out.iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def test_generate_writes_one_run_with_outputs(api):
    client_with, out, _ = api
    r = client_with(FakeManager()).post("/api/generate", json={
        "prompt": "Hello", "seed": -1, "input_image_ids": ["img.png"], "mask_image_id": "mask.png",
        "ref_slots": [{"imageId": "img.png", "maskId": "mask.png", "strength": 0.6}]})
    events = _events(r.text)
    run = _only_run(out)
    assert run.name.endswith("_hello")
    data = json.loads((run / "workflow.json").read_text())
    assert [o["seed"] for o in data["outputs"]] == [111, 222]
    assert data["ref_slots"] == [{"image": "refs/slot_1.png", "mask": "masks/slot_1.png", "strength": 0.6}]
    images = [e for e in events if e["type"] == "image"]
    for e, o in zip(images, data["outputs"]):
        assert Path(e["path"]) == run / o["file"] and Path(e["path"]).is_file()
        assert e["url"] == f"/api/output/{run.name}/{o['file']}"
    assert events[-1] == {"type": "done"}


def test_generate_without_ref_slots_derives_them(api):
    client_with, out, _ = api
    client_with(FakeManager(seeds=(5,))).post("/api/generate", json={
        "prompt": "p", "input_image_ids": ["img.png"], "mask_image_id": "mask.png", "img_strength": 0.4})
    data = json.loads((_only_run(out) / "workflow.json").read_text())
    assert data["ref_slots"] == [{"image": "refs/slot_1.png", "mask": "masks/slot_1.png", "strength": 0.4}]


def test_generate_with_no_output_trashes_the_run(api):
    client_with, out, trashed = api
    client_with(FakeManager(seeds=())).post("/api/generate", json={"prompt": "nothing"})
    assert len(trashed) == 1 and trashed[0].endswith("_nothing")


def test_generate_ignores_temp_ids_outside_temp_dir(api):
    client_with, out, _ = api
    client_with(FakeManager(seeds=(1,))).post("/api/generate", json={
        "prompt": "p", "ref_slots": [{"imageId": "../../etc/hosts", "maskId": None, "strength": 1.0}]})
    assert json.loads((_only_run(out) / "workflow.json").read_text())["ref_slots"] == []


def test_batch_generate_one_run_per_image(api, tmp_path):
    client_with, out, _ = api
    folder = tmp_path / "batch"
    folder.mkdir()
    for n in ("a.png", "b.png"):
        Image.new("RGB", (8, 6)).save(folder / n)
    client_with(FakeManager(seeds=(3,))).post("/api/batch/generate", json={
        "prompt": "batch", "input_folder": str(folder)})
    runs = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(runs) == 2
    for run in runs:
        data = json.loads((run / "workflow.json").read_text())
        assert len(data["outputs"]) == 1 and data["ref_slots"][0]["image"] == "refs/slot_1.png"
