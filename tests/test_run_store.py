"""core/run_store: run folder naming, creation, outputs and loading."""
import json
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from core import run_store

NOW = datetime(2026, 9, 27, 15, 18, 5)


def _png(path: Path, mode: str = "RGB", color="white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (8, 6), color).save(path)
    return path


@pytest.fixture
def src(tmp_path):
    return {
        "img":  _png(tmp_path / "up" / "a.png"),
        "jpg":  _png(tmp_path / "up" / "b.jpg"),
        "mask": _png(tmp_path / "up" / "m.png", "L", 255),
    }


def test_slugify():
    assert run_store.slugify("Edit image 1: a Photo of!! the wall") == "edit-image-1-a-photo-of-the"
    assert run_store.slugify("!!!") == ""
    assert run_store.slugify("") == ""


def test_run_name_without_slug_and_collision(tmp_path):
    assert run_store.new_run_name(tmp_path, "", NOW) == "260927-151805"
    (tmp_path / "260927-151805_hello").mkdir()
    assert run_store.new_run_name(tmp_path, "hello", NOW) == "260927-151805_hello-2"


def test_create_run_copies_slots_and_writes_v2(tmp_path, src):
    params = {"prompt": "Hello world", "seed": -1, "width": 64, "height": 48, "input_image_ids": ["x"]}
    slots = [{"image_path": src["img"], "mask_path": src["mask"], "strength": 0.8},
             {"image_path": src["jpg"], "mask_path": None, "strength": 1.0}]
    run = run_store.create_run(tmp_path / "out", params, slots, now=NOW)
    assert run.name == "260927-151805_hello-world"
    data = json.loads((run / "workflow.json").read_text())
    assert data["version"] == 2 and data["created"] == "2026-09-27T15:18:05"
    assert data["name"] == run.name
    assert data["prompt"] == "Hello world" and data["seed"] == -1
    assert "input_image_ids" not in data
    assert data["ref_slots"] == [
        {"image": "refs/slot_1.png", "mask": "masks/slot_1.png", "strength": 0.8},
        {"image": "refs/slot_2.jpg", "mask": None, "strength": 1.0},
    ]
    assert data["outputs"] == []
    assert (run / "refs/slot_1.png").is_file() and (run / "masks/slot_1.png").is_file()
    assert (run / "outputs").is_dir()


def test_output_filename_and_add_output(tmp_path):
    run = run_store.create_run(tmp_path, {"prompt": "hello"}, [], now=NOW)
    f = run_store.output_filename(run, 812345, ".png")
    assert f == run / "outputs" / "260927-151805_hello_s812345.png"
    _png(f)
    assert run_store.output_filename(run, 812345, ".png").name == "260927-151805_hello_s812345-2.png"
    assert run_store.output_filename(run, None, ".mp4").name == "260927-151805_hello.mp4"
    entry = run_store.add_output(run, f, "image", seed=812345)
    assert entry == {"file": "outputs/260927-151805_hello_s812345.png", "kind": "image", "seed": 812345}
    assert json.loads((run / "workflow.json").read_text())["outputs"] == [entry]


def test_long_prompt_output_stem_is_truncated(tmp_path):
    run = run_store.create_run(tmp_path, {"prompt": "a" * 80}, [], now=NOW)
    assert run_store.output_filename(run, 1, ".png").name == run.name[:40] + "_s1.png"


def test_add_output_keeps_unknown_keys(tmp_path):
    run = run_store.create_run(tmp_path, {"prompt": "x"}, [], now=NOW)
    data = json.loads((run / "workflow.json").read_text())
    data["job"] = "bagno-1"
    (run / "workflow.json").write_text(json.dumps(data))
    run_store.add_output(run, _png(run / "outputs" / "o.png"), "image")
    assert json.loads((run / "workflow.json").read_text())["job"] == "bagno-1"


def test_load_builds_urls_and_skips_missing(tmp_path, src):
    run = run_store.create_run(tmp_path, {"prompt": "x", "seed": 0}, [
        {"image_path": src["img"], "mask_path": src["mask"], "strength": 0.5},
        {"image_path": src["jpg"], "mask_path": None, "strength": 1.0},
    ], now=NOW)
    (run / "refs/slot_2.jpg").unlink()
    run_store.add_output(run, _png(run / "outputs" / "o.png"), "image", seed=7)
    wf = run_store.load(run, f"/api/output/{run.name}")
    assert wf["seed"] == 0
    assert wf["ref_slots"] == [{
        "imageUrl": f"/api/output/{run.name}/refs/slot_1.png",
        "maskUrl":  f"/api/output/{run.name}/masks/slot_1.png",
        "strength": 0.5,
    }]
    assert len(wf["warnings"]) == 1 and "slot 2" in wf["warnings"][0]
    assert wf["outputs"][0]["url"] == f"/api/output/{run.name}/outputs/o.png"


def test_load_rejects_paths_escaping_the_folder(tmp_path):
    run = run_store.create_run(tmp_path / "out", {"prompt": "x"}, [], now=NOW)
    _png(tmp_path / "secret.png")
    data = json.loads((run / "workflow.json").read_text())
    data["ref_slots"] = [{"image": "../../secret.png", "mask": None, "strength": 1.0}]
    (run / "workflow.json").write_text(json.dumps(data))
    assert run_store.load(run, "/x")["ref_slots"] == []


def test_load_malformed_json_raises(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "workflow.json").write_text("{nope")
    with pytest.raises(ValueError):
        run_store.load(tmp_path / "bad", "/x")


def test_load_v1_legacy_lora(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "workflow.json").write_text(json.dumps(
        {"prompt": "p", "lora_file": "/l.safetensors", "lora_strength": 0.7}))
    assert run_store.load(tmp_path / "w", "/x")["lora_files"] == [{"path": "/l.safetensors", "strength": 0.7}]
