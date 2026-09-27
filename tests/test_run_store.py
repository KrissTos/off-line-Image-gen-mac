"""core/run_store: run folder naming, creation, outputs and loading."""
import json
import os
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


def _run_with_outputs(base, prompt, seeds, now):
    run = run_store.create_run(base, {"prompt": prompt, "model_choice": "M"}, [], now=now)
    for s in seeds:
        f = _png(run_store.output_filename(run, s, ".png"))
        run_store.add_output(run, f, "image", seed=s)
    return run


def test_list_outputs_newest_first_with_params(tmp_path):
    a = _run_with_outputs(tmp_path, "first", [1], NOW)
    b = _run_with_outputs(tmp_path, "second", [2, 3], NOW.replace(second=9))
    os.utime(a / "outputs" / f"{a.name}_s1.png", (1000, 1000))
    os.utime(b / "outputs" / f"{b.name}_s2.png", (2000, 2000))
    os.utime(b / "outputs" / f"{b.name}_s3.png", (3000, 3000))
    items = run_store.list_outputs(tmp_path, limit=2)
    assert [i["seed"] for i in items] == [3, 2]
    it = items[0]
    assert it["name"] == f"{b.name}/outputs/{b.name}_s3.png"
    assert it["url"] == f"/api/output/{it['name']}"
    assert it["run"] == b.name and it["file"] == f"outputs/{b.name}_s3.png"
    assert it["prompt"] == "second" and it["model_choice"] == "M" and it["kind"] == "image"


def test_list_outputs_skips_missing_files_and_bad_json(tmp_path):
    run = _run_with_outputs(tmp_path, "x", [1], NOW)
    next((run / "outputs").glob("*.png")).unlink()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "workflow.json").write_text("{")
    assert run_store.list_outputs(tmp_path) == []


def test_upscale_inherits_source_seed(tmp_path):
    run = _run_with_outputs(tmp_path, "x", [5], NOW)
    base = next((run / "outputs").glob("*.png"))
    up = _png(base.with_name(base.stem + "_16x12.png"))
    run_store.add_output(run, up, "image", upscaled_from=f"outputs/{base.name}")
    items = {i["file"]: i for i in run_store.list_outputs(tmp_path)}
    assert items[f"outputs/{up.name}"]["seed"] == 5


def test_remove_output_reports_empty_run(tmp_path):
    run = _run_with_outputs(tmp_path, "x", [1, 2], NOW)
    files = [o["file"] for o in json.loads((run / "workflow.json").read_text())["outputs"]]
    assert run_store.remove_output(run, files[0]) is False
    assert not (run / files[0]).exists()
    assert run_store.remove_output(run, files[1]) is True


def test_remove_output_rejects_traversal(tmp_path):
    run = _run_with_outputs(tmp_path, "x", [1], NOW)
    with pytest.raises(ValueError):
        run_store.remove_output(run, "../../etc/passwd")


def test_save_workflow_new_and_collision(tmp_path, src):
    slots = [{"image_path": src["img"], "mask_path": src["mask"], "strength": 1.0}]
    name = run_store.save_workflow(tmp_path, {"prompt": "p"}, slots, "bathroom floor", now=NOW)
    assert name == "26-09-27_bathroom_floor"
    data = json.loads((tmp_path / name / "workflow.json").read_text())
    assert data["version"] == 2 and data["name"] == name and "outputs" not in data
    assert data["ref_slots"] == [{"image": "refs/slot_1.png", "mask": "masks/slot_1.png", "strength": 1.0}]
    assert run_store.save_workflow(tmp_path, {}, [], "bathroom floor", now=NOW) == "26-09-27_bathroom_floor-2"


def test_save_workflow_overwrite_keeps_unknown_keys_and_drops_stale_files(tmp_path, src):
    name = run_store.save_workflow(tmp_path, {"prompt": "old"},
                                   [{"image_path": src["img"], "mask_path": src["mask"], "strength": 1.0}],
                                   "job", now=NOW)
    wf = tmp_path / name
    data = json.loads((wf / "workflow.json").read_text())
    data["job_id"] = "abc"
    (wf / "workflow.json").write_text(json.dumps(data))
    out = run_store.save_workflow(tmp_path, {"prompt": "new"},
                                  [{"image_path": src["jpg"], "mask_path": None, "strength": 0.5}],
                                  "ignored", overwrite=name, now=NOW)
    assert out == name
    data = json.loads((wf / "workflow.json").read_text())
    assert data["prompt"] == "new" and data["job_id"] == "abc"
    assert data["ref_slots"] == [{"image": "refs/slot_1.jpg", "mask": None, "strength": 0.5}]
    assert not (wf / "refs" / "slot_1.png").exists()
    assert not (wf / "masks" / "slot_1.png").exists()


@pytest.mark.parametrize("bad", ["../x", "missing", "."])
def test_save_workflow_overwrite_rejects_bad_target(tmp_path, bad):
    (tmp_path / "wf").mkdir()
    with pytest.raises(ValueError):
        run_store.save_workflow(tmp_path / "wf", {}, [], "n", overwrite=bad)
