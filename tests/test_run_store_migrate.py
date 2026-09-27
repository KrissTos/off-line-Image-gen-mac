"""core/run_store.migrate: flat outputs and v1 workflows → v2 run folders."""
import json
import os
import shutil
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


def _legacy_tree(d: Path) -> None:
    # image + sidecar + companion folder + upscale
    _png(d / "20260927_edit_2.png")
    (d / "20260927_edit_2.json").write_text(json.dumps({
        "prompt": "Edit image 1", "seed": 42, "img_strength": 0.7, "model_choice": "M",
        "ref_image_count": 2, "has_mask": True}))
    _png(d / "20260927_edit_2" / "ref_slot_1.png")
    _png(d / "20260927_edit_2" / "ref_slot_2.png")
    _png(d / "20260927_edit_2" / "mask.png", "L", 255)
    (d / "20260927_edit_2" / "params.json").write_text("{}")
    _png(d / "20260927_edit_2_16x12.png")
    # image without sidecar
    _png(d / "lonely.png")
    # video + sidecar
    (d / "clip.mp4").write_bytes(b"\0\0")
    (d / "clip.json").write_text(json.dumps({"prompt": "A clip", "seed": 9}))
    # v1 saved workflow
    w = d / "26-01-01_old"
    _png(w / "slot_1_image.png")
    _png(w / "slot_1_mask.png", "L", 255)
    (w / "workflow.json").write_text(json.dumps({
        "name": "old", "timestamp": "x", "prompt": "p",
        "ref_slots": [{"image": "slot_1_image.png", "mask": "slot_1_mask.png", "strength": 0.9}]}))
    # something it can't classify
    (d / "notes.txt").write_text("hi")
    ts = NOW.timestamp()
    for p in (d / "20260927_edit_2.png", d / "20260927_edit_2_16x12.png", d / "lonely.png", d / "clip.mp4"):
        os.utime(p, (ts, ts))


@pytest.fixture
def trashed(monkeypatch):
    names = []

    def fake_trash(p):
        p = Path(p)
        names.append(p.name)
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    monkeypatch.setattr(run_store, "trash", fake_trash)
    return names


def test_migrate_dry_run_changes_nothing(tmp_path, trashed):
    _legacy_tree(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    report = run_store.migrate(tmp_path)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert {r["run"] for r in report["runs"]} == {
        "260927-151805_edit-image-1", "260927-151805_lonely", "260927-151805_a-clip"}
    assert [w["workflow"] for w in report["workflows"]] == ["26-01-01_old"]
    assert report["unclassified"] == ["notes.txt"]
    assert trashed == []


def test_migrate_apply_builds_runs(tmp_path, trashed):
    _legacy_tree(tmp_path)
    run_store.migrate(tmp_path, apply=True)

    run = tmp_path / "260927-151805_edit-image-1"
    data = json.loads((run / "workflow.json").read_text())
    assert data["version"] == 2 and data["seed"] == 42 and data["model_choice"] == "M"
    assert "ref_image_count" not in data
    assert data["ref_slots"] == [
        {"image": "refs/slot_1.png", "mask": "masks/slot_1.png", "strength": 0.7},
        {"image": "refs/slot_2.png", "mask": None, "strength": 1.0},
    ]
    base = "outputs/260927-151805_edit-image-1_s42.png"
    assert data["outputs"] == [
        {"file": base, "kind": "image", "seed": 42},
        {"file": "outputs/260927-151805_edit-image-1_s42_16x12.png", "kind": "image", "upscaled_from": base},
    ]
    assert (run / base).is_file() and (run / "refs" / "slot_2.png").is_file()
    assert sorted(trashed) == ["20260927_edit_2", "20260927_edit_2.json", "clip.json"]

    lonely = json.loads((tmp_path / "260927-151805_lonely" / "workflow.json").read_text())
    assert lonely["prompt"] == "" and lonely["ref_slots"] == []
    assert lonely["outputs"] == [{"file": "outputs/lonely.png", "kind": "image"}]

    clip = json.loads((tmp_path / "260927-151805_a-clip" / "workflow.json").read_text())
    assert clip["outputs"] == [{"file": "outputs/260927-151805_a-clip_s9.mp4", "kind": "video", "seed": 9}]

    wf = json.loads((tmp_path / "26-01-01_old" / "workflow.json").read_text())
    assert wf["version"] == 2 and wf["timestamp"] == "x" and wf["name"] == "26-01-01_old"
    assert wf["ref_slots"] == [{"image": "refs/slot_1.png", "mask": "masks/slot_1.png", "strength": 0.9}]
    assert (tmp_path / "26-01-01_old" / "refs" / "slot_1.png").is_file()

    assert (tmp_path / "notes.txt").is_file()
    assert len(run_store.list_outputs(tmp_path, limit=10)) == 4


def test_migrate_is_idempotent(tmp_path, trashed):
    _legacy_tree(tmp_path)
    run_store.migrate(tmp_path, apply=True)
    report = run_store.migrate(tmp_path, apply=True)
    assert report["runs"] == [] and report["workflows"] == []
    assert len(report["skipped"]) == 4
    assert report["unclassified"] == ["notes.txt"]


def test_cli_dry_run(tmp_path, capsys):
    _legacy_tree(tmp_path)
    assert run_store._main(["migrate", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "260927-151805_edit-image-1" in out
    assert (tmp_path / "20260927_edit_2.png").is_file()
