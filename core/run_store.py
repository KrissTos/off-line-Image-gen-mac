"""Run folders: one self-contained folder per generation.

Spec: docs/superpowers/specs/2026-09-27-run-folders-design.md. Pure file logic, no FastAPI.
A run folder holds workflow.json (version 2), refs/, masks/ and outputs/; a saved workflow is
the same folder without outputs. Paths inside workflow.json are relative to the folder.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

VERSION = 2
MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov"}
VIDEO_EXTS = {".mp4", ".webm", ".mov"}
OUTPUT_STEM_MAX = 40
PARAM_KEYS = (
    "prompt", "model_choice", "model_source", "width", "height", "steps", "seed", "guidance",
    "device", "img_strength", "repeat_count", "lora_files", "upscale_enabled",
    "upscale_model_path", "num_frames", "fps", "fast_preview", "mask_mode", "outpaint_align",
)


def slugify(text: str, n: int = 30) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "")[:n].lower()).strip("-")


def _unique_name(parent: Path, name: str, ext: str = "") -> Path:
    cand, i = parent / f"{name}{ext}", 2
    while cand.exists():
        cand, i = parent / f"{name}-{i}{ext}", i + 1
    return cand


def new_run_name(base_dir, prompt: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%y%m%d-%H%M%S")
    slug = slugify(prompt)
    return _unique_name(Path(base_dir), f"{stamp}_{slug}" if slug else stamp).name


def safe_join(folder, rel: str) -> Path | None:
    """Resolve `rel` inside `folder`; None when it escapes the folder."""
    try:
        base = Path(folder).resolve()
        p = (base / rel).resolve()
    except (OSError, ValueError):
        return None
    return p if p.is_relative_to(base) else None


def trash(path) -> None:
    """Move to the macOS Trash (never rm -rf)."""
    subprocess.run(["/usr/bin/trash", str(path)], check=True)


def read_workflow(folder) -> dict:
    with open(Path(folder) / "workflow.json") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("workflow.json is not a JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".workflow-", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _params(params: dict) -> dict:
    return {k: params[k] for k in PARAM_KEYS if k in params}


def _copy_slots(folder: Path, slots: list[dict]) -> list[dict]:
    """Copy slot sources into refs/ and masks/; return the v2 ref_slots entries."""
    out = []
    for i, s in enumerate(slots, start=1):
        img = Path(s["image_path"])
        img_rel = f"refs/slot_{i}{img.suffix.lower() or '.png'}"
        (folder / "refs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, folder / img_rel)
        mask_rel = None
        if s.get("mask_path"):
            mask = Path(s["mask_path"])
            mask_rel = f"masks/slot_{i}{mask.suffix.lower() or '.png'}"
            (folder / "masks").mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask, folder / mask_rel)
        out.append({"image": img_rel, "mask": mask_rel, "strength": float(s.get("strength", 1.0))})
    return out


def create_run(base_dir, params: dict, slots: list[dict], now: datetime | None = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    folder = base / new_run_name(base, params.get("prompt", ""), now)
    (folder / "outputs").mkdir(parents=True)
    data = {
        "version": VERSION, "name": folder.name, "created": now.isoformat(timespec="seconds"),
        **_params(params), "ref_slots": _copy_slots(folder, slots), "outputs": [],
    }
    _write_json(folder / "workflow.json", data)
    return folder


def output_filename(run_dir, seed: int | None, ext: str) -> Path:
    run_dir = Path(run_dir)
    stem = run_dir.name[:OUTPUT_STEM_MAX].rstrip("-_")
    return _unique_name(run_dir / "outputs", f"{stem}_s{seed}" if seed is not None else stem, ext)


def add_output(run_dir, file_path, kind: str, seed: int | None = None,
               upscaled_from: str | None = None) -> dict:
    run_dir = Path(run_dir)
    data = read_workflow(run_dir)
    entry: dict = {"file": Path(file_path).resolve().relative_to(run_dir.resolve()).as_posix(),
                   "kind": kind}
    if seed is not None:
        entry["seed"] = int(seed)
    if upscaled_from:
        entry["upscaled_from"] = upscaled_from
    data.setdefault("outputs", []).append(entry)
    _write_json(run_dir / "workflow.json", data)
    return entry


def load(folder, url_prefix: str) -> dict:
    """Workflow dict for the UI: slot/output URLs under url_prefix, missing files skipped."""
    folder = Path(folder)
    data = read_workflow(folder)
    slots, warnings = [], []
    for i, s in enumerate(data.get("ref_slots") or [], start=1):
        img_rel = s.get("image") or ""
        img = safe_join(folder, img_rel) if img_rel else None
        if img is None or not img.is_file():
            warnings.append(f"slot {i}: image missing ({img_rel})")
            continue
        mask_url = None
        if s.get("mask"):
            mask = safe_join(folder, s["mask"])
            if mask is not None and mask.is_file():
                mask_url = f"{url_prefix}/{s['mask']}"
            else:
                warnings.append(f"slot {i}: mask missing ({s['mask']})")
        slots.append({"imageUrl": f"{url_prefix}/{img_rel}", "maskUrl": mask_url,
                      "strength": float(s.get("strength", 1.0))})
    out = dict(data)
    out["ref_slots"] = slots
    out["outputs"] = [dict(o, url=f"{url_prefix}/{o['file']}")
                      for o in data.get("outputs") or []
                      if o.get("file") and safe_join(folder, o["file"]) is not None]
    if not out.get("lora_files") and out.get("lora_file"):
        out["lora_files"] = [{"path": out["lora_file"], "strength": out.get("lora_strength", 1.0)}]
    out["warnings"] = warnings
    for w in warnings:
        print(f"[run_store] {folder.name}: {w}")
    return out


def list_outputs(base_dir, limit: int = 20) -> list[dict]:
    """Every output of every run under base_dir, newest file first."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    items = []
    for wf in base.glob("*/workflow.json"):
        run = wf.parent
        try:
            data = read_workflow(run)
        except (OSError, ValueError):
            continue
        outputs = data.get("outputs") or []
        scalars = _params(data)
        by_file = {o.get("file"): o for o in outputs}
        for o in outputs:
            f = safe_join(run, o.get("file") or "")
            if f is None or not f.is_file():
                continue
            seed = o.get("seed", (by_file.get(o.get("upscaled_from")) or {}).get("seed"))
            rel = f"{run.name}/{o['file']}"
            items.append({**scalars, "name": rel, "url": f"/api/output/{rel}",
                          "mtime": f.stat().st_mtime, "kind": o.get("kind", "image"),
                          "run": run.name, "file": o["file"], "seed": seed})
    items.sort(key=lambda d: d["mtime"], reverse=True)
    return items[:limit]


def remove_output(run_dir, file_rel: str) -> bool:
    """Delete one output file and its entry. True when the run has no outputs left."""
    run_dir = Path(run_dir)
    f = safe_join(run_dir, file_rel)
    if f is None or f == run_dir.resolve():
        raise ValueError(f"Invalid output path: {file_rel}")
    if f.is_file():
        f.unlink()
    data = read_workflow(run_dir)
    data["outputs"] = [o for o in data.get("outputs") or [] if o.get("file") != file_rel]
    _write_json(run_dir / "workflow.json", data)
    return not data["outputs"]


def save_workflow(base_dir, params: dict, slots: list[dict], name: str,
                  overwrite: str | None = None, now: datetime | None = None) -> str:
    """Write a saved workflow (no outputs). overwrite = existing folder name to rewrite in place."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    if overwrite:
        folder = safe_join(base, overwrite)
        if folder is None or folder == base.resolve() or not (folder / "workflow.json").is_file():
            raise ValueError(f"Workflow not found: {overwrite}")
        data = read_workflow(folder)
    else:
        custom = (name or "").strip().replace(" ", "_").replace("/", "_")
        stamp = now.strftime("%y-%m-%d")
        folder = _unique_name(base, f"{stamp}_{custom}" if custom else stamp)
        folder.mkdir()
        data = {}
    old = {p for sub in ("refs", "masks") if (folder / sub).is_dir()
           for p in (folder / sub).iterdir() if p.is_file()}
    ref_slots = _copy_slots(folder, slots)
    keep = {folder / s["image"] for s in ref_slots} | {folder / s["mask"] for s in ref_slots if s["mask"]}
    for p in old - keep:
        p.unlink(missing_ok=True)
    data.pop("outputs", None)
    data.update({"version": VERSION, "name": folder.name, "saved": now.isoformat(timespec="seconds"),
                 **_params(params), "ref_slots": ref_slots})
    _write_json(folder / "workflow.json", data)
    return folder.name
