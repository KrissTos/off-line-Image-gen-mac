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
    outputs = data.setdefault("outputs", [])
    entry: dict = {"file": Path(file_path).resolve().relative_to(run_dir.resolve()).as_posix(),
                   "kind": kind}
    if seed is None and upscaled_from:    # upscale keeps its source's seed, even if the source goes
        seed = next((o.get("seed") for o in outputs if o.get("file") == upscaled_from), None)
    if seed is not None:
        entry["seed"] = int(seed)
    if upscaled_from:
        entry["upscaled_from"] = upscaled_from
    # Same file again (e.g. upscaled twice at the same size) replaces its entry
    data["outputs"] = [o for o in outputs if o.get("file") != entry["file"]] + [entry]
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


# ── Migration: flat outputs + v1 workflows → v2 folders ────────────────────────

UPSCALE_RE = re.compile(r"^(?P<base>.+)_(?P<w>\d+)x(?P<h>\d+)$")
REF_SLOT_RE = re.compile(r"^ref_slot_(\d+)$")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _reserve(parent: Path, name: str, reserved: set[str]) -> str:
    cand, i = name, 2
    while cand in reserved or (parent / cand).exists():
        cand, i = f"{name}-{i}", i + 1
    reserved.add(cand)
    return cand


def _move_all(moves: list[tuple[Path, Path]]) -> None:
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _migrate_output(d: Path, media: Path, upscales: list[Path], reserved: set[str], apply: bool) -> dict:
    sidecar, companion = media.with_suffix(".json"), d / media.stem
    meta = _read_json(sidecar) if sidecar.is_file() else {}
    when = datetime.fromtimestamp(media.stat().st_mtime)
    slug = slugify(meta.get("prompt") or media.stem)
    name = _reserve(d, f"{when:%y%m%d-%H%M%S}_{slug}" if slug else f"{when:%y%m%d-%H%M%S}", reserved)
    run = d / name
    seed = meta.get("seed") if isinstance(meta.get("seed"), int) and meta["seed"] >= 0 else None
    ext = media.suffix.lower()
    new_stem = f"{name[:OUTPUT_STEM_MAX].rstrip('-_')}_s{seed}" if seed is not None else media.stem
    base_rel = f"outputs/{new_stem}{ext}"
    moves = [(media, run / base_rel)]
    outputs = [{"file": base_rel, "kind": "video" if ext in VIDEO_EXTS else "image",
                **({"seed": seed} if seed is not None else {})}]
    for up in upscales:
        m = UPSCALE_RE.match(up.stem)
        up_rel = f"outputs/{new_stem}_{m['w']}x{m['h']}{up.suffix.lower()}"
        moves.append((up, run / up_rel))
        outputs.append({"file": up_rel, "kind": "image", "upscaled_from": base_rel})
    ref_slots = []
    if companion.is_dir():
        refs = sorted((int(m[1]), p) for p in companion.iterdir()
                      if p.is_file() and (m := REF_SLOT_RE.match(p.stem)))
        mask = next((p for p in companion.iterdir() if p.is_file() and p.stem == "mask"), None)
        for n, p in refs:
            img_rel = f"refs/slot_{n}{p.suffix.lower()}"
            moves.append((p, run / img_rel))
            mask_rel = None
            if n == 1 and mask is not None:
                mask_rel = f"masks/slot_1{mask.suffix.lower()}"
                moves.append((mask, run / mask_rel))
            ref_slots.append({"image": img_rel, "mask": mask_rel,
                              "strength": float(meta.get("img_strength", 1.0)) if n == 1 else 1.0})
    data = {"version": VERSION, "name": name, "created": when.isoformat(timespec="seconds"),
            "prompt": meta.get("prompt", ""), **_params(meta), "ref_slots": ref_slots, "outputs": outputs}
    to_trash = [p for p in (sidecar, companion) if p.exists()]
    if apply:
        (run / "outputs").mkdir(parents=True)
        _move_all(moves)
        _write_json(run / "workflow.json", data)
        for p in to_trash:
            trash(p)
    return {"run": name, "moves": [[str(a), str(b)] for a, b in moves],
            "trash": [str(p) for p in to_trash],
            "consumed": [str(p) for p in (media, *upscales, *to_trash)]}


def _migrate_v1_workflow(folder: Path, apply: bool) -> dict:
    data = _read_json(folder / "workflow.json")
    moves, slots = [], []
    for i, s in enumerate(data.get("ref_slots") or [], start=1):
        new = dict(s)
        for key, sub in (("image", "refs"), ("mask", "masks")):
            rel = s.get(key)
            src = safe_join(folder, rel) if rel else None
            if src is not None and src.is_file() and not rel.startswith(f"{sub}/"):
                new[key] = f"{sub}/slot_{i}{src.suffix.lower()}"
                moves.append((src, folder / new[key]))
        slots.append(new)
    data.update({"version": VERSION, "name": folder.name, "ref_slots": slots})
    if apply:
        _move_all(moves)
        _write_json(folder / "workflow.json", data)
    return {"workflow": folder.name, "moves": [[str(a), str(b)] for a, b in moves]}


def migrate(directory, apply: bool = False) -> dict:
    """Group flat outputs into run folders and convert v1 workflow folders. Dry-run unless apply."""
    d = Path(directory)
    report: dict = {"runs": [], "workflows": [], "skipped": [], "unclassified": []}
    if not d.is_dir():
        return report
    entries = sorted(d.iterdir())
    consumed: set[Path] = set()
    for p in entries:
        if p.is_dir() and (p / "workflow.json").is_file():
            consumed.add(p)
            if _read_json(p / "workflow.json").get("version") == VERSION:
                report["skipped"].append(p.name)
            else:
                report["workflows"].append(_migrate_v1_workflow(p, apply))
    media = {p.stem: p for p in entries if p.is_file() and p.suffix.lower() in MEDIA_EXTS}
    upscales: dict[str, list[Path]] = {}
    bases: list[Path] = []
    for stem, p in media.items():
        m = UPSCALE_RE.match(stem)
        if m and m["base"] in media:
            upscales.setdefault(m["base"], []).append(p)
        else:
            bases.append(p)
    reserved: set[str] = set()
    for base in bases:
        plan = _migrate_output(d, base, sorted(upscales.get(base.stem, [])), reserved, apply)
        report["runs"].append(plan)
        consumed.update(Path(x) for x in plan["consumed"])
    report["unclassified"] = [p.name for p in entries if p not in consumed and not p.name.startswith(".")]
    return report


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m core.run_store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mig = sub.add_parser("migrate", help="Convert flat outputs / v1 workflows to v2 run folders")
    mig.add_argument("dir")
    mig.add_argument("--apply", action="store_true", help="Execute (default: dry run, changes nothing)")
    args = ap.parse_args(argv)
    report = migrate(args.dir, apply=args.apply)
    print("APPLIED" if args.apply else "DRY RUN (nothing changed; re-run with --apply)")
    for r in report["runs"]:
        print(f"RUN       {r['run']}  <- {len(r['moves'])} file(s), trash {len(r['trash'])}")
    for w in report["workflows"]:
        print(f"WORKFLOW  {w['workflow']}  ({len(w['moves'])} file(s) moved)")
    for name in report["skipped"]:
        print(f"SKIP      {name} (already v2)")
    for name in report["unclassified"]:
        print(f"LEFT      {name} (not recognised, untouched)")
    print(f"{len(report['runs'])} run(s), {len(report['workflows'])} workflow(s), "
          f"{len(report['skipped'])} skipped, {len(report['unclassified'])} left")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
