"""CivitAI LoRA discovery: which CivitAI LoRAs this app can load, as Model Sources rows.

Pure mapping and merge; the network is injected (`fetch_models`) so tests need no connection.
Downloads, the API key and the installed-file registry live in core/civitai_install.py.
"""
from __future__ import annotations

import html
import re

from core import model_sources

API_MODELS = "https://civitai.com/api/v1/models"
TOP_N = 100                       # per base model; the API's maximum page size
NSFW_LEVEL_MASK = 4 | 8 | 16      # version nsfwLevel bits: R, X, XXX

# CivitAI baseModel -> this app's LoRA family. Anything not listed is unusable here.
BASE_FAMILY = {
    "Flux.2 Klein 9B":      "klein-9B",
    "Flux.2 Klein 9B-base": "klein-9B",
    "Flux.2 Klein 4B":      "klein-4B",
    "Flux.2 Klein 4B-base": "klein-4B",
    "ZImageTurbo":          "Z-Image",
}


def family_of(base_model: str | None) -> str | None:
    return BASE_FAMILY.get(base_model or "")


def _safetensors_file(version: dict) -> dict | None:
    files = [f for f in version.get("files") or []
             if str(f.get("name", "")).lower().endswith(".safetensors") and f.get("type", "Model") == "Model"]
    files.sort(key=lambda f: not f.get("primary"))          # stable: primary first
    return files[0] if files else None


def _is_nsfw(model: dict, version: dict) -> bool:
    return bool(model.get("nsfw")) or bool((version.get("nsfwLevel") or 0) & NSFW_LEVEL_MASK)


def _plain(text: str | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text or "")).strip()


def rows_from_model(model: dict) -> list[dict]:
    """One row per family the model has a usable version for, from that family's newest version
    (CivitAI lists versions newest first)."""
    rows, seen = [], set()
    for v in model.get("modelVersions") or []:
        fam = family_of(v.get("baseModel"))
        f = _safetensors_file(v)
        if fam is None or f is None or fam in seen:
            continue
        seen.add(fam)
        desc = model_sources.describe(_plain(model.get("description")))
        rows.append({
            "id": f"civ-{model['id']}-{fam}",
            "name": model.get("name", ""),
            "type": "lora",
            "provider": "civitai",
            "url": f"https://civitai.com/models/{model['id']}",
            "family": fam,
            "description": desc,
            "function": model_sources.lora_function(model.get("name", ""), desc),
            "nsfw": _is_nsfw(model, v),
            "described": True,                  # enrich() must not try an HF card for these
            "civitai": {
                "modelId": model["id"], "versionId": v["id"], "fileId": f.get("id"),
                "file": f["name"], "sha256": ((f.get("hashes") or {}).get("SHA256") or "").lower(),
                "sizeKB": f.get("sizeKB"), "trained": list(v.get("trainedWords") or []),
            },
        })
    return rows


def discover(fetch_models, show_nsfw: bool, top_n: int = TOP_N) -> tuple[list[dict], list[str]]:
    """(rows, failed base models). `fetch_models(base_model, limit)` returns model dicts or raises."""
    by_id: dict[str, dict] = {}
    failed: list[str] = []
    for base in BASE_FAMILY:
        try:
            models = fetch_models(base, top_n)
        except Exception:
            failed.append(base)
            continue
        for m in models:
            for row in rows_from_model(m):
                if row["nsfw"] and not show_nsfw:
                    continue
                by_id.setdefault(row["id"], row)
    return list(by_id.values()), failed


def merge_rows(current: list[dict], fresh: list[dict], failed_families: set[str],
               installed_keys: set[tuple]) -> tuple[list[dict], int]:
    """Replace the CivitAI rows with the fresh ones. An old row survives only when its family could
    not be fetched or its file is installed (so it stays deletable). Returns (rows, added)."""
    old = {s["id"]: s for s in current if s.get("provider") == "civitai"}
    fresh_ids = {r["id"] for r in fresh}
    keep = [s for sid, s in old.items() if sid not in fresh_ids and (
        s.get("family") in failed_families
        or ((s.get("civitai") or {}).get("modelId"), s.get("family")) in installed_keys)]
    others = [s for s in current if s.get("provider") != "civitai"]
    added = sum(1 for r in fresh if r["id"] not in old)
    return others + fresh + keep, added


def annotate(rows: list[dict], installed: dict, show_nsfw: bool) -> list[dict]:
    """Add `installed` / `update` / `installed_version` to CivitAI rows; hide NSFW rows unless the
    toggle is on or the file is installed. HF rows pass through untouched."""
    out = []
    for s in rows:
        if s.get("provider") != "civitai":
            out.append(s)
            continue
        c = s.get("civitai") or {}
        have = installed.get((c.get("modelId"), s.get("family")))
        if s.get("nsfw") and not show_nsfw and have is None:
            continue
        s = dict(s)
        s["installed"] = have is not None
        s["update"] = have is not None and have.get("versionId") != c.get("versionId")
        s["installed_version"] = have.get("versionId") if have else None
        out.append(s)
    return out


def http_fetch_models(base_model: str, limit: int = TOP_N) -> list[dict]:
    """Top LoRAs for one base model, anonymous (listing needs no key). Raises on any failure."""
    import requests
    r = requests.get(API_MODELS, params={
        "types": "LORA", "baseModels": base_model, "sort": "Most Downloaded",
        "period": "AllTime", "limit": limit}, timeout=30)
    r.raise_for_status()
    items = r.json().get("items")
    if not isinstance(items, list):
        raise ValueError("unexpected CivitAI response")
    return items
