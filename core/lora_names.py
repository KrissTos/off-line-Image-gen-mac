"""Friendly LoRA names for the dropdown: no extension, underscores as spaces, the CivitAI model name
when the file came from CivitAI. The file name stays the identity (paths, workflows); this is display only."""
from __future__ import annotations

import re
from pathlib import Path

_EXT = re.compile(r"\.(safetensors|pt|bin)$", re.I)


def clean_name(file_name: str) -> str:
    stem = _EXT.sub("", Path(file_name).name)
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip(" .") or file_name


def display_name(file_name: str, registry: dict | None = None) -> str:
    """CivitAI model name from the download registry, else the cleaned file name."""
    if registry is None:
        from core.civitai_install import load_registry
        registry = load_registry()
    given = str((registry.get(file_name) or {}).get("name") or "").strip()
    return given or clean_name(file_name)
