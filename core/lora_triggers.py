"""Trigger-word hints for LoRAs.

A LoRA file does not carry its trigger (ai-toolkit stores none; a few keep a lone caption tag), so
known LoRAs are seeded here by file name (from each model's page) and the user can override or add
one per file. Overrides live in lora_triggers.json (runtime data, gitignored).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

STORE_PATH = Path(__file__).resolve().parent.parent / "lora_triggers.json"

_HF = "https://huggingface.co/"
_BFS_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and "
    "background. remove the head from Picture 1 completely and replace it with the head from "
    "Picture 2, strictly preserving the hair, eye color, and nose structure of Picture 2. copy the "
    "direction of the eye, head rotation, and micro-expressions from Picture 1. Keep the body, "
    "clothing, pose, framing, and background intact."
)
_BFS_NOTE = ("Image 1 = target body, image 2 = the head to use (don't reverse). "
             "Start at strength 1.0.")

# file name -> {trigger, note, source}. trigger None = the model needs none.
SEED: dict[str, dict] = {
    "realistic.safetensors": {
        "trigger": "realistic",
        "note": 'Use as "Turn XXX image into a realistic photograph".',
        "source": _HF + "dx8152/Flux2-Klein-9B-Enhanced-Details",
    },
    "Klein-consistency.safetensors": {
        "trigger": None,
        "note": "No trigger needed: it improves edit consistency on its own.",
        "source": _HF + "dx8152/Flux2-Klein-9B-Consistency",
    },
    "dever_devil_may_cry_flux2_klein_9b.safetensors": {
        "trigger": "dmc_style",
        "note": 'Works for text-to-image and edits ("Transform into dmc_style"). '
                "Avoid camera names in the prompt.",
        "source": _HF + "DeverStyle/Flux.2-Klein-Loras",
    },
    "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors": {
        "trigger": _BFS_PROMPT, "note": _BFS_NOTE,
        "source": _HF + "Alissonerdx/BFS-Best-Face-Swap",
    },
    "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors": {
        "trigger": _BFS_PROMPT, "note": _BFS_NOTE,
        "source": _HF + "Alissonerdx/BFS-Best-Face-Swap",
    },
    "Eyes_direction_Lora_Flux2Klein_9B_v1.safetensors": {
        "trigger": "change the eyes to match the reference dot direction",
        "note": "Needs the red-dot control image as ref #2. Strength 0.5-1 realistic, "
                "1.25-1.5 anime / non-realistic.",
        "source": _HF + "eric-venti-seeds/Eyes_Direction_Lora_Flux2Klein9B",
    },
}


def _load_overrides() -> dict:
    try:
        data = json.loads(STORE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _metadata_tag(path: str) -> str | None:
    """The one caption tag some ai-toolkit LoRAs keep, only when it is unambiguous."""
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            raw = (f.metadata() or {}).get("ss_tag_frequency")
        tags = {t for folder in json.loads(raw).values() for t in folder} if raw else set()
    except Exception:
        return None
    return next(iter(tags)) if len(tags) == 1 else None


def get_trigger_info(path: str) -> dict:
    """{trigger, note, source, origin} for a LoRA file; origin is user|seed|metadata|none."""
    name = os.path.basename(path)
    seed = SEED.get(name, {})
    user = _load_overrides().get(name)
    if user:
        return {"trigger": user, "note": seed.get("note", ""),
                "source": seed.get("source", ""), "origin": "user"}
    if seed:
        return {"trigger": seed["trigger"], "note": seed["note"],
                "source": seed["source"], "origin": "seed"}
    tag = _metadata_tag(path)
    if tag:
        return {"trigger": tag, "note": "", "source": "", "origin": "metadata"}
    return {"trigger": None, "note": "", "source": "", "origin": "none"}


def set_trigger(name: str, trigger: str) -> None:
    """Save (or, with an empty string, clear) the user's trigger for a LoRA file name."""
    if name != os.path.basename(name) or not name:
        raise ValueError("LoRA name must be a plain file name")
    data = _load_overrides()
    trigger = trigger.strip()
    if trigger:
        data[name] = trigger
    else:
        data.pop(name, None)
    STORE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
