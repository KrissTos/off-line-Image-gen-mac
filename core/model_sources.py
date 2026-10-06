"""Model Sources: what is good for this app, and what each LoRA does.

Pure helpers used by `server.py` when the list is read and when "Update" discovers new repos.
Network access is injected (`fetch_card`, `fetch_files`) so everything here is unit-testable;
`hf_fetch_card` / `hf_fetch_files` are the real implementations.
"""
import re
from concurrent.futures import ThreadPoolExecutor, wait

DESC_MAX = 140
ENRICH_DEADLINE_S = 25
WORKERS = 8

# Keyword → function tag, first match wins (checked on the repo name, then on the description).
_FUNCTIONS = (
    ("face",     ("eyes", "eye", "face", "portrait", "head", "lip", "skin")),
    ("camera",   ("camera", "dolly", "jib", "zoom", "pan", "orbit", "angle")),
    ("lighting", ("relight", "delight", "light", "hdr")),
    ("edit",     ("outpaint", "inpaint", "edit", "migration", "consistency", "tryon", "transfer", "swap", "try-on")),
    ("detail",   ("detail", "detailer", "enhanced", "sharp", "deblur", "restore", "quality", "upscal")),
    ("speed",    ("turbo", "distill", "lightning", "flash", "fast")),
    ("realism",  ("realistic", "realism", "photoreal", "photograph", "real")),
    ("style",    ("style", "anime", "cartoon", "art", "watercolor", "comic")),
    ("control",  ("canny", "depth", "pose", "control")),
    ("effect",   ("effect", "cakeify", "squish")),
)
_UPSCALER_TOKENS = {"esrgan", "realesrgan", "drct", "swinir", "hat", "plksr", "atd", "remacri",
                    "ultrasharp", "gyre"}
_UPSCALER_PREFIXES = ("upscal", "4x", "2x", "8x")
_UPSCALER_EXT = (".safetensors", ".pth", ".pt")


def repo_id(url: str) -> str:
    return url.split("huggingface.co/", 1)[-1].strip("/")


def lora_family(name: str) -> str | None:
    """Which of this app's models a LoRA is for, or None when the app cannot load it.

    The app runs FLUX.2-klein (4B / 9B), Z-Image and LTX-Video 0.9.x; LoRAs for LTX-2.x, FLUX.1,
    FLUX.2-dev, SDXL, Qwen, Krea, Wan... are not usable here."""
    n = (name or "").lower()
    if re.search(r"ltx-?2", n):
        return None                                   # LTX-2.x generation: different pipeline
    if re.search(r"ltx-?video|ltxv", n):
        return "LTX-Video"
    if re.search(r"z-?image", n):
        return "Z-Image"
    if "klein" in n:
        if "9b" in n:
            return "klein-9B"
        if "4b" in n:
            return "klein-4B"
        return "klein"
    return None


def _has_word(text: str, kw: str) -> bool:
    return kw in re.split(r"[^a-z0-9]+", text) if kw.isalnum() and len(kw) <= 3 else kw in text


def lora_function(name: str, text: str = "") -> str:
    """Short tag for what the LoRA does; name first, then the description."""
    for source in ((name or "").lower(), (text or "").lower()):
        if not source:
            continue
        for tag, kws in _FUNCTIONS:
            if any(_has_word(source, kw) for kw in kws):
                return tag
    return "other"


_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TAG = re.compile(r"<[^>]*>")
_SENTENCE = re.compile(r"(.+?[.!?])(?:\s|$)")


def describe(card_text: str) -> str:
    """First real sentence of a model card, <= DESC_MAX chars. '' when the card has no prose."""
    text = card_text or ""
    if text.lstrip().startswith("---"):
        parts = text.lstrip().split("---", 2)
        text = parts[2] if len(parts) == 3 else ""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    in_code = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line:
            continue
        line = _TAG.sub(" ", _MD_LINK.sub(r"\1", re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line))).strip()
        if not line or line.startswith(("#", "|", "-", "*", ">")) and not line.startswith("**"):
            continue
        if re.match(r"^\W*trigger", line, re.I) or re.search(r"https?://|www\.", line):
            continue
        line = re.sub(r"[*`_]+", "", line).strip()
        if not line:
            continue
        m = _SENTENCE.match(line)
        sentence = m.group(1) if m else line
        if len(sentence) > DESC_MAX:
            sentence = sentence[:DESC_MAX - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        return sentence
    return ""


def _upscaler_name_ok(name: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]
    return any(t in _UPSCALER_TOKENS or t.startswith(_UPSCALER_PREFIXES) or "nomos" in t for t in tokens)


def is_supported(type_: str, name: str, files) -> bool | None:
    """True/False from the repo's file list; None when it could not be read. Base models are
    decided elsewhere (app.KNOWN_MODELS) and always pass here."""
    if type_ == "base":
        return True
    if files is None:
        return None
    names = [f.lower() for f in files]
    if type_ == "lora":
        return lora_family(name) is not None and any(f.endswith(".safetensors") for f in names)
    if type_ == "upscaler":
        return _upscaler_name_ok(name) and any(f.endswith(_UPSCALER_EXT) for f in names)
    return True


def prune_list(sources: list[dict]) -> list[dict]:
    """Drop LoRAs this app cannot use (unless the user added them: `custom`), tag the rest with
    their family. Upscalers and base entries pass through. Does not mutate the input."""
    out = []
    for s in sources:
        s = dict(s)
        if s.get("type") == "lora":
            fam = lora_family(s.get("name", "") or repo_id(s.get("url", "")))
            if fam is None and not s.get("custom"):
                continue
            if fam:
                s["family"] = fam
        out.append(s)
    return out


def screen(candidates: list[dict], fetch_files) -> tuple[list[dict], list[dict]]:
    """Update-time filter for freshly discovered repos: (kept, skipped[{name, url, reason}]).
    Reads each repo's file list in parallel; an unreadable repo is skipped, never fatal."""
    checkable = [c for c in candidates if c.get("type") in ("lora", "upscaler")]

    def files_of(c):
        try:
            return list(fetch_files(repo_id(c["url"])))
        except Exception:
            return None

    with ThreadPoolExecutor(WORKERS) as ex:
        results = dict(zip((c["id"] for c in checkable), ex.map(files_of, checkable)))

    kept, skipped = [], []
    for c in candidates:
        if c.get("type") not in ("lora", "upscaler"):
            kept.append(c)
            continue
        files = results[c["id"]]
        ok = is_supported(c["type"], c.get("name", ""), files)
        if ok:
            c = dict(c)
            if c["type"] == "lora":
                c["family"] = lora_family(c.get("name", ""))
            kept.append(c)
            continue
        if files is None:
            reason = "repo unreadable"
        elif c["type"] == "lora" and lora_family(c.get("name", "")) is None:
            reason = "unsupported model family"
        elif c["type"] == "lora":
            reason = "no .safetensors file"
        else:
            reason = "not a usable upscaler"
        skipped.append({"name": c.get("name", ""), "url": c.get("url", ""), "reason": reason})
    return kept, skipped


def enrich(sources: list[dict], fetch_card, deadline_s: float = ENRICH_DEADLINE_S) -> tuple[list[dict], dict]:
    """Fill in `description` (+ `function` for LoRAs) from each repo's model card.

    Only rows with no description, not `custom`, not already `described`. A failed fetch leaves the
    row untouched (retried next Update); a card with no usable prose is marked `described` so it is
    not re-fetched every time. Returns (new_list, {"described": n, "failed": n})."""
    todo = [i for i, s in enumerate(sources)
            if s.get("type") in ("lora", "upscaler") and not s.get("custom")
            and not (s.get("description") or "").strip() and not s.get("described")]
    out = [dict(s) for s in sources]
    report = {"described": 0, "failed": 0}
    if not todo:
        return out, report

    ex = ThreadPoolExecutor(WORKERS)
    futures = {i: ex.submit(fetch_card, repo_id(sources[i]["url"])) for i in todo}
    wait(list(futures.values()), timeout=deadline_s)
    ex.shutdown(wait=False, cancel_futures=True)
    for i, fut in futures.items():
        try:
            text = fut.result(timeout=0)
        except Exception:
            report["failed"] += 1
            continue
        desc = describe(text)
        out[i]["description"] = desc
        out[i]["described"] = True
        if out[i].get("type") == "lora":
            out[i]["function"] = lora_function(out[i].get("name", ""), desc)
        report["described"] += 1
    return out, report


# ── real network implementations (not unit-tested; exercised by a live Update) ─────

def hf_fetch_card(repo: str) -> str:
    from huggingface_hub import hf_hub_download
    return open(hf_hub_download(repo, "README.md", etag_timeout=10), encoding="utf-8", errors="replace").read()


def hf_fetch_files(repo: str) -> list[str]:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo)
