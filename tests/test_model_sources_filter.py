"""Model Sources: what is good for this app, what each LoRA does, and the Update enrichment.

Card fixtures in tests/fixtures/model_cards/ are real HF cards (`hf models card <id> --text`).
"""
from pathlib import Path

import pytest

from core import model_sources as ms

CARDS = Path(__file__).parent / "fixtures" / "model_cards"


def card(repo: str) -> str:
    return (CARDS / (repo.replace("/", "_") + ".md")).read_text()


# ── family: which of this app's models a LoRA is for ────────────────────────────

@pytest.mark.parametrize("name, family", [
    ("Flux2-Klein-9B-Enhanced-Details", "klein-9B"),
    ("flux-2-klein-4B-zoom-lora", "klein-4B"),
    ("FLUX.2-klein-base-9B-UnifiedReward-Flex-lora", "klein-9B"),
    ("Flux2-Klein-Delight-LoRA", "klein"),                 # size not stated
    ("Eyes_Direction_Lora_Flux2Klein9B", "klein-9B"),
    ("Z-Image-loras", "Z-Image"),
    ("LTX-Video-Cakeify-LoRA", "LTX-Video"),
    ("LTXV-LoRAs", "LTX-Video"),
    ("LTX-Video-ICLoRA-depth-13b-0.9.7", "LTX-Video"),
])
def test_lora_family_supported(name, family):
    assert ms.lora_family(name) == family


@pytest.mark.parametrize("name", [
    "huggy_v17", "lora-xl-3d-icon-0.0001-1500-1-5",        # SDXL training runs
    "Plushie-Kontext-Dev-LoRA", "FLUX.1-Canny-dev-lora",   # FLUX.1
    "FLUX.2-dev-Turbo",                                    # FLUX.2-dev, not klein
    "Qwen-Image-Edit-2509-Fusion", "Krea2-Anime-Style-Collection",
    "LTX-2.3-22b-IC-LoRA-Relight", "LTX-2-19b-LoRA-Camera-Control-Static",
    "LTX2.3-Multifunctional",                              # LTX-2 generation, app runs LTX-Video 0.9.x
    "rick-and-morty-style-wan-21",
])
def test_lora_family_unsupported(name):
    assert ms.lora_family(name) is None


# ── function: a short tag for what the LoRA does ────────────────────────────────

@pytest.mark.parametrize("name, function", [
    ("Eyes_Direction_Lora_Flux2Klein9B", "face"),
    ("Flux2-Klein-9B-Enhanced-Details", "detail"),
    ("flux-2-klein-4B-zoom-lora", "camera"),
    ("LTX-Video-Cakeify-LoRA", "effect"),
    ("Flux2-Klein-Delight-LoRA", "lighting"),
    ("flux-2-klein-4B-outpaint-lora", "edit"),
    ("flux2-klein-base-9b-distill-lora", "speed"),
    ("Z-Image-loras", "other"),
    ("Klein_AnimeHDupscaling", "detail"),            # upscaling beats the anime style word
    ("Virtual Try-on (klein 9B)", "edit"),
])
def test_lora_function(name, function):
    assert ms.lora_function(name) == function


# ── description: first real sentence of the card ────────────────────────────────

def test_describe_skips_headings_triggers_images_and_html():
    d = ms.describe(card("eric-venti-seeds/Eyes_Direction_Lora_Flux2Klein9B"))
    assert d.startswith("LoRA trained to change where the eyes are looking")
    assert "#" not in d and "Trigger" not in d


def test_describe_strips_markdown_links_and_stays_short():
    d = ms.describe(card("Lightricks/LTX-Video-Cakeify-LoRA"))
    assert d.startswith("This repository contains a LoRA model trained on top of LTX Video v0.9.5")
    assert "](" not in d and "http" not in d
    assert len(d) <= 140


def test_describe_skips_bare_url_lines():
    d = ms.describe(card("dx8152/Flux2-Klein-9B-Enhanced-Details"))
    assert "runninghub" not in d and d != ""


def test_describe_empty_or_prose_free_card():
    assert ms.describe("") == ""
    assert ms.describe("# Title\n\n![img](a.png)\n\n<div></div>\n") == ""


def test_describe_cuts_at_a_word_boundary():
    d = ms.describe("word " * 80)
    assert len(d) <= 140 and d.endswith("…") and not d.endswith(" …")


# ── is_supported: repo contents decide ──────────────────────────────────────────

def test_is_supported_lora_needs_family_and_safetensors():
    ok = ["README.md", "weights.safetensors"]
    assert ms.is_supported("lora", "Flux2-Klein-9B-Enhanced-Details", ok) is True
    assert ms.is_supported("lora", "Flux2-Klein-9B-Enhanced-Details", ["README.md", "w.ckpt"]) is False
    assert ms.is_supported("lora", "huggy_v17", ok) is False


def test_is_supported_upscaler_accepts_safetensors_or_pth():
    assert ms.is_supported("upscaler", "4x-UltraSharp", ["4x-UltraSharp.pth"]) is True
    assert ms.is_supported("upscaler", "4x-UltraSharp", ["m.safetensors"]) is True
    assert ms.is_supported("upscaler", "4x-UltraSharp", ["README.md"]) is False
    assert ms.is_supported("upscaler", "random-diffusion-thing", ["m.safetensors"]) is False


def test_is_supported_unknown_when_files_unreadable():
    assert ms.is_supported("lora", "Flux2-Klein-9B-Enhanced-Details", None) is None


# ── prune_list: existing entries ────────────────────────────────────────────────

def _src(name, type_="lora", **kw):
    return {"id": name, "name": name, "url": f"https://huggingface.co/org/{name}",
            "type": type_, "description": "", "model_choice": "", **kw}


def test_prune_drops_unsupported_loras_keeps_custom_upscalers_and_base():
    src = [_src("Flux2-Klein-9B-Enhanced-Details"), _src("huggy_v17"),
           _src("my-own-thing", custom=True), _src("4x-UltraSharp", "upscaler"), _src("m", "base")]
    out = ms.prune_list(src)
    assert [s["name"] for s in out] == ["Flux2-Klein-9B-Enhanced-Details", "my-own-thing",
                                        "4x-UltraSharp", "m"]
    assert out[0]["family"] == "klein-9B"
    assert "family" not in src[0]                          # input not mutated


# ── enrich: what Update does for new and existing entries ───────────────────────

def test_enrich_fills_missing_description_and_function():
    src = [_src("LTX-Video-Cakeify-LoRA")]
    out, report = ms.enrich(src, fetch_card=lambda repo: card("Lightricks/LTX-Video-Cakeify-LoRA"))
    assert out[0]["description"].startswith("This repository contains a LoRA model")
    assert out[0]["function"] == "effect" and out[0]["described"] is True
    assert report["described"] == 1


def test_enrich_never_overwrites_a_description_or_touches_custom():
    src = [_src("Flux2-Klein-9B-Enhanced-Details", description="mine"),
           _src("Flux2-Klein-Delight-LoRA", custom=True)]
    calls = []
    out, _ = ms.enrich(src, fetch_card=lambda repo: calls.append(repo) or "Some text here.")
    assert out[0]["description"] == "mine"
    assert out[1]["description"] == ""
    assert calls == []                                      # nothing to fetch


def test_enrich_failed_fetch_leaves_row_and_retries_next_time():
    def boom(repo):
        raise TimeoutError("hf down")
    src = [_src("Flux2-Klein-Delight-LoRA")]
    out, report = ms.enrich(src, fetch_card=boom)
    assert out[0]["description"] == "" and not out[0].get("described")
    assert report["failed"] == 1


def test_enrich_prose_free_card_is_marked_described_not_retried():
    src = [_src("Flux2-Klein-Delight-LoRA")]
    out, _ = ms.enrich(src, fetch_card=lambda repo: "# T\n\n![i](a.png)\n")
    assert out[0]["described"] is True and out[0]["function"] == "lighting"
    calls = []
    ms.enrich(out, fetch_card=lambda repo: calls.append(repo) or "x")
    assert calls == []


def test_enrich_fetches_each_repo_once_in_parallel_with_cap():
    src = [_src(f"Flux2-Klein-9B-Thing{i}") for i in range(12)]
    seen = []
    out, report = ms.enrich(src, fetch_card=lambda repo: seen.append(repo) or "A real sentence here.")
    assert len(seen) == 12 and len(set(seen)) == 12
    assert report["described"] == 12


# ── screen: Update-time filter on freshly discovered candidates ──────────────────

def test_screen_keeps_supported_skips_rest_with_reasons():
    cands = [_src("Flux2-Klein-9B-Enhanced-Details"), _src("huggy_v17"),
             _src("Flux2-Klein-Delight-LoRA"), _src("4x-UltraSharp", "upscaler")]
    files = {"org/Flux2-Klein-9B-Enhanced-Details": ["a.safetensors"],
             "org/huggy_v17": ["a.safetensors"],
             "org/Flux2-Klein-Delight-LoRA": ["a.ckpt"],
             "org/4x-UltraSharp": ["x.pth"]}
    kept, skipped = ms.screen(cands, fetch_files=lambda repo: files[repo])
    assert [s["name"] for s in kept] == ["Flux2-Klein-9B-Enhanced-Details", "4x-UltraSharp"]
    assert kept[0]["family"] == "klein-9B"
    reasons = {s["name"]: s["reason"] for s in skipped}
    assert reasons == {"huggy_v17": "unsupported model family",
                       "Flux2-Klein-Delight-LoRA": "no .safetensors file"}


def test_screen_skips_on_unreadable_repo_instead_of_raising():
    def boom(repo):
        raise OSError("hf down")
    kept, skipped = ms.screen([_src("Flux2-Klein-9B-Enhanced-Details")], fetch_files=boom)
    assert kept == [] and skipped[0]["reason"] == "repo unreadable"


def test_screen_leaves_base_entries_alone():
    base = _src("m", "base")
    kept, skipped = ms.screen([base], fetch_files=lambda repo: [])
    assert kept == [base] and skipped == []


def test_prune_uses_the_repo_id_when_the_label_says_nothing():
    s = {"id": "x", "name": "Style Pack", "type": "lora", "description": "", "model_choice": "",
         "url": "https://huggingface.co/DeverStyle/Flux.2-Klein-Loras"}
    assert ms.prune_list([s])[0]["family"] == "klein"


# ── merge_discovery: one Update = screen + remember skips + merge + describe ─────

def _repo_files(mapping):
    return lambda repo: mapping[repo]


def test_merge_discovery_adds_supported_skips_rest_and_remembers_skips():
    current = [_src("Flux2-Klein-9B-Consistency", description="kept")]
    cands = [_src("Flux2-Klein-9B-Enhanced-Details"), _src("huggy_v17"), _src("Flux2-Klein-Delight-LoRA")]
    files = _repo_files({"org/Flux2-Klein-9B-Enhanced-Details": ["a.safetensors"],
                         "org/huggy_v17": ["a.safetensors"],
                         "org/Flux2-Klein-Delight-LoRA": ["a.ckpt"]})
    cards = lambda repo: "A real sentence about " + repo + "."
    out, ignored, rep = ms.merge_discovery(current, cands, [], files, cards)
    assert [s["name"] for s in out] == ["Flux2-Klein-9B-Consistency", "Flux2-Klein-9B-Enhanced-Details"]
    assert rep["added"] == 1 and rep["skipped"] == 2
    assert set(ignored) == {"https://huggingface.co/org/huggy_v17",
                            "https://huggingface.co/org/Flux2-Klein-Delight-LoRA"}
    assert out[1]["description"].startswith("A real sentence") and out[1]["function"] == "detail"
    assert out[0]["description"] == "kept"


def test_merge_discovery_ignores_previously_skipped_and_existing_urls():
    current = [_src("Flux2-Klein-9B-Consistency")]
    cands = [_src("Flux2-Klein-9B-Consistency"), _src("huggy_v17"), _src("Flux2-Klein-Delight-LoRA")]
    seen = []
    def files(repo):
        seen.append(repo)
        return ["a.safetensors"]
    ignored = ["https://huggingface.co/org/huggy_v17"]
    out, ignored2, rep = ms.merge_discovery(current, cands, ignored, files, lambda r: "Text here.")
    assert seen == ["org/Flux2-Klein-Delight-LoRA"]          # existing + ignored never re-checked
    assert rep["added"] == 1 and "https://huggingface.co/org/huggy_v17" in ignored2


def test_merge_discovery_does_not_remember_unreadable_repos():
    def boom(repo):
        raise OSError("hf down")
    _, ignored, rep = ms.merge_discovery([], [_src("Flux2-Klein-9B-Enhanced-Details")], [], boom, lambda r: "")
    assert ignored == [] and rep["skipped"] == 1             # retried next Update


def test_merge_discovery_backfills_descriptions_of_existing_rows():
    current = [_src("Flux2-Klein-9B-Consistency")]
    out, _, rep = ms.merge_discovery(current, [], [], lambda r: [], lambda r: "Keeps faces consistent.")
    assert out[0]["description"] == "Keeps faces consistent." and rep["described"] == 1
