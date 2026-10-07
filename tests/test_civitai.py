"""CivitAI LoRA discovery: mapping, filters, merge, annotate. No network."""
import pytest

from core import civitai, model_sources as ms

SHA = "ab" * 32


def file_(name="a.safetensors", fid=1, primary=True, kb=1000.0, type_="Model", sha=SHA.upper()):
    return {"id": fid, "name": name, "type": type_, "primary": primary, "sizeKB": kb,
            "hashes": {"SHA256": sha}}


def version(vid, base, files=None, level=1, words=()):
    return {"id": vid, "baseModel": base, "nsfwLevel": level, "trainedWords": list(words),
            "files": [file_()] if files is None else files}


def model(mid, name, versions, nsfw=False, desc="<p>Spanish comic style. Second sentence.</p>"):
    return {"id": mid, "name": name, "nsfw": nsfw, "description": desc, "modelVersions": versions}


@pytest.mark.parametrize("base, family", [
    ("Flux.2 Klein 9B", "klein-9B"), ("Flux.2 Klein 9B-base", "klein-9B"),
    ("Flux.2 Klein 4B", "klein-4B"), ("Flux.2 Klein 4B-base", "klein-4B"),
    ("ZImageTurbo", "Z-Image"),
])
def test_family_of_supported(base, family):
    assert civitai.family_of(base) == family


@pytest.mark.parametrize("base", ["ZImageBase", "Flux.1 D", "SDXL 1.0", "LTXV", "Qwen", "", None])
def test_family_of_unsupported(base):
    assert civitai.family_of(base) is None


def test_row_fields():
    m = model(2674582, "Spanish Comic", [version(9, "Flux.2 Klein 9B", words=["ComSpa"])])
    (row,) = civitai.rows_from_model(m)
    assert row["id"] == "civ-2674582-klein-9B"
    assert row["name"] == "Spanish Comic"
    assert row["type"] == "lora" and row["provider"] == "civitai"
    assert row["url"] == "https://civitai.com/models/2674582"
    assert row["family"] == "klein-9B"
    assert row["description"] == "Spanish comic style."
    assert row["described"] is True and row["nsfw"] is False
    assert row["civitai"] == {"modelId": 2674582, "versionId": 9, "fileId": 1, "file": "a.safetensors",
                              "sha256": SHA, "sizeKB": 1000.0, "trained": ["ComSpa"]}


def test_newest_version_per_family_and_two_families():
    m = model(7, "Both", [version(30, "Flux.2 Klein 9B"), version(20, "Flux.2 Klein 9B"),
                          version(10, "Flux.2 Klein 4B")])
    rows = {r["family"]: r for r in civitai.rows_from_model(m)}
    assert set(rows) == {"klein-9B", "klein-4B"}
    assert rows["klein-9B"]["civitai"]["versionId"] == 30
    assert rows["klein-4B"]["civitai"]["versionId"] == 10
    assert rows["klein-9B"]["id"] != rows["klein-4B"]["id"]


def test_requires_a_safetensors_model_file():
    pt = [file_("a.pt"), file_("b.safetensors", type_="Training Data")]
    assert civitai.rows_from_model(model(1, "x", [version(1, "Flux.2 Klein 9B", files=pt)])) == []
    assert civitai.rows_from_model(model(1, "x", [version(1, "Flux.2 Klein 9B", files=[])])) == []


def test_primary_file_preferred():
    files = [file_("other.safetensors", fid=1, primary=False), file_("main.safetensors", fid=2)]
    (row,) = civitai.rows_from_model(model(1, "x", [version(1, "ZImageTurbo", files=files)]))
    assert row["civitai"]["file"] == "main.safetensors"


def test_unsupported_base_gives_no_row():
    assert civitai.rows_from_model(model(1, "x", [version(1, "ZImageBase")])) == []


@pytest.mark.parametrize("model_flag, level, nsfw", [
    (False, 1, False), (False, 3, False), (False, 4, True), (False, 8, True), (False, 16, True),
    (True, 1, True),
])
def test_nsfw_detection(model_flag, level, nsfw):
    (row,) = civitai.rows_from_model(model(1, "x", [version(1, "ZImageTurbo", level=level)], nsfw=model_flag))
    assert row["nsfw"] is nsfw


def test_missing_description_is_empty_not_error():
    (row,) = civitai.rows_from_model(model(1, "x", [version(1, "ZImageTurbo")], desc=None))
    assert row["description"] == ""


def fake_fetch(table, calls=None):
    def fetch(base, limit):
        if calls is not None:
            calls.append((base, limit))
        v = table[base]
        if isinstance(v, Exception):
            raise v
        return v
    return fetch


def test_discover_dedupes_across_base_variants_and_respects_top_n():
    m = model(1, "x", [version(1, "Flux.2 Klein 9B")])
    table = {b: [] for b in civitai.BASE_FAMILY}
    table["Flux.2 Klein 9B"] = [m]
    table["Flux.2 Klein 9B-base"] = [m]
    calls = []
    rows, failed = civitai.discover(fake_fetch(table, calls), show_nsfw=False, top_n=50)
    assert [r["id"] for r in rows] == ["civ-1-klein-9B"] and failed == []
    assert {limit for _, limit in calls} == {50}


def test_discover_drops_nsfw_unless_shown():
    m = model(1, "x", [version(1, "ZImageTurbo", level=8)])
    table = {b: [] for b in civitai.BASE_FAMILY}
    table["ZImageTurbo"] = [m]
    assert civitai.discover(fake_fetch(table), show_nsfw=False)[0] == []
    assert len(civitai.discover(fake_fetch(table), show_nsfw=True)[0]) == 1


def test_discover_failure_of_one_base_keeps_the_rest():
    table = {b: [] for b in civitai.BASE_FAMILY}
    table["ZImageTurbo"] = [model(1, "x", [version(1, "ZImageTurbo")])]
    table["Flux.2 Klein 4B"] = RuntimeError("HTTP 429")
    rows, failed = civitai.discover(fake_fetch(table), show_nsfw=False)
    assert [r["family"] for r in rows] == ["Z-Image"]
    assert failed == ["Flux.2 Klein 4B"]


def crow(mid, fam, vid, nsfw=False):
    return {"id": f"civ-{mid}-{fam}", "name": f"m{mid}", "type": "lora", "provider": "civitai",
            "family": fam, "nsfw": nsfw, "civitai": {"modelId": mid, "versionId": vid}}


def test_merge_replaces_civitai_rows_and_keeps_others():
    hf = {"id": "src-1", "name": "hf", "type": "lora", "url": "https://huggingface.co/o/hf"}
    current = [hf, crow(1, "klein-9B", 1), crow(2, "klein-9B", 1)]
    fresh = [crow(1, "klein-9B", 5), crow(3, "klein-9B", 1)]
    rows, added = civitai.merge_rows(current, fresh, failed_families=set(), installed_keys=set())
    ids = [r["id"] for r in rows]
    assert ids[0] == "src-1"
    assert sorted(ids[1:]) == ["civ-1-klein-9B", "civ-3-klein-9B"]     # civ-2 dropped (not installed)
    assert added == 1
    assert next(r for r in rows if r["id"] == "civ-1-klein-9B")["civitai"]["versionId"] == 5


def test_merge_keeps_old_rows_of_failed_family_and_installed_rows():
    current = [crow(1, "klein-4B", 1), crow(2, "klein-9B", 1), crow(3, "klein-9B", 1)]
    rows, added = civitai.merge_rows(current, [], failed_families={"klein-4B"},
                                     installed_keys={(2, "klein-9B")})
    assert sorted(r["id"] for r in rows) == ["civ-1-klein-4B", "civ-2-klein-9B"]
    assert added == 0


def test_annotate_flags_hides_nsfw_and_leaves_hf_rows():
    hf = {"id": "src-1", "name": "hf", "type": "lora"}
    rows = [hf, crow(1, "klein-9B", 5), crow(2, "klein-9B", 1), crow(3, "klein-9B", 1, nsfw=True),
            crow(4, "klein-9B", 1, nsfw=True)]
    installed = {(1, "klein-9B"): {"versionId": 2}, (2, "klein-9B"): {"versionId": 1},
                 (4, "klein-9B"): {"versionId": 1}}
    out = {r["id"]: r for r in civitai.annotate(rows, installed, show_nsfw=False)}
    assert "installed" not in out["src-1"]
    assert out["civ-1-klein-9B"]["installed"] and out["civ-1-klein-9B"]["update"]
    assert out["civ-1-klein-9B"]["installed_version"] == 2
    assert out["civ-2-klein-9B"]["installed"] and not out["civ-2-klein-9B"]["update"]
    assert "civ-3-klein-9B" not in out                      # NSFW, not installed: hidden
    assert "civ-4-klein-9B" in out                          # NSFW but installed: stays deletable
    assert len(civitai.annotate(rows, {}, show_nsfw=True)) == 5


def test_prune_list_keeps_civitai_rows_without_klein_in_the_name():
    rows = [crow(1, "klein-9B", 1), {"id": "x", "name": "huggy_v17", "type": "lora", "url": "https://huggingface.co/o/huggy_v17"}]
    out = ms.prune_list(rows)
    assert [r["id"] for r in out] == ["civ-1-klein-9B"]
    assert out[0]["family"] == "klein-9B"
