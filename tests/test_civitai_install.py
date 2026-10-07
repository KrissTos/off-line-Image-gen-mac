"""CivitAI key, registry, safe download and delete. Fake network, tmp dirs."""
import hashlib
import json
import os
import stat

import pytest

from core import civitai_install as ci


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "BASE_DIR", tmp_path)
    monkeypatch.setattr(ci, "CHUNK", 4)
    ci._jobs.clear()
    (tmp_path / "lora_uploads").mkdir()
    return tmp_path


class FakeResp:
    def __init__(self, status=200, body=b"", headers=None, fail_after=None):
        self.status_code, self.body, self.headers = status, body, headers or {}
        self.fail_after, self.closed = fail_after, False

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            if self.fail_after is not None and i >= self.fail_after:
                raise OSError(28, "No space left on device")
            yield self.body[i:i + n]

    def close(self):
        self.closed = True


def row(vid=11, mid=5, fam="klein-9B", file="Comic.safetensors", body=b"abcdefgh", sha=None):
    return {"id": f"civ-{mid}-{fam}", "family": fam, "civitai": {
        "modelId": mid, "versionId": vid, "file": file, "sizeKB": 1,
        "sha256": sha if sha is not None else hashlib.sha256(body).hexdigest(), "trained": ["ComSpa"]}}


def run(r, body=b"abcdefgh", status=200, verify=lambda p, f: None, **kw):
    streams = []

    def open_stream(url, key):
        streams.append((url, key))
        return kw.get("resp") or FakeResp(status, body)
    ci.start_download(r, open_stream=open_stream, verify=verify, threaded=False)
    return ci.job_state(r["civitai"]["versionId"]), streams


def lora_files(env):
    return sorted(p.name for p in (env / "lora_uploads").iterdir())


# ── key ──────────────────────────────────────────────────────────────────────

def test_key_roundtrip_mode_600_and_strip(env):
    assert ci.get_key() is None
    ci.set_key("  abc123  ")
    assert ci.get_key() == "abc123"
    assert stat.S_IMODE(os.stat(env / "civitai" / "token").st_mode) == 0o600
    ci.clear_key()
    assert ci.get_key() is None
    ci.clear_key()                                           # idempotent


@pytest.mark.parametrize("bad", ["", "   ", "ab cd", "ab\ncd"])
def test_key_rejects_empty_or_whitespace_inside(bad):
    with pytest.raises(ValueError):
        ci.set_key(bad)


# ── file names ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, safe", [
    ("Comic.safetensors", "Comic.safetensors"),
    ("../../evil.safetensors", "evil.safetensors"),
    ("/etc/passwd.safetensors", "passwd.safetensors"),
    ("C:\\x\\y.safetensors", "y.safetensors"),
    (".hidden.safetensors", "hidden.safetensors"),
    ("a b$%.safetensors", "a b__.safetensors"),
])
def test_safe_filename(raw, safe):
    assert ci.safe_filename(raw) == safe


@pytest.mark.parametrize("raw", ["x.pt", "noext", ".safetensors", "..", ""])
def test_safe_filename_rejects(raw):
    with pytest.raises(ValueError):
        ci.safe_filename(raw)


def test_safe_filename_caps_length_keeping_extension():
    out = ci.safe_filename("a" * 400 + ".safetensors")
    assert len(out) <= 120 and out.endswith(".safetensors")


# ── download ─────────────────────────────────────────────────────────────────

def test_download_success(env):
    ci.set_key("K")
    st, streams = run(row())
    assert st["state"] == "done" and st["file"] == "Comic.safetensors" and st["bytes"] == 8
    assert streams == [("https://civitai.com/api/download/models/11", "K")]
    assert lora_files(env) == ["Comic.safetensors"]                       # no .part left
    reg = json.loads((env / "civitai_installed.json").read_text())
    assert reg["Comic.safetensors"]["versionId"] == 11 and reg["Comic.safetensors"]["trained"] == ["ComSpa"]
    assert ci.installed_map()[(5, "klein-9B")]["versionId"] == 11


def test_checksum_mismatch_discards(env):
    st, _ = run(row(sha="00" * 32))
    assert st["state"] == "error" and "Checksum" in st["error"]
    assert lora_files(env) == [] and not (env / "civitai_installed.json").exists()


@pytest.mark.parametrize("status, key, text", [
    (401, None, "needs a CivitAI API key"), (401, "tok_9f8e7d6c5b4a", "Key rejected"),
    (403, "tok_9f8e7d6c5b4a", "early access"), (404, None, "removed"), (429, None, "Rate limited"),
    (500, None, "HTTP 500"),
])
def test_http_errors_are_readable(env, status, key, text):
    if key:
        ci.set_key(key)
    st, _ = run(row(), status=status)
    assert st["state"] == "error" and text.lower() in st["error"].lower()
    assert lora_files(env) == []


def test_verify_rejection_deletes_file_and_reports(env):
    def verify(path, fam):
        raise RuntimeError("This LoRA is for klein-4B, not 9B")
    st, _ = run(row(), verify=verify)
    assert st["state"] == "error" and "klein-4B" in st["error"]
    assert lora_files(env) == []


def test_disk_error_mid_stream_cleans_up_and_reports(env):
    st, _ = run(row(), resp=FakeResp(body=b"abcdefgh", fail_after=4))
    assert st["state"] == "error" and "Disk error" in st["error"]
    assert lora_files(env) == []


def test_error_text_never_contains_the_key(env):
    ci.set_key("SECRET-KEY")

    def boom(url, key):
        raise RuntimeError(f"connection to {url} with {key} failed")
    ci.start_download(row(), open_stream=boom, verify=lambda p, f: None, threaded=False)
    st = ci.job_state(11)
    assert st["state"] == "error" and "SECRET-KEY" not in json.dumps(st)


def test_hostile_file_name_stays_inside_lora_uploads(env):
    st, _ = run(row(file="../../evil.safetensors"))
    assert st["state"] == "done"
    assert lora_files(env) == ["evil.safetensors"]
    assert not (env.parent / "evil.safetensors").exists()


def test_name_collision_with_user_upload_gets_suffix_and_user_file_survives(env):
    (env / "lora_uploads" / "Comic.safetensors").write_bytes(b"mine")
    st, _ = run(row())
    assert st["file"] == "Comic__civ11.safetensors"
    assert (env / "lora_uploads" / "Comic.safetensors").read_bytes() == b"mine"


def test_update_replaces_old_version_after_verify(env):
    run(row(vid=11, file="Comic_v1.safetensors"))
    st, _ = run(row(vid=12, file="Comic_v2.safetensors", body=b"newnewnew"), body=b"newnewnew")
    assert st["state"] == "done"
    assert lora_files(env) == ["Comic_v2.safetensors"]
    assert list(ci.load_registry()) == ["Comic_v2.safetensors"]
    assert ci.installed_map()[(5, "klein-9B")]["versionId"] == 12


def test_failed_update_keeps_old_version(env):
    run(row(vid=11, file="Comic_v1.safetensors"))
    st, _ = run(row(vid=12, file="Comic_v2.safetensors", sha="00" * 32))
    assert st["state"] == "error"
    assert lora_files(env) == ["Comic_v1.safetensors"]


def test_active_job_is_not_restarted(env):
    ci._jobs[11] = {"state": "downloading", "bytes": 3, "total": 8, "error": None}
    called = []
    out = ci.start_download(row(), open_stream=lambda u, k: called.append(1), threaded=False)
    assert out["state"] == "downloading" and called == []


def test_stale_partial_files_are_swept(env):
    old = env / "lora_uploads" / ".tmp_old.safetensors.part"
    fresh = env / "lora_uploads" / ".tmp_new.safetensors.part"
    old.write_bytes(b"x"); fresh.write_bytes(b"x")
    os.utime(old, (1, 1))
    run(row())
    assert not old.exists() and fresh.exists()


# ── redirects: the key goes to civitai.com only ──────────────────────────────

def test_open_stream_drops_key_on_redirect_to_storage_host():
    seen = []
    resps = [FakeResp(307, headers={"Location": "https://b2.civitai.com/file/x?sig=1"}), FakeResp(200, b"ok")]

    def get(url, headers=None, **kw):
        seen.append((url, dict(headers or {})))
        return resps.pop(0)
    r = ci.open_stream("https://civitai.com/api/download/models/1", "K", get=get)
    assert r.status_code == 200
    assert seen[0][1] == {"Authorization": "Bearer K"} and seen[1][1] == {}


def test_open_stream_rejects_http_redirect_and_loops():
    def http(url, headers=None, **kw):
        return FakeResp(302, headers={"Location": "http://evil.example/x"})
    with pytest.raises(ci.DownloadError):
        ci.open_stream("https://civitai.com/a", "K", get=http)

    def loop(url, headers=None, **kw):
        return FakeResp(302, headers={"Location": "https://civitai.com/a"})
    with pytest.raises(ci.DownloadError):
        ci.open_stream("https://civitai.com/a", "K", get=loop)


# ── verify_lora ──────────────────────────────────────────────────────────────

def test_verify_lora_unreadable_and_klein_size_mismatch(env, monkeypatch):
    bad = env / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(RuntimeError, match="Could not read"):
        ci.verify_lora(str(bad), "klein-9B")

    import torch
    from safetensors.torch import save_file
    good = env / "good.safetensors"
    save_file({"a": torch.zeros(1)}, str(good))
    import core.lora_flux2 as lf
    monkeypatch.setattr(lf, "check_lora_compatibility", lambda p: None)
    monkeypatch.setattr(lf, "lora_variant", lambda p: "4b")
    with pytest.raises(RuntimeError, match="klein-4B"):
        ci.verify_lora(str(good), "klein-9B")
    ci.verify_lora(str(good), "klein-4B")                                  # matching size passes


def test_verify_lora_zimage_skips_the_klein_check(env, monkeypatch):
    import torch
    from safetensors.torch import save_file
    good = env / "z.safetensors"
    save_file({"a": torch.zeros(1)}, str(good))
    import core.lora_flux2 as lf

    def no(p):
        raise AssertionError("klein check must not run for Z-Image")
    monkeypatch.setattr(lf, "check_lora_compatibility", no)
    ci.verify_lora(str(good), "Z-Image")


# ── delete + trigger ─────────────────────────────────────────────────────────

def test_delete_only_registry_files_and_not_when_loaded(env):
    run(row())
    (env / "lora_uploads" / "mine.safetensors").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="[Uu]nload"):
        ci.delete_installed(11, loaded_paths=[str(env / "lora_uploads" / "Comic.safetensors")])
    assert "Comic.safetensors" in lora_files(env)
    with pytest.raises(LookupError):
        ci.delete_installed(999)
    assert ci.delete_installed(11) == "Comic.safetensors"
    assert lora_files(env) == ["mine.safetensors"] and ci.load_registry() == {}


def test_trained_trigger_comes_from_registry_and_vanishes_on_delete(env):
    run(row())
    assert ci.trained_trigger("Comic.safetensors") == "ComSpa"
    assert ci.trained_trigger("other.safetensors") is None
    ci.delete_installed(11)
    assert ci.trained_trigger("Comic.safetensors") is None
