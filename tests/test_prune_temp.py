import os


def test_prune_only_old_files(monkeypatch, tmp_path):
    import server
    from core import run_store
    monkeypatch.setattr(server, "TEMP_DIR", tmp_path)
    trashed = []
    monkeypatch.setattr(run_store, "trash", lambda p: trashed.append(p.name))
    old, new = tmp_path / "old.png", tmp_path / "new.png"
    old.write_bytes(b"o"); new.write_bytes(b"n")
    now = 1000 + 7200
    os.utime(old, (1000, 1000))
    os.utime(new, (now, now))
    (tmp_path / "sub").mkdir()
    assert server._prune_temp_dir(max_age_s=3600, now=now) == 1
    assert trashed == ["old.png"]


def test_prune_continues_after_a_failing_file(monkeypatch, tmp_path):
    import server
    from core import run_store
    monkeypatch.setattr(server, "TEMP_DIR", tmp_path)
    done = []

    def fake_trash(p):
        if p.name == "a.png":
            raise RuntimeError("trash failed")
        done.append(p.name)
    monkeypatch.setattr(run_store, "trash", fake_trash)
    for n in ("a.png", "b.png"):
        (tmp_path / n).write_bytes(b"x")
        os.utime(tmp_path / n, (1000, 1000))
    assert server._prune_temp_dir(max_age_s=3600, now=1000 + 7200) == 1
    assert done == ["b.png"]
