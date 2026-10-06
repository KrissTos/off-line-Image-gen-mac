def test_record_output_rewrites_saved_path(tmp_path):
    import server
    from core import run_store
    run = run_store.create_run(tmp_path, {"prompt": "p", "seed": 1}, [])
    src = run / "outputs" / "20261006_x.mp4"
    src.write_bytes(b"v")
    ev = {"type": "video", "path": str(src), "info": f"Seed: 5 | Saved: {src}"}
    server._record_output(run, ev, 5)
    assert str(src) not in ev["info"]
    assert ev["info"].endswith(f"Saved: {ev['path']}")


def test_record_output_info_without_saved_untouched(tmp_path):
    import server
    from core import run_store
    run = run_store.create_run(tmp_path, {"prompt": "p", "seed": 1}, [])
    src = run / "outputs" / "a.png"
    src.write_bytes(b"i")
    ev = {"type": "image", "path": str(src), "info": "Seed: 5 | Model: m"}
    server._record_output(run, ev, 5)
    assert ev["info"] == "Seed: 5 | Model: m"
