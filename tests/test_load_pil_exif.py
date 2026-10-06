from PIL import Image


def _save(path, orientation):
    im = Image.new("RGB", (40, 20), "red")      # landscape pixels
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    im.save(path, "JPEG", exif=exif)


def test_orientation_6_is_rotated_upright(monkeypatch, tmp_path):
    import server
    monkeypatch.setattr(server, "TEMP_DIR", tmp_path)
    _save(tmp_path / "a.jpg", 6)
    assert server._load_pil("a.jpg").size == (20, 40)


def test_no_exif_unchanged(monkeypatch, tmp_path):
    import server
    monkeypatch.setattr(server, "TEMP_DIR", tmp_path)
    _save(tmp_path / "b.jpg", None)
    assert server._load_pil("b.jpg").size == (40, 20)
