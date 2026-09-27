"""SAM click-to-mask service (core.segment) — SAM replaced by a fake so tests run
without weights. Real-model behaviour is checked in the Task 6 Bagno 1 run."""
import numpy as np
import pytest
import torch


class FakeImageProcessor:
    def post_process_masks(self, pred_masks, original_sizes, reshaped_input_sizes):
        # pred_masks (1, 1, k, H, W) already at original size in this fake
        return [pred_masks[0] > 0]


class FakeProcessor:
    """Mimics SamProcessor(images=...) for an image of original size (H, W),
    resized so the long side is 1024."""
    def __init__(self, h, w):
        self.h, self.w = h, w
        s = 1024 / max(h, w)
        self.rh, self.rw = round(h * s), round(w * s)
        self.image_processor = FakeImageProcessor()
        self.calls = 0

    def __call__(self, images=None, return_tensors="pt"):
        self.calls += 1
        return {
            "pixel_values": torch.zeros(1, 3, 4, 4, dtype=torch.float64),
            "original_sizes": torch.tensor([[self.h, self.w]]),
            "reshaped_input_sizes": torch.tensor([[self.rh, self.rw]]),
        }


class FakeOut:
    def __init__(self, pred_masks, iou_scores):
        self.pred_masks, self.iou_scores = pred_masks, iou_scores


class FakeModel:
    """k candidate masks; candidate i marks column i. iou picks candidate 2."""
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.embed_calls = 0
        self.last_kwargs = None

    def get_image_embeddings(self, pixel_values):
        assert pixel_values.dtype == torch.float32      # MPS needs float32
        self.embed_calls += 1
        return torch.zeros(1, 256, 64, 64)

    def __call__(self, **kw):
        self.last_kwargs = kw
        k = 3 if kw["multimask_output"] else 1
        pred = torch.full((1, 1, k, self.h, self.w), -1.0)
        for i in range(k):
            pred[0, 0, i, :, i] = 1.0
        iou = torch.tensor([[[0.1, 0.5, 0.9][:k]]])
        return FakeOut(pred, iou)


def _segmenter(h=300, w=400):
    from core.segment import SamSegmenter
    model, proc = FakeModel(h, w), FakeProcessor(h, w)
    return SamSegmenter(loader=lambda: (model, proc), device="cpu"), model, proc


def _img():
    from PIL import Image
    return Image.new("RGB", (400, 300))


def test_single_point_uses_multimask_and_best_iou():
    seg, model, _ = _segmenter()
    m = seg.segment("a", _img, {"points": [{"x": 100, "y": 50, "label": 1}], "box": None})
    assert model.last_kwargs["multimask_output"] is True
    assert m.shape == (300, 400) and m.dtype == bool
    assert m[:, 2].all() and not m[:, 0].any()          # candidate 2 (iou 0.9) chosen


def test_points_are_scaled_to_the_1024_input():
    seg, model, _ = _segmenter()                          # 400x300 → 1024x768
    seg.segment("a", _img, {"points": [{"x": 100, "y": 50, "label": 1},
                                        {"x": 200, "y": 150, "label": 0}], "box": None})
    pts = model.last_kwargs["input_points"]
    lbl = model.last_kwargs["input_labels"]
    assert pts.shape == (1, 1, 2, 2) and lbl.shape == (1, 1, 2)
    assert torch.allclose(pts[0, 0, 0], torch.tensor([256.0, 128.0]))
    assert lbl.tolist() == [[[1, 0]]]
    assert model.last_kwargs["multimask_output"] is False
    assert model.last_kwargs["input_boxes"] is None


def test_box_only():
    seg, model, _ = _segmenter()
    seg.segment("a", _img, {"points": [], "box": {"x0": 0, "y0": 0, "x1": 400, "y1": 300}})
    box = model.last_kwargs["input_boxes"]
    assert box.shape == (1, 1, 4)
    assert torch.allclose(box[0, 0], torch.tensor([0.0, 0.0, 1024.0, 768.0]))
    assert model.last_kwargs["input_points"] is None
    assert model.last_kwargs["multimask_output"] is False


def test_embedding_cached_per_image():
    seg, model, proc = _segmenter()
    p = {"points": [{"x": 1, "y": 1, "label": 1}], "box": None}
    seg.prepare("a", _img)
    seg.segment("a", _img, p)
    seg.segment("a", _img, p)
    assert model.embed_calls == 1 and proc.calls == 1


def test_lru_evicts_oldest():
    seg, model, _ = _segmenter()
    for i in range(5):
        seg.prepare(f"img{i}", _img)
    assert model.embed_calls == 5
    seg.prepare("img4", _img)                             # still cached
    assert model.embed_calls == 5
    seg.prepare("img0", _img)                             # evicted → re-embed
    assert model.embed_calls == 6


def test_empty_prompts_raise():
    seg, _, _ = _segmenter()
    with pytest.raises(ValueError):
        seg.segment("a", _img, {"points": [], "box": None})


# ── Endpoints ──────────────────────────────────────────────────────────────────
import io
import uuid

from fastapi.testclient import TestClient


class FakeSegmenter:
    def __init__(self):
        self.prepared, self.seen = [], []

    def prepare(self, image_id, image_loader):
        self.prepared.append((image_id, image_loader().size))

    def segment(self, image_id, image_loader, prompts):
        if not prompts["points"] and not prompts["box"]:
            raise ValueError("Need at least one point or a box")
        im = image_loader()
        self.seen.append((image_id, im.size, prompts))
        m = np.zeros((im.height, im.width), dtype=bool)
        m[:, : im.width // 2] = True
        return m


@pytest.fixture
def api(monkeypatch):
    import core.segment as seg
    import server
    fake = FakeSegmenter()
    monkeypatch.setattr(seg, "get_segmenter", lambda: fake)
    made = []

    def put_image(size=(40, 30), exif_orientation=None):
        from PIL import Image
        name = f"{uuid.uuid4().hex}.jpg"
        path = server.TEMP_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", size, "white")
        if exif_orientation:
            exif = Image.Exif(); exif[0x0112] = exif_orientation
            im.save(path, exif=exif)
        else:
            im.save(path)
        made.append(path)
        return name

    yield TestClient(server.app), fake, put_image    # no `with`: startup (heartbeat) not run
    for p in made:
        p.unlink(missing_ok=True)


def test_prepare_endpoint(api):
    client, fake, put = api
    img = put()
    r = client.post("/api/segment/prepare", json={"image_id": img})
    assert r.status_code == 200 and r.json()["ready"] is True
    assert fake.prepared == [(img, (40, 30))]


def test_segment_endpoint_returns_png_mask(api):
    from PIL import Image
    client, fake, put = api
    img = put()
    r = client.post("/api/segment", json={"image_id": img,
                                          "points": [{"x": 5, "y": 5, "label": 1}], "box": None})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    m = Image.open(io.BytesIO(r.content))
    assert m.mode == "L" and m.size == (40, 30)
    assert m.getpixel((1, 1)) == 255 and m.getpixel((39, 1)) == 0
    assert fake.seen[0][2] == {"points": [{"x": 5.0, "y": 5.0, "label": 1}], "box": None}


def test_exif_orientation_is_applied(api):
    """Browsers show EXIF-rotated photos upright; SAM must see the same pixels."""
    client, fake, put = api
    img = put(size=(40, 30), exif_orientation=6)          # 6 = rotate 90° → upright is 30x40
    client.post("/api/segment/prepare", json={"image_id": img})
    assert fake.prepared[0][1] == (30, 40)


def test_bad_ids_rejected(api):
    client, _, _ = api
    for bad in ("../../etc/passwd", "missing.png"):
        r = client.post("/api/segment", json={"image_id": bad,
                                              "points": [{"x": 1, "y": 1, "label": 1}], "box": None})
        assert r.status_code == 400


def test_empty_prompts_400(api):
    client, _, put = api
    r = client.post("/api/segment", json={"image_id": put(), "points": [], "box": None})
    assert r.status_code == 400


def test_sam_failure_500(api, monkeypatch):
    client, fake, put = api
    def boom(*a, **k): raise RuntimeError("MPS exploded")
    monkeypatch.setattr(fake, "segment", boom)
    r = client.post("/api/segment", json={"image_id": put(),
                                          "points": [{"x": 1, "y": 1, "label": 1}], "box": None})
    assert r.status_code == 500 and "MPS exploded" in r.json()["detail"]
