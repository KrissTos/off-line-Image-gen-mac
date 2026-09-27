# core/segment.py
"""SAM click-to-mask service for the mask editor.

The slow part of SAM is the image encoder; it runs once per image (cached by temp
image id, LRU) and every click/box only runs the prompt decoder (milliseconds).
MPS rules (proven in ai-generate floor_swap/masks.py): float inputs cast to float32
before .to("mps"); original/reshaped sizes stay on CPU for post_process_masks.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable

import numpy as np

SAM_MODEL = "facebook/sam-vit-large"


def _load_sam(model_id: str, device: str):
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device).eval()
    return model, proc


class SamSegmenter:
    def __init__(self, model_id: str = SAM_MODEL, device: str = "mps",
                 loader: Callable | None = None, cache_size: int = 4):
        self.model_id, self.device = model_id, device
        self._loader = loader or (lambda: _load_sam(model_id, device))
        self._model = self._proc = None
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            self._model, self._proc = self._loader()

    def _embedding(self, image_id: str, image_loader: Callable):
        import torch
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
            return self._cache[image_id]
        self._ensure_model()
        inputs = self._proc(images=image_loader(), return_tensors="pt")
        pixel = inputs["pixel_values"].float().to(self.device)
        with torch.no_grad():
            emb = self._model.get_image_embeddings(pixel)
        entry = (emb, inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        self._cache[image_id] = entry
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return entry

    def prepare(self, image_id: str, image_loader: Callable) -> None:
        with self._lock:
            self._embedding(image_id, image_loader)

    def segment(self, image_id: str, image_loader: Callable, prompts: dict) -> np.ndarray:
        import torch
        points = prompts.get("points") or []
        box = prompts.get("box")
        if not points and not box:
            raise ValueError("Need at least one point or a box")
        with self._lock:
            emb, orig, resh = self._embedding(image_id, image_loader)
            h, w = (int(v) for v in orig[0].tolist())
            rh, rw = (int(v) for v in resh[0].tolist())
            sx, sy = rw / w, rh / h
            kw = {"input_points": None, "input_labels": None, "input_boxes": None}
            if points:
                kw["input_points"] = torch.tensor(
                    [[[[p["x"] * sx, p["y"] * sy] for p in points]]], dtype=torch.float32).to(self.device)
                kw["input_labels"] = torch.tensor(
                    [[[int(p["label"]) for p in points]]], dtype=torch.long).to(self.device)
            if box:
                kw["input_boxes"] = torch.tensor(
                    [[[box["x0"] * sx, box["y0"] * sy, box["x1"] * sx, box["y1"] * sy]]],
                    dtype=torch.float32).to(self.device)
            multi = len(points) == 1 and not box
            with torch.no_grad():
                out = self._model(image_embeddings=emb, multimask_output=multi, **kw)
            if multi:
                idx = int(out.iou_scores.cpu()[0, 0].argmax())
                pred = out.pred_masks[:, :, idx:idx+1]
            else:
                pred = out.pred_masks
            masks = self._proc.image_processor.post_process_masks(pred.cpu(), orig, resh)[0]
            return masks[0, 0].numpy().astype(bool)


_segmenter: SamSegmenter | None = None


def get_segmenter() -> SamSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = SamSegmenter()
    return _segmenter
