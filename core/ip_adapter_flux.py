# core/ip_adapter_flux.py
"""
IP-Adapter management for FLUX.2-klein pipelines.
Uses InstantX/FLUX.1-dev-IP-Adapter weights via native diffusers IPAdapterMixin.

Weight layout on disk:
  ./models/ip_adapter/instantx/ip_adapter.bin       (~5.3 GB)
  (SigLIP encoder is downloaded automatically by HF Hub into HF_HUB_CACHE)
"""
from __future__ import annotations
from pathlib import Path

# Where we store the adapter weights (not in HF_HUB_CACHE — user-visible)
IP_ADAPTER_DIR  = Path("./models/ip_adapter/instantx")
IP_ADAPTER_FILE = IP_ADAPTER_DIR / "ip_adapter.bin"
REPO_ID         = "InstantX/FLUX.1-dev-IP-Adapter"
WEIGHT_NAME     = "ip_adapter.bin"


def is_downloaded() -> bool:
    """Return True if adapter weights file exists locally."""
    return IP_ADAPTER_FILE.exists()


def load_ip_adapter(pipe) -> None:
    """
    Inject IP-Adapter into an already-loaded Flux2KleinPipeline.
    Must be called after the model is loaded, before generation.
    Idempotent — calling twice is safe (diffusers handles it).
    """
    if not is_downloaded():
        raise RuntimeError("IP-Adapter weights not downloaded. Call download first.")
    pipe.load_ip_adapter(
        str(IP_ADAPTER_DIR),
        weight_name=WEIGHT_NAME,
        image_encoder_pretrained_model_name_or_path="google/siglip-so400m-patch14-384",
    )


def unload_ip_adapter(pipe) -> None:
    """Remove IP-Adapter from pipeline. Safe to call even if not loaded."""
    try:
        pipe.unload_ip_adapter()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("unload_ip_adapter: %s", exc)


def set_scale(pipe, scales: list[float]) -> None:
    """
    Set per-image IP-Adapter scale(s).
    Pass a list even for a single image — diffusers handles both forms.
    """
    if not scales:
        raise ValueError("scales must contain at least one value")
    pipe.set_ip_adapter_scale(scales if len(scales) > 1 else scales[0])


def download(progress_cb=None) -> None:
    """
    Download ip_adapter.bin with byte-level progress reporting.
    progress_cb(downloaded: int, total: int) — both in bytes.
    total may be 0 if server doesn't send Content-Length.
    """
    import requests
    from huggingface_hub import HfApi

    IP_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    url = api.hf_hub_url(repo_id=REPO_ID, filename=WEIGHT_NAME)
    headers = {}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest = IP_ADAPTER_FILE
    tmp  = dest.with_suffix(".tmp")

    try:
        with requests.get(url, headers=headers, stream=True, timeout=(10, None)) as r:
            r.raise_for_status()
            total      = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            progress_cb(downloaded, total)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.rename(dest)


def _hf_token() -> str | None:
    """Read HF token from local file (same location app.py uses)."""
    token_path = Path("huggingface/token")
    if token_path.exists():
        return token_path.read_text().strip() or None
    return None
