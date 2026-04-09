"""
Flux Image Generator - Gradio Web Interface

Offline AI image generation for Mac Silicon (MPS).
Supports multiple models:
- Z-Image Turbo (quantized/full)
- FLUX.2-klein-4B (int8 quantized)

FLUX.2-klein also supports image-to-image editing!
"""

import os
import time
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"   # let MPS use all available unified memory
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import torch
from PIL import Image
import json
import shutil
from datetime import datetime

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen")
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_settings.json")


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_setting(key: str, value):
    settings = load_settings()
    settings[key] = value
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


# Global state
pipe = None
img2img_pipe = None   # cached ZImageImg2ImgPipeline — shared weights with pipe
inpaint_pipe = None   # cached FluxInpaintPipeline / ZImageInpaintPipeline
current_device = None
current_model = None  # "zimage-quant", "zimage-full", "flux2-klein-int8"
current_lora_paths: list = []  # [{path: str, strength: float}, ...]
_zimage_lora_networks: list = []  # LoRANetwork instances for Z-Image (must be removed on unload)
model_source = "local"  # "local" or "hf_cache"
last_sync_status = ""
online_version_cache = {}  # repo_id -> sha string, or None if error
upscaler_model_cache = {}  # safetensors path -> loaded spandrel model
video_pipe           = None
current_video_device = None

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")


def _dbg(msg: str) -> None:
    """Print a [DEBUG] line — only when pipeline.DEBUG is True."""
    try:
        import pipeline as _pl
        if _pl.DEBUG:
            print(f"[DEBUG] {msg}")
    except Exception:
        pass


# Model choices
MODEL_CHOICES = [
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)",
    "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)",
    "FLUX.2-klein-4B (Int8)",
    "Z-Image Turbo (Quantized - Fast)",
    "Z-Image Turbo (Full - LoRA support)",
    "LTX-Video  (txt2video · img2video with ref)",
]


def get_available_devices():
    """Get list of available devices."""
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


# =============================================================================
# Model Source / Cache Directory Helpers
# =============================================================================

def get_local_models_dir():
    """Project-local model cache (ultra-fast-image-gen-main/models)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def get_hf_global_cache_dir():
    """System-wide HuggingFace cache (~/.cache/huggingface/hub)."""
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def get_active_cache_dir():
    """Return the cache dir to USE for loading, based on model_source."""
    if model_source == "hf_cache":
        return get_hf_global_cache_dir()
    return get_local_models_dir()


# Map UI model choices to their primary HuggingFace repo IDs
MODEL_PRIMARY_REPO = {
    "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)":      "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
    "FLUX.2-klein-9B (4bit SDNQ - Higher Quality)": "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
    "FLUX.2-klein-4B (Int8)":                        "aydin99/FLUX.2-klein-4B-int8",
    "Z-Image Turbo (Quantized - Fast)":             "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
    "Z-Image Turbo (Full - LoRA support)":          "Tongyi-MAI/Z-Image-Turbo",
}


def get_model_snapshot_info(cache_dir, repo_id):
    """Return (commit_hash, mtime) for a model in a cache dir, or (None, None) if absent."""
    refs_main = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}", "refs", "main")
    if not os.path.exists(refs_main):
        return None, None
    with open(refs_main) as f:
        commit_hash = f.read().strip()
    return commit_hash, os.path.getmtime(refs_main)


def compare_model_versions(repo_id):
    """
    Compare local (project/models) vs HF global cache versions.
    Returns (status, message) where status is one of:
      'none', 'same', 'hf_only', 'local_only', 'hf_newer', 'local_newer'
    """
    local_hash, local_mtime = get_model_snapshot_info(get_local_models_dir(), repo_id)
    hf_hash, hf_mtime = get_model_snapshot_info(get_hf_global_cache_dir(), repo_id)

    if local_hash is None and hf_hash is None:
        return "none", "Not downloaded"
    if local_hash is None:
        return "hf_only", f"Only in HF Cache ({hf_hash[:8]})"
    if hf_hash is None:
        return "local_only", f"Only local ({local_hash[:8]})"
    if local_hash == hf_hash:
        return "same", f"Same ({local_hash[:8]})"
    if hf_mtime > local_mtime:
        return "hf_newer", f"HF newer ({hf_hash[:8]} vs local {local_hash[:8]})"
    return "local_newer", f"Local newer ({local_hash[:8]} vs HF {hf_hash[:8]})"


def sync_from_hf_cache(repo_id):
    """Copy a model from HF global cache to local project folder. Returns status string."""
    display = KNOWN_MODELS.get(repo_id, repo_id)
    src = os.path.join(get_hf_global_cache_dir(), f"models--{repo_id.replace('/', '--')}")
    dst = os.path.join(get_local_models_dir(), f"models--{repo_id.replace('/', '--')}")

    if not os.path.exists(src):
        return f"{display}: not found in HF Cache"
    try:
        os.makedirs(get_local_models_dir(), exist_ok=True)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        return f"{display}: synced to local"
    except Exception as e:
        return f"{display}: error — {e}"


def auto_sync_if_hf_newer(model_choice):
    """
    If HF cache has a newer version of the selected model, copy it to local.
    Returns a human-readable status string (may be empty if nothing to do).
    """
    repo_id = MODEL_PRIMARY_REPO.get(model_choice)
    if not repo_id:
        return ""
    status, _ = compare_model_versions(repo_id)
    if status in ("hf_newer", "hf_only"):
        return sync_from_hf_cache(repo_id)
    return ""


def check_model_source_status(model_choice, source_choice):
    """
    Update model_source global and return a status string describing
    the version relationship between local and HF cache for the chosen model.
    """
    global model_source
    model_source = "hf_cache" if "HF Cache" in source_choice else "local"

    repo_id = MODEL_PRIMARY_REPO.get(model_choice)
    if not repo_id:
        return ""

    status, detail = compare_model_versions(repo_id)

    if model_source == "local":
        labels = {
            "none":        "Not downloaded locally yet",
            "same":        f"✓ Local: {detail}",
            "local_only":  f"✓ Local only: {detail}",
            "hf_only":     "Not downloaded locally (available in HF Cache)",
            "hf_newer":    f"⚠ HF Cache is newer — switch to HF Cache to update",
            "local_newer": f"✓ Local is newer: {detail}",
        }
    else:
        labels = {
            "none":        "Not in local or HF Cache",
            "same":        f"✓ Up to date — no sync needed",
            "hf_only":     f"⬇ Only in HF Cache — will copy to local on next load",
            "local_only":  f"✓ Only local (HF Cache not available)",
            "hf_newer":    f"🔄 HF Cache is newer — will copy to local on next load",
            "local_newer": f"✓ Local is newer — no sync needed",
        }
    return labels.get(status, detail)


def get_versions_display():
    """Return a markdown table comparing local vs HF cache for all known models."""
    hf_dir = get_hf_global_cache_dir()
    if not os.path.exists(hf_dir):
        return "_HF Cache not found (`~/.cache/huggingface/hub` does not exist)._"

    lines = [
        "| Model | Local | HF Cache | Status |",
        "|-------|-------|----------|--------|",
    ]
    for repo_id, display_name in KNOWN_MODELS.items():
        local_hash, _ = get_model_snapshot_info(get_local_models_dir(), repo_id)
        hf_hash, _ = get_model_snapshot_info(hf_dir, repo_id)
        local_str = f"`{local_hash[:8]}`" if local_hash else "—"
        hf_str    = f"`{hf_hash[:8]}`"    if hf_hash    else "—"
        status, _ = compare_model_versions(repo_id)
        status_icon = {
            "none":        "—",
            "same":        "✓ Same",
            "hf_only":     "⬇ HF only",
            "local_only":  "📁 Local only",
            "hf_newer":    "🔄 HF newer",
            "local_newer": "📁 Local newer",
        }.get(status, status)
        lines.append(f"| {display_name} | {local_str} | {hf_str} | {status_icon} |")

    return "\n".join(lines)


def sync_all_newer_from_hf():
    """Sync all models where HF cache is newer than local. Returns status string."""
    msgs = []
    for repo_id in KNOWN_MODELS:
        status, _ = compare_model_versions(repo_id)
        if status in ("hf_newer", "hf_only"):
            msgs.append(sync_from_hf_cache(repo_id))
    if not msgs:
        return "All local models are already up to date (or HF Cache not available)."
    return "\n".join(msgs)


# =============================================================================
# Online Version Check (HuggingFace Hub API)
# =============================================================================

def _build_online_table():
    """
    Build the online-vs-local comparison table from the cached online_version_cache dict.
    Returns (markdown_string, list_of_display_names_that_can_be_updated).
    """
    if not online_version_cache:
        return "_Click **Check Latest Versions Online** to see results._", []

    lines = [
        "| Model | Local | Latest Online | Status |",
        "|-------|-------|---------------|--------|",
    ]
    updatable = []

    for repo_id, display_name in KNOWN_MODELS.items():
        local_hash, _ = get_model_snapshot_info(get_local_models_dir(), repo_id)
        local_str = f"`{local_hash[:8]}`" if local_hash else "—"

        if repo_id not in online_version_cache:
            online_str, status = "?", "—"
        elif online_version_cache[repo_id] is None:
            online_str, status = "—", "⚠ Error / not found"
        else:
            oh = online_version_cache[repo_id]
            online_str = f"`{oh[:8]}`"
            if local_hash is None:
                status = "⬇ Not downloaded"
                updatable.append(display_name)
            elif local_hash == oh:
                status = "✓ Up to date"
            else:
                status = "🔄 Update available"
                updatable.append(display_name)

        lines.append(f"| {display_name} | {local_str} | {online_str} | {status} |")

    return "\n".join(lines), updatable


def check_online_versions():
    """
    Query HF Hub for the latest commit hash of every known model.
    Populates online_version_cache and returns (table_markdown, dropdown_update).
    """
    from huggingface_hub import HfApi
    global online_version_cache

    api = HfApi()
    for repo_id in KNOWN_MODELS:
        try:
            info = api.model_info(repo_id)
            online_version_cache[repo_id] = info.sha
        except Exception:
            online_version_cache[repo_id] = None

    table, updatable = _build_online_table()
    return table, None


def download_model_update(model_selection):
    """Download / update a single model from HF Hub into the local cache."""
    from huggingface_hub import snapshot_download

    table, updatable = _build_online_table()   # default return values

    if not model_selection:
        return "No model selected.", table, None

    repo_id = next((rid for rid, dn in KNOWN_MODELS.items() if dn == model_selection), None)
    if not repo_id:
        return f"Unknown model: {model_selection}", table, None

    try:
        print(f"[Online update] Downloading {model_selection}...")
        snapshot_download(repo_id, cache_dir=get_local_models_dir())
        msg = f"✓ {model_selection}: updated successfully"
    except Exception as e:
        msg = f"⚠ {model_selection}: {e}"

    table, updatable = _build_online_table()   # refresh after download
    return msg, table, None


def download_model_update_stream(model_selection, progress_queue):
    """Like download_model_update but pushes SSE-style dicts to progress_queue."""
    import tqdm as _tqdm_mod
    from huggingface_hub import snapshot_download
    from pathlib import Path as _Path

    if not model_selection:
        progress_queue.put({"type": "error", "message": "No model selected."})
        return

    repo_id = next((rid for rid, dn in KNOWN_MODELS.items() if dn == model_selection), None)
    if not repo_id:
        progress_queue.put({"type": "error", "message": f"Unknown model: {model_selection}"})
        return

    class _ProgressTqdm(_tqdm_mod.tqdm):
        def display(self, *args, **kwargs):
            pass  # suppress console clutter
        def update(self, n=1):
            super().update(n)
            try:
                raw = (self.desc or "").strip()
                filename = _Path(raw).name if raw else "downloading…"
                total = int(self.total or 0)
                downloaded = int(self.n or 0)
                progress_queue.put({
                    "type": "file_progress",
                    "filename": filename,
                    "downloaded": downloaded,
                    "total": total,
                    "pct": round(100 * downloaded / total, 1) if total else 0,
                })
            except Exception:
                pass

    try:
        print(f"[Online update] Downloading {model_selection}...")
        snapshot_download(repo_id, cache_dir=get_local_models_dir(), tqdm_class=_ProgressTqdm)
        progress_queue.put({"type": "done", "message": f"✓ {model_selection}: updated successfully"})
    except Exception as e:
        progress_queue.put({"type": "error", "message": str(e)})


def download_all_online_updates():
    """Download every model that has an update available (per online_version_cache)."""
    from huggingface_hub import snapshot_download

    if not online_version_cache:
        table, updatable = _build_online_table()
        return "Run 'Check Latest Versions Online' first.", table, None

    msgs = []
    for repo_id, online_hash in online_version_cache.items():
        if not online_hash:
            continue
        local_hash, _ = get_model_snapshot_info(get_local_models_dir(), repo_id)
        if local_hash == online_hash:
            continue
        display = KNOWN_MODELS.get(repo_id, repo_id)
        try:
            print(f"[Online update] Downloading {display}...")
            snapshot_download(repo_id, cache_dir=get_local_models_dir())
            msgs.append(f"✓ {display}: updated")
        except Exception as e:
            msgs.append(f"⚠ {display}: {e}")

    table, updatable = _build_online_table()
    if not msgs:
        return "All models are already up to date.", table, None
    return "\n".join(msgs), table, None


def _prepare_outpaint(ref_img, target_w, target_h, align="center"):
    """
    Composite ref_img onto a black canvas at (target_w × target_h).
    Returns (canvas_image, mask_image):
      canvas_image — ref pasted on black background
      mask_image   — white where content must be generated, black where ref was placed
    Align is one of: top-left, top, top-right, left, center, right,
                     bottom-left, bottom, bottom-right
    """
    from PIL import Image, ImageDraw

    rw, rh = ref_img.size
    # Scale down if ref is larger than target (never upscale)
    scale = min(target_w / rw, target_h / rh, 1.0)
    if scale < 1.0:
        rw = int(rw * scale)
        rh = int(rh * scale)
        ref_img = ref_img.resize((rw, rh), Image.LANCZOS)

    parts = align.split("-")
    x = 0 if "left" in parts else (target_w - rw if "right" in parts else (target_w - rw) // 2)
    y = 0 if "top" in parts  else (target_h - rh if "bottom" in parts else (target_h - rh) // 2)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(ref_img, (x, y))

    # Mask: white = generate, black = preserve original
    mask = Image.new("L", (target_w, target_h), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + rw - 1, y + rh - 1], fill=0)

    return canvas, mask


def load_zimage_pipeline(device="mps", use_full_model=False):
    """Load Z-Image pipeline (quantized or full)."""
    import sdnq  # Required for quantized model
    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

    cache_dir = get_active_cache_dir()

    if use_full_model:
        print(f"Loading Z-Image-Turbo (full precision) on {device}...")
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )
    else:
        print(f"Loading Z-Image-Turbo UINT4 (quantized) on {device}...")
        dtype = torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    pipe.to(device)
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    return 0

def print_memory(label):
    """Print memory usage with label."""
    mem = get_memory_usage()
    print(f"  [MEM] {label}: {mem:.2f} GB")


def load_flux2_klein_pipeline(device="mps"):
    """Load FLUX.2-klein-4B with int8 quantized transformer and text encoder."""
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM, AutoTokenizer, AutoConfig
    from optimum.quanto import requantize
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from core.quantized_flux2 import QuantizedFlux2Transformer2DModel

    cache_dir = get_active_cache_dir()

    print(f"Loading FLUX.2-klein-4B (int8 quantized) on {device}...")
    print_memory("Before loading")

    model_path = snapshot_download("aydin99/FLUX.2-klein-4B-int8", cache_dir=cache_dir)

    print("  Loading int8 transformer...")
    qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
    qtransformer.to(device=device, dtype=torch.bfloat16)
    print_memory("After transformer")

    print("  Loading int8 text encoder...")
    config = AutoConfig.from_pretrained(f"{model_path}/text_encoder", trust_remote_code=True)
    with init_empty_weights():
        text_encoder = Qwen3ForCausalLM(config)

    with open(f"{model_path}/text_encoder/quanto_qmap.json", "r") as f:
        qmap = json.load(f)
    state_dict = load_file(f"{model_path}/text_encoder/model.safetensors")
    requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
    text_encoder.eval()
    text_encoder.to(device, dtype=torch.bfloat16)
    print_memory("After text encoder")

    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

    print("  Loading VAE and scheduler...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    print_memory("After VAE/scheduler download")
    
    pipe.transformer = qtransformer._wrapped
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    pipe.to(device)
    print_memory("After pipe.to(device)")
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")
    
    print("  FLUX.2-klein-4B ready!")
    return pipe


def load_flux2_klein_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    cache_dir = get_active_cache_dir()

    print(f"Loading FLUX.2-klein-4B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model (SDNQ model missing vocab files)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        subfolder="tokenizer",
        use_fast=False,
        cache_dir=cache_dir,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    print_memory("After loading")
    
    pipe.to(device)
    print_memory("After pipe.to(device)")
    
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")
    
    print("  FLUX.2-klein-4B (SDNQ) ready!")
    return pipe


def load_flux2_klein_9b_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    cache_dir = get_active_cache_dir()

    print(f"Loading FLUX.2-klein-9B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        subfolder="tokenizer",
        use_fast=False,
        cache_dir=cache_dir,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    print_memory("After loading")
    
    pipe.to(device)
    print_memory("After pipe.to(device)")
    
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")
    
    print("  FLUX.2-klein-9B (SDNQ) ready!")
    return pipe


def load_pipeline(model_choice: str, device: str = "mps"):
    global pipe, img2img_pipe, inpaint_pipe, current_device, current_model, current_lora_paths, last_sync_status

    # If HF Cache mode, sync to local before loading (only when HF version is newer/missing locally)
    last_sync_status = ""
    if model_source == "hf_cache":
        last_sync_status = auto_sync_if_hf_newer(model_choice)
        if last_sync_status:
            print(f"  [Sync] {last_sync_status}")

    if "Quantized" in model_choice:
        model_type = "zimage-quant"
    elif "Full" in model_choice:
        model_type = "zimage-full"
    elif "9B" in model_choice and "SDNQ" in model_choice:
        model_type = "flux2-klein-9b-sdnq"
    elif "4bit SDNQ" in model_choice:
        model_type = "flux2-klein-sdnq"
    elif "FLUX" in model_choice:
        model_type = "flux2-klein-int8"
    else:
        model_type = "zimage-quant"
    
    if pipe is not None and current_device == device and current_model == model_type:
        return pipe
    
    if pipe is not None:
        print(f"Switching from {current_model} to {model_type}...")
        if img2img_pipe is not None:
            del img2img_pipe
            img2img_pipe = None
        if inpaint_pipe is not None:
            del inpaint_pipe
            inpaint_pipe = None
        del pipe
        current_lora_paths = []
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if model_type == "flux2-klein-int8":
        pipe = load_flux2_klein_pipeline(device)
    elif model_type == "flux2-klein-sdnq":
        pipe = load_flux2_klein_sdnq_pipeline(device)
    elif model_type == "flux2-klein-9b-sdnq":
        pipe = load_flux2_klein_9b_sdnq_pipeline(device)
    elif model_type == "zimage-full":
        pipe = load_zimage_pipeline(device, use_full_model=True)
    else:
        pipe = load_zimage_pipeline(device, use_full_model=False)
    
    current_device = device
    current_model = model_type
    print(f"Pipeline loaded on {device}! (Model: {model_type})")
    return pipe


def _unload_zimage_loras():
    """Remove all custom Z-Image LoRA networks (forward-patch based)."""
    global _zimage_lora_networks
    for net in _zimage_lora_networks:
        try:
            net.remove()
        except Exception:
            pass
    _zimage_lora_networks = []


def load_loras(loras: list, device: str) -> str:
    """Load one or more LoRA adapters. loras = [{path, strength}, ...]
    Supports Z-Image Full and all FLUX.2-klein models."""
    global current_lora_paths, pipe, _zimage_lora_networks

    is_flux   = current_model is not None and current_model.startswith("flux2")
    is_zimage = current_model is not None and current_model == "zimage-full"

    if not is_flux and not is_zimage:
        return "LoRA requires FLUX.2-klein or Z-Image Full model"

    # Filter valid entries
    valid = [l for l in (loras or []) if l.get("path") and os.path.exists(l["path"])]

    if not valid:
        if is_flux:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
        else:
            _unload_zimage_loras()
        current_lora_paths = []
        return "No LoRAs loaded"

    # Unload previous adapters before reloading
    if is_flux:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    else:
        _unload_zimage_loras()

    if is_flux:
        from core.lora_flux2 import load_loras as _flux_load_loras
        try:
            status = _flux_load_loras(pipe, valid)
            current_lora_paths = valid
            _dbg(f"FLUX LoRA loaded: {[l.get('path') for l in valid]}")
            return status
        except RuntimeError as e:
            return f"LoRA error: {e}"
    else:
        # Z-Image Full — custom forward-patch loader (bypasses PEFT module-name matching)
        from core.lora_zimage import load_lora_for_pipeline
        loaded = []
        for lora in valid:
            try:
                net = load_lora_for_pipeline(
                    pipe,
                    lora_path=lora["path"],
                    multiplier=float(lora.get("strength", 1.0)),
                    device=device,
                )
                _zimage_lora_networks.append(net)
                loaded.append(os.path.basename(lora["path"]))
                _dbg(f"Z-Image LoRA patched: {lora['path']} strength={lora.get('strength', 1.0)}")
            except Exception as e:
                print(f"  LoRA load failed for {lora['path']}: {e}")
        current_lora_paths = valid
        if loaded:
            return f"Loaded {len(loaded)} LoRA(s): {', '.join(loaded)}"
        return "LoRA load failed — check console for details"


def load_lora(lora_file, lora_strength: float, device: str):
    """Legacy single-LoRA wrapper — kept for Gradio UI compatibility."""
    if lora_file:
        return load_loras([{"path": lora_file, "strength": lora_strength}], device)
    return load_loras([], device)


# =============================================================================
# LTX-Video pipeline
# =============================================================================

def load_ltx_pipeline(device="mps"):
    """Load LTX-Video for text-to-video (reference image handled at inference time)."""
    from diffusers import LTXPipeline

    cache_dir = get_active_cache_dir()
    print(f"Loading LTX-Video on {device}...")
    print_memory("Before loading")
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    p = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    p.to(device)
    p.enable_attention_slicing()
    if hasattr(p, "enable_vae_slicing"):
        p.enable_vae_slicing()
    print_memory("After loading")
    print("  LTX-Video ready!")
    return p


def export_frames_to_video(frames, output_path, fps=24):
    """Export a list of PIL images to MP4 via imageio."""
    import imageio
    import numpy as np
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(np.array(frame.convert("RGB")))
    writer.close()
    return output_path


# =============================================================================
# Mask helpers (for masked img2img)
# =============================================================================

def get_mask_bbox(mask_pil: Image.Image, padding: int = 32) -> tuple | None:
    """
    Return (x0, y0, x1, y1) bounding box of the non-black region in *mask_pil*
    with extra *padding* pixels on each side, clamped to image bounds.
    Returns None if the mask is empty.
    Dimensions are rounded up to the nearest 64 for model compatibility.
    """
    import numpy as np
    mask_np = np.array(mask_pil.convert("L"))
    rows = np.any(mask_np > 10, axis=1)
    cols = np.any(mask_np > 10, axis=0)
    if not rows.any() or not cols.any():
        return None  # empty mask

    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    ih, iw = mask_np.shape

    x0 = max(0, cmin - padding)
    y0 = max(0, rmin - padding)
    x1 = min(iw, cmax + padding + 1)
    y1 = min(ih, rmax + padding + 1)

    # Round crop dimensions to multiples of 64
    crop_w = max(64, ((x1 - x0 + 63) // 64) * 64)
    crop_h = max(64, ((y1 - y0 + 63) // 64) * 64)
    x1 = min(iw, x0 + crop_w)
    y1 = min(ih, y0 + crop_h)
    return (x0, y0, x1, y1)


def apply_mask_composite(
    original: Image.Image,
    generated: Image.Image,
    mask_l: Image.Image,
    bbox: tuple,
    blur_radius: int = 4,
) -> Image.Image:
    """
    Paste *generated* (the inpainted crop) back onto *original* using *mask_l*
    (grayscale, already sized to *original*).  *bbox* = (x0, y0, x1, y1).
    A Gaussian blur softens the mask edges for a natural blend.
    """
    from PIL import ImageFilter
    x0, y0, x1, y1 = bbox
    target_size = (x1 - x0, y1 - y0)
    if generated.size != target_size:
        generated = generated.resize(target_size, Image.LANCZOS)
    mask_crop = mask_l.crop(bbox)
    mask_soft = mask_crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    result = original.copy()
    result.paste(generated, (x0, y0), mask_soft)
    return result


# =============================================================================
# Upscaling (spandrel — supports 4x-UltraSharp and any RRDB/ESRGAN safetensors)
# =============================================================================

def load_upscaler(safetensors_path: str):
    """Load a .safetensors upscaler via spandrel. Models are cached by path."""
    import spandrel
    if safetensors_path not in upscaler_model_cache:
        print(f"[Upscaler] Loading model: {safetensors_path}")
        model = spandrel.ModelLoader().load_from_file(safetensors_path)
        model.eval()
        upscaler_model_cache[safetensors_path] = model
    return upscaler_model_cache[safetensors_path]


def upscale_image(image: Image.Image, safetensors_path: str, device: str) -> Image.Image:
    """Upscale a PIL image using a spandrel-compatible upscaler model.
    Falls back to CPU if the requested device fails (e.g. limited MPS support)."""
    import numpy as np

    model = load_upscaler(safetensors_path)

    img_rgb = image.convert("RGB") if image.mode != "RGB" else image
    img_np = np.array(img_rgb).astype("float32") / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Try requested device, fall back to CPU if unsupported
    for run_device in ([device, "cpu"] if device != "cpu" else ["cpu"]):
        try:
            model.to(run_device)
            with torch.inference_mode():
                out = model(img_tensor.to(run_device))
            if run_device != device:
                print(f"[Upscaler] Note: fell back to CPU (device '{device}' unsupported)")
            break
        except Exception as e:
            if run_device == "cpu":
                raise  # nothing left to try
            print(f"[Upscaler] {device} failed ({e}), retrying on CPU…")

    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = (out_np.clip(0, 1) * 255).astype("uint8")
    result = Image.fromarray(out_np)
    print(f"[Upscaler] {image.size} → {result.size}")
    return result


def generate_image(
    prompt,
    height,
    width,
    steps,
    seed,
    guidance,
    device,
    model_choice,
    model_source_choice,
    input_images,
    lora_file,
    lora_strength,
    img_strength,
    repeat_count,
    auto_save,
    output_dir,
    upscale_enabled,
    upscale_model_path,
    num_frames,
    fps_val,
    mask_image=None,
    mask_mode="Crop & Composite (Fast)",
    progress=None,
    step_callback=None,
    outpaint_align="center",
    lora_files=None,       # list of {path, strength} — multi-LoRA; takes precedence over lora_file/lora_strength
):
    global pipe, img2img_pipe, inpaint_pipe, video_pipe, current_video_device, model_source

    # Merge legacy lora_file/lora_strength into lora_files list
    # Also handles lora_files=[] sent by pipeline.py when no new-style LoRAs are set
    if not lora_files and lora_file:
        lora_files = [{"path": lora_file, "strength": lora_strength}]
    elif lora_files is None:
        lora_files = []

    # Apply model source selection before loading
    model_source = "hf_cache" if "HF Cache" in model_source_choice else "local"

    is_video_model = "LTX-Video" in model_choice

    if is_video_model:
        # Unload image pipeline to free memory
        if pipe is not None:
            import gc
            if img2img_pipe is not None:
                del img2img_pipe
                img2img_pipe = None
            if inpaint_pipe is not None:
                del inpaint_pipe
                inpaint_pipe = None
            del pipe
            pipe = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        # Load (or reuse) video pipeline
        if video_pipe is None or current_video_device != device:
            if video_pipe is not None:
                del video_pipe
            video_pipe = load_ltx_pipeline(device)
            current_video_device = device
    else:
        # Unload video pipeline if switching back to image
        if video_pipe is not None:
            import gc
            del video_pipe
            video_pipe = None
            current_video_device = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        if "Z-Image" in model_choice and lora_files:
            model_choice = "Z-Image Turbo (Full - LoRA support)"
        pipe = load_pipeline(model_choice, device)

    if not is_video_model and (current_model == "zimage-full" or (current_model and current_model.startswith("flux2"))) and lora_files:
        load_loras(lora_files, device)

    # Pre-process reference images once — same size/mode for every iteration
    img_w, img_h = int(width), int(height)

    # ── Auto-outpaint: if ref image aspect ratio ≠ output size and no explicit mask,
    #    composite the ref onto a black canvas and auto-generate the extension mask.
    if (input_images is not None and len(input_images) > 0
            and mask_image is None and not is_video_model):
        raw0 = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
        rw, rh = raw0.size
        # Only activate when there is actual size difference (>4px margin to avoid float rounding)
        if abs(rw - img_w) > 4 or abs(rh - img_h) > 4:
            canvas, auto_mask = _prepare_outpaint(raw0, img_w, img_h, outpaint_align)
            # Replace slot #1 with the composited canvas; keep extra slots intact
            rest = input_images[1:] if len(input_images) > 1 else []
            input_images = [canvas] + rest
            mask_image = auto_mask
            mask_mode  = "Inpainting Pipeline (Quality)"
            print(f"  Auto-outpaint: {rw}×{rh} → {img_w}×{img_h} (align={outpaint_align})")

    preprocessed_flux_refs  = None
    preprocessed_zimage_ref = None
    if input_images is not None and len(input_images) > 0:
        if current_model in ("flux2-klein-int8", "flux2-klein-sdnq", "flux2-klein-9b-sdnq"):
            preprocessed_flux_refs = []
            for img_data in input_images[:6]:
                img = img_data[0] if isinstance(img_data, tuple) else img_data
                resized = img.copy().resize((img_w, img_h), Image.LANCZOS)
                if resized.mode != "RGB":
                    resized = resized.convert("RGB")
                preprocessed_flux_refs.append(resized)
            print(f"  Pre-processed {len(preprocessed_flux_refs)} reference image(s) → {img_w}×{img_h}")
        elif current_model == "zimage-full":
            raw = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
            preprocessed_zimage_ref = raw.copy().resize((img_w, img_h), Image.LANCZOS)
            if preprocessed_zimage_ref.mode != "RGB":
                preprocessed_zimage_ref = preprocessed_zimage_ref.convert("RGB")
            print(f"  Pre-processed reference image → {img_w}×{img_h}")
        elif is_video_model:
            raw = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
            preprocessed_zimage_ref = raw.copy().resize((img_w, img_h), Image.LANCZOS)
            if preprocessed_zimage_ref.mode != "RGB":
                preprocessed_zimage_ref = preprocessed_zimage_ref.convert("RGB")
            print(f"  Pre-processed LTX reference image → {img_w}×{img_h}")

    # ── Masking pre-processing ───────────────────────────────────────────────
    # Runs once before the repeat loop; if crop mode, it overrides
    # preprocessed_flux_refs / preprocessed_zimage_ref with a smaller crop
    # and sets gen_w / gen_h so the model generates only the masked area.
    has_mask  = (mask_image is not None
                 and input_images is not None and len(input_images) > 0
                 and not is_video_model)
    mask_full  = None   # mask resized to full output dims (L mode)
    ref_full   = None   # first ref resized to full output dims (for compositing)
    mask_bbox  = None   # (x0,y0,x1,y1) crop region in crop mode
    gen_w, gen_h = img_w, img_h  # generation dims; smaller than output in crop mode

    if has_mask:
        import numpy as np
        raw_ref = input_images[0][0] if isinstance(input_images[0], tuple) else input_images[0]
        ref_full  = raw_ref.copy().resize((img_w, img_h), Image.LANCZOS).convert("RGB")
        mask_full = mask_image.convert("L").resize((img_w, img_h), Image.NEAREST)

        is_crop_mode = "Crop" in (mask_mode or "Crop")

        if is_crop_mode:
            bbox = get_mask_bbox(mask_full, padding=32)
            if bbox is None:
                print("  Mask is empty — ignoring masking")
                has_mask = False
            else:
                mask_bbox = bbox
                x0, y0, x1, y1 = bbox
                gen_w, gen_h    = x1 - x0, y1 - y0
                ref_crop        = ref_full.crop(bbox)

                # Replace preprocessed refs with the cropped region
                if current_model in ("flux2-klein-int8", "flux2-klein-sdnq",
                                     "flux2-klein-9b-sdnq"):
                    preprocessed_flux_refs = [ref_crop]
                elif current_model == "zimage-full":
                    preprocessed_zimage_ref = ref_crop

                speedup = max(1, (img_w * img_h) // (gen_w * gen_h))
                print(f"  Mask crop: ({x0},{y0})–({x1},{y1}) "
                      f"→ generate {gen_w}×{gen_h} "
                      f"(full {img_w}×{img_h}, ~{speedup}× faster)")
        # Inpainting pipeline mode: gen dims stay the same (full image),
        # but we'll use FluxInpaintPipeline / ZImageInpaintPipeline in the loop.

    base_seed    = int(seed)
    repeat_count = max(1, int(repeat_count or 1))

    for i in range(repeat_count):
        # Seed: random each iteration when -1, otherwise seed / seed+1 / seed+2 …
        if base_seed == -1:
            current_seed = torch.randint(0, 2**32, (1,)).item()
        else:
            current_seed = base_seed + i

        if device == "mps":
            generator = torch.Generator("mps").manual_seed(current_seed)
        else:
            generator = torch.Generator().manual_seed(current_seed)

        print_memory(f"Before generation {i + 1}/{repeat_count}")

        # Build diffusers step-end callback for progress reporting
        _n_steps = int(steps)
        def _step_cb(pipeline, step_index, timestep, callback_kwargs,
                     _cb=step_callback, _total=_n_steps):
            if _cb is not None:
                _cb(step_index + 1, _total)
            return callback_kwargs
        _cb = _step_cb if step_callback is not None else None


        iter_label = f"[{i + 1}/{repeat_count}] " if repeat_count > 1 else ""
        yield None, None, f"{iter_label}Generating…"

        _t0 = time.perf_counter()
        # Guard against UnboundLocalError: some conditional branches below may be
        # skipped entirely (e.g. inpainting fallback when FluxInpaintPipeline is
        # unavailable).  Initialise here so the post-try composite & save code is safe.
        image        = None
        video_frames = None
        mode         = "txt2img"
        try:
            with torch.inference_mode():
                if current_model in ("flux2-klein-int8", "flux2-klein-sdnq", "flux2-klein-9b-sdnq"):
                    # ── FLUX inpainting: FluxInpaintPipeline is incompatible with
                    #    Flux2KleinPipeline (different transformer/vae types), so fall
                    #    straight through to img2img for masked generations.
                    if (has_mask and mask_full is not None
                            and "Inpainting" in (mask_mode or "")
                            and ref_full is not None):
                        preprocessed_flux_refs = preprocessed_flux_refs or [ref_full]
                        _dbg("FLUX: inpainting fall-through → img2img (FluxInpaintPipeline incompatible)")

                    # ── FLUX img2img (reference images, incl. inpainting fall-through) ──
                    if preprocessed_flux_refs is not None:
                        if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
                            pipe.vae.disable_tiling()

                        image = pipe(
                            prompt=prompt,
                            image=(preprocessed_flux_refs
                                   if len(preprocessed_flux_refs) > 1
                                   else preprocessed_flux_refs[0]),
                            height=gen_h,
                            width=gen_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                            callback_on_step_end=_cb,
                        ).images[0]

                        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                            pipe.vae.enable_tiling()

                        mode = f"img2img ({len(preprocessed_flux_refs)} ref)"
                        video_frames = None
                        _dbg(f"FLUX branch: img2img, {len(preprocessed_flux_refs)} ref(s)")

                    # ── FLUX txt2img (no reference) ──
                    else:
                        image = pipe(
                            prompt=prompt,
                            height=img_h,
                            width=img_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                            callback_on_step_end=_cb,
                        ).images[0]
                        mode = "txt2img"
                        video_frames = None
                        _dbg("FLUX branch: txt2img (no reference)")

                elif current_model == "zimage-full" and preprocessed_zimage_ref is not None:
                    _dbg(f"Z-Image full branch: has_mask={has_mask} mask_mode={mask_mode!r}")
                    _zimage_inpaint_failed = False
                    # ── Z-Image inpainting pipeline (mask + quality mode) ──────
                    if (has_mask and mask_full is not None
                            and "Inpainting" in (mask_mode or "")
                            and ref_full is not None):
                        try:
                            from diffusers import ZImageInpaintPipeline
                            if inpaint_pipe is None:
                                print("  Creating ZImageInpaintPipeline (shared weights, one-time cost)...")
                                inpaint_pipe = ZImageInpaintPipeline.from_pipe(pipe)
                            image = inpaint_pipe(
                                prompt=prompt,
                                image=ref_full,
                                mask_image=mask_full,
                                strength=float(img_strength),
                                height=img_h,
                                width=img_w,
                                num_inference_steps=int(steps),
                                guidance_scale=float(guidance),
                                generator=generator,
                                callback_on_step_end=_cb,
                            ).images[0]
                            mode = "inpainting (Z-Image)"
                            _dbg("Z-Image branch: ZImageInpaintPipeline succeeded")
                        except Exception as _e:
                            print(f"  ZImageInpaintPipeline unavailable ({_e}) — falling back to img2img")
                            _dbg(f"Z-Image branch: inpainting failed ({_e}), falling back to img2img")
                            _zimage_inpaint_failed = True
                    # ── Z-Image img2img (reference, incl. crop mode) ─────────
                    if _zimage_inpaint_failed or not (has_mask and "Inpainting" in (mask_mode or "")):
                        from diffusers import ZImageImg2ImgPipeline
                        if img2img_pipe is None:
                            print("  Creating img2img pipeline (shared weights, one-time cost)...")
                            img2img_pipe = ZImageImg2ImgPipeline.from_pipe(pipe)
                        image = img2img_pipe(
                            prompt=prompt,
                            image=preprocessed_zimage_ref,
                            strength=float(img_strength),
                            height=gen_h,
                            width=gen_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                            callback_on_step_end=_cb,
                        ).images[0]
                        mode = "img2img (ref)"
                        _dbg("Z-Image branch: img2img (ref)")
                    video_frames = None
                elif is_video_model:
                    n_frames = max(9, int(num_frames or 25))
                    # LTX requires num_frames = 8k+1
                    n_frames = ((n_frames - 1) // 8) * 8 + 1
                    # Width/height must be divisible by 32
                    ltx_w = max(256, (img_w // 32) * 32)
                    ltx_h = max(256, (img_h // 32) * 32)
                    if preprocessed_zimage_ref is not None:
                        from diffusers import LTXImageToVideoPipeline
                        i2v = LTXImageToVideoPipeline.from_pipe(video_pipe)
                        result = i2v(
                            prompt=prompt,
                            image=preprocessed_zimage_ref,
                            num_frames=n_frames,
                            height=ltx_h,
                            width=ltx_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                            callback_on_step_end=_cb,
                        )
                        mode = "img2video"
                    else:
                        result = video_pipe(
                            prompt=prompt,
                            num_frames=n_frames,
                            height=ltx_h,
                            width=ltx_w,
                            num_inference_steps=int(steps),
                            guidance_scale=float(guidance),
                            generator=generator,
                            callback_on_step_end=_cb,
                        )
                        mode = "txt2video"
                    video_frames = result.frames[0]
                    image = None
                else:
                    image = pipe(
                        prompt=prompt,
                        height=img_h,
                        width=img_w,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        generator=generator,
                        callback_on_step_end=_cb,
                    ).images[0]
                    mode = "txt2img"
                    video_frames = None

        except RuntimeError as e:
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if "out of memory" in str(e).lower():
                tips = "Try: lower resolution, fewer frames, or close other apps to free RAM."
                yield None, None, f"Out of memory — {tips}"
                return
            raise

        # ── Crop & composite: paste generated crop back onto full reference ──
        if (image is not None
                and has_mask
                and mask_bbox is not None
                and "Crop" in (mask_mode or "Crop")
                and ref_full is not None
                and mask_full is not None):
            image = apply_mask_composite(ref_full, image, mask_full, mask_bbox)
            mode += " (masked-crop)"

        _elapsed = time.perf_counter() - _t0

        print_memory("After generation")

        # Force memory cleanup
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        print_memory("After cache clear")

        if lora_files:
            lora_parts = [f"{os.path.basename(l['path'])}({l.get('strength', 1.0):.2f})" for l in lora_files]
            lora_info  = f" | LoRA: {', '.join(lora_parts)}"
        else:
            lora_info  = ""
        cfg_info   = f" | CFG: {guidance}" if guidance > 0 else ""
        sync_note  = f" | Sync: {last_sync_status}" if last_sync_status else ""
        time_info  = f" | Time: {_elapsed:.1f}s"

        model_short = {
            "zimage-quant":        "Z-Image (quant)",
            "zimage-full":         "Z-Image (full)",
            "flux2-klein-int8":    "FLUX.2-klein-4B (int8)",
            "flux2-klein-sdnq":    "FLUX.2-klein-4B (4bit)",
            "flux2-klein-9b-sdnq": "FLUX.2-klein-9B (4bit)",
            "ltx-video":           "LTX-Video",
        }.get(current_model or ("ltx-video" if is_video_model else ""), current_model or "")

        if is_video_model and video_frames is not None:
            # Save video to output dir
            v_fps = max(1, int(fps_val or 24))
            timestamp_v = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prompt_slug = "".join(c if c.isalnum() else "_" for c in (prompt or "")[:30]).strip("_")
            vname = f"{timestamp_v}_{prompt_slug}.mp4" if prompt_slug else f"{timestamp_v}.mp4"
            vpath = os.path.join(get_output_dir(output_dir), vname)
            export_frames_to_video(video_frames, vpath, fps=v_fps)
            info = (f"{iter_label}Seed: {current_seed} | Model: {model_short} | Mode: {mode}"
                    f" | {len(video_frames)} frames @ {v_fps}fps | Device: {device}{cfg_info}"
                    f"{time_info}{sync_note} | Saved: {vpath}")
            yield None, vpath, info
        else:
            # Optional upscaling before save/display
            upscale_note = ""
            if upscale_enabled and upscale_model_path:
                if not os.path.isfile(upscale_model_path):
                    upscale_note = " | Upscale skipped: model file not found"
                else:
                    try:
                        image = upscale_image(image, upscale_model_path, device)
                        upscale_note = f" | Upscaled ✓ {image.size[0]}×{image.size[1]}"
                    except Exception as ue:
                        upscale_note = f" | Upscale failed: {ue}"
                        print(f"[Upscaler] Error: {ue}")
            elif upscale_enabled and not upscale_model_path:
                upscale_note = " | Upscale skipped: no model loaded"

            info = (f"{iter_label}Seed: {current_seed} | Model: {model_short} | Mode: {mode}"
                    f" | Device: {device}{cfg_info}{lora_info}{upscale_note}{time_info}{sync_note}")

            if auto_save:
                save_result = save_image(image, output_dir, prompt)
                info += f" | {save_result}"

            yield image, None, info


def clear_lora():
    """Clear all loaded LoRAs."""
    global current_lora_paths, pipe
    _unload_zimage_loras()
    if pipe is not None:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    current_lora_paths = []
    return None, "LoRA cleared"


# =============================================================================
# Output/Save Functions
# =============================================================================

def get_output_dir(custom_dir=None):
    """Get output directory, creating if needed."""
    output_dir = custom_dir.strip() if custom_dir and custom_dir.strip() else DEFAULT_OUTPUT_DIR
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def browse_upscaler_file(current_path):
    """Open a native macOS file picker for upscaler model files (.pth, .safetensors)."""
    import subprocess
    try:
        # Use choose file without type restriction so both .pth and .safetensors appear
        script = 'POSIX path of (choose file with prompt "Select upscaler model (.pth or .safetensors):")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            return path if path else current_path
    except Exception:
        pass
    return current_path


def browse_input_folder(current_dir):
    """Open a native macOS folder-picker dialog for input folder selection."""
    import subprocess
    try:
        script = 'POSIX path of (choose folder with prompt "Select folder with low-res images:")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            folder = result.stdout.strip().rstrip("/")
            return folder if folder else current_dir
    except Exception:
        pass
    return current_dir


def batch_upscale_folder(input_folder, output_folder, scale_choice, model_path):
    """
    Generator: upscale all images in input_folder and save to output_folder.
    scale_choice: "×2", "×3", or "×4"
    Uses spandrel 4x upscaler; ×2 and ×3 are achieved by resizing down from 4x.
    """
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    if not model_path or not os.path.isfile(model_path):
        yield "⚠️  No upscaler model selected. Set the Upscaler model path above first."
        return

    if not input_folder or not os.path.isdir(input_folder):
        yield "⚠️  Input folder not found. Please pick a valid folder."
        return

    files = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    if not files:
        yield "⚠️  No image files (.png .jpg .jpeg .webp .bmp .tiff) found in the selected folder."
        return

    out_dir = output_folder.strip() if output_folder and output_folder.strip() else input_folder
    os.makedirs(out_dir, exist_ok=True)

    scale_map = {"×2": 2, "×3": 3, "×4": 4}
    target_scale = scale_map.get(scale_choice, 4)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    log = f"Starting batch upscale — {len(files)} image(s) → {scale_choice}  |  device: {device}\n"
    yield log

    for idx, fname in enumerate(files, 1):
        src_path = os.path.join(input_folder, fname)
        stem, ext = os.path.splitext(fname)
        out_name = f"{stem}_x{target_scale}{ext}"
        dst_path = os.path.join(out_dir, out_name)

        try:
            img = Image.open(src_path).convert("RGB")
            orig_w, orig_h = img.size

            # Run the model at native 4× scale
            upscaled = upscale_image(img, model_path, device)

            # If target is ×2 or ×3, resize down from 4× to the desired scale
            if target_scale != 4:
                new_w = orig_w * target_scale
                new_h = orig_h * target_scale
                upscaled = upscaled.resize((new_w, new_h), Image.LANCZOS)

            # Save, preserving format where possible
            save_ext = ext.lower().lstrip(".")
            fmt = {"jpg": "JPEG", "jpeg": "JPEG"}.get(save_ext, "PNG")
            upscaled.save(dst_path, fmt)

            msg = f"[{idx}/{len(files)}] ✓  {fname}  ({orig_w}×{orig_h} → {upscaled.size[0]}×{upscaled.size[1]})"
        except Exception as e:
            msg = f"[{idx}/{len(files)}] ✗  {fname}  — Error: {e}"

        log += msg + "\n"
        yield log

    log += f"\nDone! {len(files)} image(s) saved to: {out_dir}"
    yield log


def browse_output_folder(current_dir):
    """Open a native macOS folder-picker dialog and return the chosen path."""
    import subprocess
    try:
        script = 'POSIX path of (choose folder with prompt "Select output folder:")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            folder = result.stdout.strip().rstrip("/")
            return folder if folder else current_dir
    except Exception:
        pass
    return current_dir  # user cancelled or error → keep current


def save_image(image, output_dir=None, prompt=""):
    """Save image to output directory."""
    if image is None:
        return "No image to save"
    
    output_dir = get_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = ""
    if prompt:
        prompt_slug = "_" + "".join(c if c.isalnum() else "_" for c in prompt[:30]).strip("_")
    
    filename = f"{timestamp}{prompt_slug}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath, "PNG")
    return f"Saved: {filepath}"


# =============================================================================
# HuggingFace Account Functions
# =============================================================================

def hf_get_status():
    """Return a string describing the current HF login status."""
    try:
        from huggingface_hub import whoami
        info = whoami()
        name = info.get("name") or info.get("fullname") or info.get("username", "unknown")
        return f"✅ Logged in as: **{name}**"
    except Exception:
        return "⚠️ Not logged in"

def hf_login_token(token: str):
    """Log in to HuggingFace with the provided token. Returns a status string."""
    token = (token or "").strip()
    if not token:
        return "⚠️ Please enter a token first."
    try:
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        info = whoami()
        name = info.get("name") or info.get("fullname") or info.get("username", "unknown")
        return f"✅ Logged in as: **{name}**"
    except Exception as e:
        return f"❌ Login failed: {e}"

def hf_logout():
    """Log out from HuggingFace."""
    try:
        from huggingface_hub import logout
        logout()
        return "🔓 Logged out."
    except Exception as e:
        return f"❌ Logout failed: {e}"


# =============================================================================
# Storage Management Functions
# =============================================================================

# Models this app uses (HuggingFace repo IDs)
KNOWN_MODELS = {
    "aydin99/FLUX.2-klein-4B-int8": "FLUX.2-klein-4B (Int8)",
    "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic": "FLUX.2-klein-4B (4bit SDNQ)",
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32": "FLUX.2-klein-9B (4bit SDNQ)",
    "Tongyi-MAI/Z-Image-Turbo": "Z-Image Turbo (Full)",
    "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32": "Z-Image Turbo (Quantized)",
    "filipstrand/Z-Image-Turbo-mflux-4bit": "Z-Image Turbo (mflux 4bit)",
    "Lightricks/LTX-Video": "LTX-Video",
}


def get_hf_cache_dir():
    """Get HuggingFace cache directory."""
    return os.environ.get("HF_HUB_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))


def get_dir_size(path):
    """Get total size of a directory in bytes (skips symlinks to avoid HF cache double-counting)."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp) and os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def format_size(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def scan_downloaded_models():
    """Scan HuggingFace cache for downloaded models used by this app."""
    cache_dir = get_hf_cache_dir()
    models = []
    total_size = 0
    
    if not os.path.exists(cache_dir):
        return [], "0 B"
    
    for repo_id, display_name in KNOWN_MODELS.items():
        # Convert repo_id to cache folder name (owner--model)
        cache_name = f"models--{repo_id.replace('/', '--')}"
        model_path = os.path.join(cache_dir, cache_name)
        
        if os.path.exists(model_path):
            size = get_dir_size(model_path)
            total_size += size
            models.append({
                "repo_id": repo_id,
                "display_name": display_name,
                "cache_name": cache_name,
                "path": model_path,
                "size": size,
                "size_str": format_size(size),
            })
    
    models.sort(key=lambda x: x["size"], reverse=True)
    
    return models, format_size(total_size)


def get_storage_display():
    """Get formatted storage display for Gradio."""
    models, total = scan_downloaded_models()
    
    if not models:
        return "No models downloaded yet. Models will download on first use."
    
    lines = [f"**Total Storage Used: {total}**\n"]
    lines.append("| Model | Size |")
    lines.append("|-------|------|")
    
    for m in models:
        lines.append(f"| {m['display_name']} | {m['size_str']} |")
    
    return "\n".join(lines)


def get_model_choices_for_deletion():
    """Get list of model choices for deletion dropdown."""
    models, _ = scan_downloaded_models()
    choices = []
    for m in models:
        choices.append(f"{m['display_name']} ({m['size_str']})")
    return choices


def delete_model(model_selection):
    """Delete a specific model from cache."""
    global pipe, current_model
    
    if not model_selection:
        return get_storage_display(), get_model_choices_for_deletion(), "No model selected"
    
    models, _ = scan_downloaded_models()
    
    target = None
    for m in models:
        if model_selection.startswith(m['display_name']):
            target = m
            break
    
    if not target:
        return get_storage_display(), get_model_choices_for_deletion(), f"Model not found: {model_selection}"
    
    # Unload pipeline if it's using this model
    model_repo = target['repo_id'].lower()
    if pipe is not None:
        needs_unload = False
        if "klein-4b" in model_repo and current_model and "4b" in current_model.lower():
            needs_unload = True
        elif "klein-9b" in model_repo and current_model and "9b" in current_model.lower():
            needs_unload = True
        elif "z-image" in model_repo.lower() and current_model and "zimage" in current_model.lower():
            needs_unload = True
        
        if needs_unload:
            del pipe
            pipe = None
            current_model = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    try:
        shutil.rmtree(target['path'])
        msg = f"Deleted: {target['display_name']} ({target['size_str']} freed)"
        print(msg)
    except Exception as e:
        msg = f"Error deleting {target['display_name']}: {str(e)}"
        print(msg)
    
    return get_storage_display(), get_model_choices_for_deletion(), msg


def delete_all_models():
    """Delete all downloaded models."""
    global pipe, current_model, current_lora_paths

    models, total = scan_downloaded_models()
    
    if not models:
        return get_storage_display(), get_model_choices_for_deletion(), "No models to delete"
    
    if pipe is not None:
        del pipe
        pipe = None
        current_model = None
        current_lora_paths = []
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    deleted = []
    errors = []
    
    for m in models:
        try:
            shutil.rmtree(m['path'])
            deleted.append(m['display_name'])
        except Exception as e:
            errors.append(f"{m['display_name']}: {str(e)}")
    
    if errors:
        msg = f"Deleted {len(deleted)} models. Errors: {'; '.join(errors)}"
    else:
        msg = f"Deleted {len(deleted)} models. {total} freed."
    
    print(msg)
    return get_storage_display(), get_model_choices_for_deletion(), msg


# =============================================================================
# Workflow save / load
# =============================================================================

def list_saved_workflows():
    """Return list of saved workflow folder names, newest first."""
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    entries = []
    for name in sorted(os.listdir(WORKFLOWS_DIR), reverse=True):
        if os.path.isdir(os.path.join(WORKFLOWS_DIR, name)):
            if os.path.exists(os.path.join(WORKFLOWS_DIR, name, "workflow.json")):
                entries.append(name)
    return entries


def save_workflow(wf_name, prompt, height, width, steps, seed, guidance,
                  device, model_choice, model_source, lora_file, lora_strength,
                  img_strength, repeat_count, upscale_enabled, upscale_model_path,
                  num_frames, fps_val, input_images):
    """Serialise all current parameters + reference images to a workflow folder."""
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom   = (wf_name or "").strip().replace(" ", "_")
    slug     = "".join(c if c.isalnum() else "_" for c in (prompt or "")[:30]).strip("_")
    folder_name = f"{timestamp}_{custom}" if custom else (f"{timestamp}_{slug}" if slug else timestamp)
    wf_dir = os.path.join(WORKFLOWS_DIR, folder_name)
    os.makedirs(wf_dir, exist_ok=True)

    # Save reference images
    ref_filenames = []
    if input_images:
        for idx, img_data in enumerate(input_images):
            img = img_data[0] if isinstance(img_data, tuple) else img_data
            if img is not None:
                fname = f"ref_{idx}.png"
                img.save(os.path.join(wf_dir, fname))
                ref_filenames.append(fname)

    data = {
        "name":              wf_name or folder_name,
        "timestamp":         timestamp,
        "prompt":            prompt or "",
        "height":            int(height),
        "width":             int(width),
        "steps":             int(steps),
        "seed":              int(seed),
        "guidance":          float(guidance),
        "device":            device or "",
        "model_choice":      model_choice or MODEL_CHOICES[0],
        "model_source":      model_source or "Local",
        "lora_file":         lora_file or "",
        "lora_strength":     float(lora_strength),
        "img_strength":      float(img_strength),
        "repeat_count":      int(repeat_count or 1),
        "upscale_enabled":   bool(upscale_enabled),
        "upscale_model_path": upscale_model_path or "",
        "num_frames":        int(num_frames or 25),
        "fps":               int(fps_val or 24),
        "reference_images":  ref_filenames,
    }
    with open(os.path.join(wf_dir, "workflow.json"), "w") as f:
        json.dump(data, f, indent=2)

    return f"✓ Saved: {folder_name}"


def load_workflow(workflow_name):
    """
    Load a workflow and return values for all UI components.
    Returns 18 values: scalar params × 16 + input_images + status_str.
    """
    if not workflow_name:
        return (None,) * 17 + ("No workflow selected",)
    wf_dir    = os.path.join(WORKFLOWS_DIR, workflow_name)
    json_path = os.path.join(wf_dir, "workflow.json")
    if not os.path.exists(json_path):
        return (None,) * 17 + (f"Not found: {workflow_name}",)

    with open(json_path) as f:
        d = json.load(f)

    # Load reference images back as PIL
    ref_images = []
    for fname in d.get("reference_images", []):
        fpath = os.path.join(wf_dir, fname)
        if os.path.exists(fpath):
            ref_images.append(Image.open(fpath).copy())

    lora_note = ""
    if d.get("lora_file"):
        lora_note = f"  (LoRA was: {os.path.basename(d['lora_file'])})"

    return (
        d.get("prompt", ""),                         # → prompt
        d.get("height", 512),                        # → height
        d.get("width", 512),                         # → width
        d.get("steps", 4),                           # → steps
        d.get("seed", -1),                           # → seed
        d.get("guidance", 3.5),                      # → guidance_scale
        d.get("device", "mps"),                      # → device
        d.get("model_choice", MODEL_CHOICES[0]),     # → model_choice
        d.get("model_source", "Local"),              # → model_source_radio
        d.get("lora_strength", 1.0),                 # → lora_strength
        d.get("img_strength", 0.6),                  # → img_strength
        d.get("repeat_count", 1),                    # → repeat_count
        d.get("upscale_enabled", False),             # → upscale_enabled
        d.get("upscale_model_path", ""),             # → upscale_model_path
        d.get("num_frames", 25),                     # → num_frames
        d.get("fps", 24),                            # → fps_slider
        ref_images if ref_images else None,          # → input_images
        f"✓ Loaded: {workflow_name}{lora_note}",     # → wf_status
    )


def calculate_dimensions_from_ratio(width: int, height: int, target_resolution: str) -> tuple:
    """Calculate output dimensions maintaining aspect ratio for target resolution."""
    if "1536" in target_resolution:
        target_size = 1536
    elif "1280" in target_resolution:
        target_size = 1280
    elif "2048" in target_resolution or "2K" in target_resolution:
        target_size = 2048
    elif "512" in target_resolution:
        target_size = 512
    else:
        target_size = 1024
    
    aspect_ratio = width / height
    
    if aspect_ratio >= 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    new_width = max(256, min(2048, new_width))
    new_height = max(256, min(2048, new_height))
    
    return new_width, new_height


def import_comfyui_workflow(json_file_path):
    """
    Parse a ComfyUI workflow JSON and populate all UI fields.
    Returns the same 18-value tuple as load_workflow().

    Handles:
      - Custom / third-party nodes: reported in status, generation continues
        with the parameters that were extractable.
      - Unknown checkpoint names: fuzzy-matched against locally downloaded
        models; falls back to the first available local model.
    """
    if not json_file_path:
        return (None,) * 17 + ("No file selected",)
    try:
        from core.workflow_utils import load_any_workflow, get_locally_available_models
        wf = load_any_workflow(json_file_path)
    except Exception as e:
        return (None,) * 17 + (f"Error reading file: {e}",)

    if wf.get("_source") != "comfyui":
        return (None,) * 17 + (
            "This appears to be a native app workflow — use the 'Load' button instead.",
        )

    fname = os.path.basename(json_file_path)

    # ── Model resolution: prefer locally available models ────────────────────
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    available  = get_locally_available_models(models_dir)

    matched_choice = wf.get("model_choice")   # may be None if ckpt not recognised
    ckpt_name      = wf.get("_comfyui_ckpt_name", "") or ""
    display_ckpt   = os.path.basename(ckpt_name) if ckpt_name else "(none)"

    model_notes: list[str] = []
    if matched_choice:
        if available and matched_choice not in available:
            # Recognised by name but not downloaded — use first available
            model_notes.append(
                f"⚠ '{display_ckpt}' matched '{matched_choice}' but not downloaded locally"
            )
            model_notes.append(f"  → using first available: {available[0]}")
            matched_choice = available[0]
        else:
            model_notes.append(f"✓ Model: '{display_ckpt}' → {matched_choice}")
    else:
        # Checkpoint name not in our map at all
        if available:
            matched_choice = available[0]
            model_notes.append(f"⚠ Unknown checkpoint '{display_ckpt}'")
            model_notes.append(f"  → defaulted to first local model: {matched_choice}")
        else:
            matched_choice = MODEL_CHOICES[0]
            model_notes.append(f"⚠ Unknown checkpoint '{display_ckpt}' and no local models found")
            model_notes.append(f"  → defaulted to: {matched_choice}")

    model_choice_val = matched_choice or MODEL_CHOICES[0]

    # ── Custom / unknown nodes ────────────────────────────────────────────────
    unknown_nodes = wf.get("_unknown_nodes", [])
    custom_lines: list[str] = []
    if unknown_nodes:
        if len(unknown_nodes) <= 6:
            custom_lines.append(
                f"⚠ {len(unknown_nodes)} custom node(s) skipped: {', '.join(unknown_nodes)}"
            )
        else:
            examples = ", ".join(unknown_nodes[:4])
            custom_lines.append(
                f"⚠ {len(unknown_nodes)} custom nodes skipped (e.g. {examples}…)"
            )
        custom_lines.append("  Parameters from these nodes were not imported.")

    # ── LoRA note ─────────────────────────────────────────────────────────────
    lora_lines: list[str] = []
    if wf.get("lora_file"):
        lora_lines.append(f"ℹ LoRA in workflow: {os.path.basename(wf['lora_file'])}")
        lora_lines.append("  (LoRA must be loaded manually via the LoRA file picker)")

    # ── Assemble status message ───────────────────────────────────────────────
    status_parts = [f"✓ Imported ComfyUI workflow: {fname}"]
    status_parts.extend(model_notes)
    status_parts.extend(lora_lines)
    status_parts.extend(custom_lines)
    status_msg = "\n".join(status_parts)

    return (
        wf.get("prompt", ""),                       # prompt
        wf.get("height", 512),                      # height
        wf.get("width",  512),                      # width
        wf.get("steps",  20),                       # steps
        wf.get("seed",   -1),                       # seed
        wf.get("guidance", 7.0),                    # guidance_scale
        wf.get("device", "mps"),                    # device
        model_choice_val,                           # model_choice
        wf.get("model_source", "Local"),            # model_source_radio
        wf.get("lora_strength", 1.0),               # lora_strength
        wf.get("img_strength", 1.0),                # img_strength
        wf.get("repeat_count", 1),                  # repeat_count
        wf.get("upscale_enabled", False),           # upscale_enabled
        wf.get("upscale_model_path", ""),           # upscale_model_path
        wf.get("num_frames", 25),                   # num_frames
        wf.get("fps", 24),                          # fps_slider
        None,                                       # input_images (not in ComfyUI JSON)
        status_msg,                                 # status
    )


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"
