"""
Z-Image Turbo UINT4 - Fast Image Generation on Mac

Uses the quantized uint4 model (only 3.5GB!) for fast inference on Apple Silicon.
Now with LoRA support and --workflow support (app format + ComfyUI format)!

Workflow usage:
    python generate.py --workflow my_workflow.json
    python generate.py --workflow comfyui_export.json  # auto-detected
    python generate.py --workflow my_workflow.json --seed 42  # CLI overrides workflow
"""

import os
import sys
import argparse

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["HF_HUB_CACHE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import torch
from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler


def load_pipeline(device="mps"):
    """Load the full-precision Z-Image pipeline."""
    print("Loading Z-Image-Turbo (full precision)...")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"PyTorch version: {torch.__version__}")

    # Use bfloat16 for better quality
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Use Euler with beta sigmas for cleaner images
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    pipe.to(device)

    # Memory optimizations
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("VAE slicing enabled")

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
        print("VAE tiling enabled")

    print("Pipeline loaded!")
    return pipe


def generate(
    pipe,
    prompt: str,
    height: int = 512,
    width: int = 512,
    steps: int = 5,
    seed: int = None,
    device: str = "mps",
):
    """Generate an image from a prompt."""
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Generating with seed {seed}...")

    # Use appropriate generator for device
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(seed)
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

    return image, seed


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Z-Image Turbo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple generation
  python generate.py "A cat wearing sunglasses" --height 512 --width 512

  # From a saved app workflow
  python generate.py --workflow workflows/20250301_portrait/workflow.json

  # From a ComfyUI export (auto-detected)
  python generate.py --workflow comfyui_workflow.json

  # Workflow + CLI overrides (CLI wins)
  python generate.py --workflow my_workflow.json --seed 42 --steps 8
        """,
    )
    parser.add_argument(
        "prompt", type=str, nargs="?", default=None,
        help="Text prompt for image generation (optional if --workflow is given)",
    )
    parser.add_argument(
        "--workflow", type=str, default=None,
        help="Path to workflow JSON (app format or ComfyUI format — auto-detected)",
    )
    parser.add_argument("--height", type=int, default=None, help="Image height (default from workflow or 512)")
    parser.add_argument("--width",  type=int, default=None, help="Image width  (default from workflow or 512)")
    parser.add_argument("--steps",  type=int, default=None, help="Inference steps (default from workflow or 5)")
    parser.add_argument("--seed",   type=int, default=None, help="Random seed (default from workflow or random)")
    parser.add_argument("--output", type=str, default="output.png", help="Output path (default: output.png)")
    parser.add_argument("--device", type=str, default=None,  help="Device: mps, cuda, cpu (auto-detected if omitted)")

    # LoRA arguments
    parser.add_argument("--lora",          type=str,   default=None, help="Path to LoRA .safetensors file")
    parser.add_argument("--lora-strength", type=float, default=None, help="LoRA strength (default from workflow or 1.0)")

    args = parser.parse_args()

    # ── Load workflow if specified ────────────────────────────────────────────
    wf: dict = {}
    if args.workflow:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from workflow_utils import load_any_workflow, workflow_summary
            wf = load_any_workflow(args.workflow)
            print(f"✓ Workflow loaded  ({wf.get('_source', '?')} format)")
            print(f"  {workflow_summary(wf)}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Error loading workflow: {e}")
            return

    # ── Resolve parameters: CLI arg > workflow value > hard default ───────────
    prompt        = args.prompt  or wf.get("prompt")  or ""
    height        = args.height  or int(wf.get("height",  512))
    width         = args.width   or int(wf.get("width",   512))
    steps         = args.steps   or int(wf.get("steps",   5))
    lora_strength = args.lora_strength if args.lora_strength is not None \
                    else float(wf.get("lora_strength", 1.0))

    # Seed: -1 in workflow means random
    if args.seed is not None:
        seed = args.seed
    else:
        wf_seed = wf.get("seed", -1)
        seed = None if (wf_seed is None or wf_seed == -1) else int(wf_seed)

    # LoRA: CLI path takes priority, then workflow (workflow stores filename only)
    lora = args.lora or None  # workflow LoRA is a filename, not a full path

    # Device: CLI > workflow > auto
    device = args.device or wf.get("device") or None
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Validate prompt
    if not prompt:
        print("Error: No prompt provided.")
        print("  Provide it as a positional argument: python generate.py 'my prompt'")
        print("  Or include it in a --workflow JSON file.")
        parser.print_help()
        return

    # ── Print resolved parameters ─────────────────────────────────────────────
    print(f"\nParameters:")
    print(f"  prompt  : {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
    print(f"  size    : {width}×{height}")
    print(f"  steps   : {steps}")
    print(f"  seed    : {seed if seed is not None else 'random'}")
    print(f"  device  : {device}")
    if lora:
        print(f"  LoRA    : {lora} (strength={lora_strength})")
    print()

    # ── Load pipeline and run generation ─────────────────────────────────────
    pipe = load_pipeline(device)

    if lora:
        if not os.path.exists(lora):
            print(f"Error: LoRA file not found: {lora}")
            return
        print(f"Loading LoRA: {lora} (strength={lora_strength})")
        try:
            pipe.load_lora_weights(lora, adapter_name="default")
            pipe.set_adapters(["default"], adapter_weights=[lora_strength])
            print("LoRA loaded successfully!")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return

    image, used_seed = generate(pipe, prompt, height, width, steps, seed, device)

    image.save(args.output)
    lora_info = f", LoRA: {os.path.basename(lora)}" if lora else ""
    wf_info   = f", workflow: {os.path.basename(args.workflow)}" if args.workflow else ""
    print(f"✓ Saved to {args.output}  (seed: {used_seed}{lora_info}{wf_info})")


if __name__ == "__main__":
    main()
