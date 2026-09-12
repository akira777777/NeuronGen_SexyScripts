#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuronGen CLI Generator v2.2 (High-Performance Unified Edition)
==============================================================
Fast command-line generator supporting:
  - Automatic1111 WebUI API
  - ComfyUI API
  - Local PyTorch Diffusers Pipeline
  - Fast Demo & Test Mode
  - Model and Connection Diagnostics

Usage Examples:
  python generate.py --prompt "prompts/best_quality_prompt.txt" --engine demo
  python generate.py --prompt "prompts/sexy_variant.txt" --engine webui --steps 30 --cfg 8.0
  python generate.py list-models
  python generate.py test-connection
"""

import argparse
import io
import os
import random
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from PIL import Image

# Reconfigure console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
PROMPTS_DIR = BASE_DIR / "prompts"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Local imports
from config import get_config
from prompt_processor import PromptProcessor
from webui_api import StableDiffusionWebUI
from batch_generator import BatchGenerator


def list_models():
    """Display available checkpoints, controlnets, and upscalers with status and size."""
    print("\n" + "=" * 62)
    print("NEURONGEN AVAILABLE MODELS")
    print("=" * 62)

    categories = [
        ("Checkpoints", CHECKPOINTS_DIR, ["*.safetensors", "*.ckpt"]),
        ("ControlNet", MODELS_DIR / "controlnet", ["*.pth", "*.safetensors"]),
        ("ESRGAN Upscalers", MODELS_DIR / "ESRGAN", ["*.onnx", "*.pth"]),
    ]

    for cat_name, cat_dir, patterns in categories:
        print(f"\n[{cat_name}] (Path: {cat_dir.relative_to(BASE_DIR)})")
        found = []
        if cat_dir.exists():
            for pat in patterns:
                found.extend(cat_dir.glob(pat))

        if not found:
            print("   (None found)")
            continue

        for f in found:
            size_bytes = f.stat().st_size
            if size_bytes < 1024 * 1024:
                status = "[INVALID / CORRUPTED (<1MB)]"
            else:
                size_mb = size_bytes / (1024 * 1024)
                status = f"[OK] ({size_mb:.1f} MB)"
            print(f"   {status} {f.name}")

    print("\n" + "=" * 62 + "\n")


def test_connections(webui_url: str = "http://127.0.0.1:7860", comfy_url: str = "http://127.0.0.1:8188"):
    """Check connectivity to Automatic1111 and ComfyUI servers."""
    print("\n[INFO] Testing Backend Connectivity...")
    client = StableDiffusionWebUI({"webui": {"url": webui_url}, "comfyui": {"url": comfy_url}})

    # Test Automatic1111
    a1111_ok = client.is_running(webui_url)
    print(f"   Automatic1111 ({webui_url}): {'[ONLINE]' if a1111_ok else '[OFFLINE]'}")

    # Test ComfyUI
    comfy_ok = client.is_comfyui_running(comfy_url)
    print(f"   ComfyUI       ({comfy_url}): {'[ONLINE]' if comfy_ok else '[OFFLINE]'}")

    if not a1111_ok and not comfy_ok:
        print("\n[NOTE] External backends are offline.")
        print("   - Run 'python main.py' to launch NeuronGen Web Studio on http://127.0.0.1:7861")
        print("   - Or use '--engine demo' for instant local testing without backends.")
    print()


def upscale_image_esrgan(
    image_path: Path,
    output_dir: Path,
    model_name: str = "4x-UltraSharp.onnx"
) -> Optional[Path]:
    """
    Upscale an image 4x using ESRGAN (UltraSharp or similar).
    Returns path to upscaled image, or None if failed.
    """
    try:
        import torch
        from diffusers import ImageUpscalePipeline
        
        # Find model
        esrgan_dir = MODELS_DIR / "ESRGAN"
        esrgan_model = esrgan_dir / model_name
        if not esrgan_model.exists():
            print(f"   [WARN] ESRGAN model not found: {esrgan_model}")
            return None
        
        # Load pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"   [INFO] Loading ESRGAN model onto {device}...")
        
        pipe = ImageUpscalePipeline.from_single_file(
            str(esrgan_model),
            torch_dtype=dtype,
            use_safetensors=True
        )
        pipe.to(device)
        
        # Load original image
        img = Image.open(image_path).convert("RGB")
        orig_size = img.size
        print(f"   [INFO] Upscaling {orig_size[0]}x{orig_size[1]} -> 4x resolution...")
        
        with torch.inference_mode():
            result = pipe(
                image=img,
                width=orig_size[0] * 4,
                height=orig_size[1] * 4
            )
        
        upscaled_img = result.images[0]
        
        # Save both original and upscaled
        orig_ext = image_path.suffix.lower()
        upscaler_name = f"{image_path.stem}_upscaled_1"
        upscaled_path = output_dir / f"{upscaler_name}{orig_ext}"
        upscaled_img.save(upscaled_path, format="PNG", compress_level=1)
        
        print(f"   [OK] Upscaled saved: {upscaled_path.name}")
        return upscaled_path
        
    except Exception as e:
        print(f"   [ERROR] ESRGAN upscale failed: {e}")
        return None


def generate_image_diffusers(
    prompt: str,
    output_path: str,
    steps: int = 25,
    cfg_scale: float = 7.5,
    width: int = 896,
    height: int = 1152,
    seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    negative_prompt: Optional[str] = None,
) -> List[str]:
    """Generates image locally using diffusers with Torch CUDA optimizations and memory management."""
    print(f"\n[INFO] Generating via PyTorch diffusers...")
    print(f"   Prompt: \"{prompt[:60]}...\"")
    print(f"   Resolution: {width}x{height} | Steps: {steps} | CFG: {cfg_scale}")

    # Default realistic negative prompt if none specified (under 50 tokens to avoid CLIP truncation)
    if not negative_prompt:
        negative_prompt = (
            "worst quality, low quality, bad anatomy, deformed body, bad hands, "
            "missing limbs, extra limbs, blurry, cgi, 3d, cartoon, anime, illustration, "
            "plastic skin, airbrushed, oversaturated, watermark, text"
        )

    # Check for valid checkpoint
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if not ckpt or not ckpt.exists() or ckpt.stat().st_size < 100 * 1024 * 1024:
        valid = [p for p in CHECKPOINTS_DIR.glob("*.safetensors") if p.stat().st_size > 100 * 1024 * 1024]
        if valid:
            ckpt = valid[0]
        else:
            print("[ERROR] No valid checkpoint (>100MB) found in models/checkpoints/")
            return []

    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        free_vram_mb = 0
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   [GPU] {gpu_name} (Free VRAM: {free_vram_mb:.0f} MB)")
            except Exception:
                free_vram_mb = 4000

        is_sdxl = ckpt.stat().st_size > 4 * 1024 * 1024 * 1024
        print(f"   Detected architecture: {'SDXL' if is_sdxl else 'SD 1.5'}")
        print(f"   Loading {ckpt.name} onto {device} ({dtype})...")

        if is_sdxl:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(ckpt),
                torch_dtype=dtype,
                safety_checker=None,
                use_safetensors=True
            )
        else:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_single_file(
                str(ckpt),
                torch_dtype=dtype,
                safety_checker=None,
                use_safetensors=True
            )
            if width > 768 or height > 768:
                width, height = 512, 768
                print(f"   Adjusted resolution to {width}x{height} for SD 1.5 native quality")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe = pipe.to(device)

        generator = torch.Generator(device=device)
        if seed is not None and seed >= 0:
            generator.manual_seed(seed)
        else:
            seed = int(time.time() * 1000) % (2**31)
            generator.manual_seed(seed)

        print(f"   Running {steps} inference steps (Seed: {seed})...")
        try:
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
        except torch.cuda.OutOfMemoryError as oom:
            print(f"   [WARN] CUDA OOM encountered ({oom}). Retrying with sequential CPU offload...")
            torch.cuda.empty_cache()
            pipe.enable_sequential_cpu_offload()
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator
                )

        image = result.images[0]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, format="PNG", compress_level=1)
        print(f"   [OK] Saved: {output_path}")
        return [output_path]

    except Exception as e:
        print(f"[ERROR] Diffusers generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="NeuronGen CLI Generator v2.2")

    # Main options
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt text or path to prompt file (e.g. prompts/best_quality_prompt.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: results/gen_<timestamp>.png)")
    parser.add_argument("--engine", "--method", dest="engine", type=str,
                        choices=["webui", "comfyui", "diffusers", "demo"],
                        default="demo",
                        help="Backend engine to use (default: demo)")
    
    # Generation parameters
    parser.add_argument("--steps", type=int, default=25, help="Sampling steps (default: 25)")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG Scale (default: 7.5)")
    parser.add_argument("--width", type=int, default=896, help="Image width (default: 896)")
    parser.add_argument("--height", type=int, default=1152, help="Image height (default: 1152)")
    parser.add_argument("--batch_size", type=int, default=1, help="Images per batch (default: 1)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 for random)")
    parser.add_argument("--sampler", type=str, default="DPM++ 2M Karras", help="Sampler name")

    # Server endpoints
    parser.add_argument("--webui_host", type=str, default="http://127.0.0.1:7860", help="Automatic1111 URL")
    parser.add_argument("--comfy_host", type=str, default="http://127.0.0.1:8188", help="ComfyUI URL")

    # ControlNet OpenPose parameters
    parser.add_argument("--controlnet_single", type=str, default=None,
                        help="Single pose image path for ControlNet conditioning (e.g., pose.png)")
    parser.add_argument("--controlnet_batch", type=str, default=None,
                        help="Glob pattern for multiple pose images (e.g., poses/*.png) - generates one image per pose")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Management commands")
    subparsers.add_parser("list-models", help="List available models and their paths")
    subparsers.add_parser("test-connection", help="Test backend connectivity")

    args = parser.parse_args()

    # Handle subcommands
    if args.command == "list-models":
        list_models()
        return

    if args.command == "test-connection":
        test_connections(args.webui_host, args.comfy_host)
        return

    # Check prompt
    if not args.prompt:
        print("[ERROR] No prompt provided. Specify --prompt or use --help.")
        print("Example: python generate.py --prompt \"prompts/best_quality_prompt.txt\" --engine demo")
        return

    # Handle ControlNet modes (single or batch poses)
    controlnet_poses = []
    if args.controlnet_single:
        import glob
        pose_files = sorted(glob.glob(args.controlnet_single))
        if not pose_files:
            print(f"[ERROR] No pose files found at: {args.controlnet_single}")
            return
        pose_file = pose_files[0]
        with open(pose_file, "rb") as f:
            img_bytes = f.read()
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            controlnet_poses.append((pose_file, img))
            print(f"\n[INFO] ControlNet Single Mode: Loaded pose {os.path.basename(pose_file)}")
        except Exception as e:
            print(f"[ERROR] Failed to load pose {pose_file}: {e}")
            return
    elif args.controlnet_batch:
        import glob
        pose_files = sorted(glob.glob(args.controlnet_batch))
        if not pose_files:
            print(f"[ERROR] No pose files found matching pattern: {args.controlnet_batch}")
            return
        for pf in pose_files:
            with open(pf, "rb") as f:
                img_bytes = f.read()
            from PIL import Image
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                controlnet_poses.append((pf, img))
            except Exception as e:
                print(f"[WARN] Failed to load pose {pf}: {e}")

    # Read prompt from file if path exists
    prompt_str = args.prompt
    if os.path.exists(prompt_str):
        with open(prompt_str, "r", encoding="utf-8") as f:
            prompt_str = f.read().strip()
    elif (PROMPTS_DIR / prompt_str).exists():
        with open(PROMPTS_DIR / prompt_str, "r", encoding="utf-8") as f:
            prompt_str = f.read().strip()

    # Process ControlNet batch poses into list of prompts
    controlnet_poses = []
    if args.controlnet_batch:
        import glob
        pose_files = sorted(glob.glob(args.controlnet_batch))
        for pf in pose_files:
            with open(pf, "rb") as f:
                img_bytes = f.read()
            from PIL import Image
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                controlnet_poses.append((pf, img))
            except Exception as e:
                print(f"[WARN] Failed to load pose {pf}: {e}")

    # Process prompt parameters
    proc = PromptProcessor()
    norm_prompt, _ = proc.process_prompt(prompt_str)
    _, inline_params = proc.extract_parameters(prompt_str)

    steps = inline_params.get("steps", args.steps)
    cfg = inline_params.get("cfg_scale", args.cfg)
    width = inline_params.get("width", args.width)
    height = inline_params.get("height", args.height)
    seed = inline_params.get("seed", args.seed)

    print("\n" + "=" * 62)
    print("NEURONGEN GENERATION PIPELINE")
    print("=" * 62)
    print(f"Engine: {args.engine.upper()}")
    print(f"Prompt: \"{norm_prompt[:70]}...\"")
    print(f"Params: {width}x{height} | Steps: {steps} | CFG: {cfg} | Seed: {seed}")
    print("=" * 62)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(RESULTS_DIR / f"gen_{timestamp}.png")

    b_gen = BatchGenerator()

    # Handle ControlNet generation (single or batch modes)
    if controlnet_poses:
        print(f"\n[INFO] Generating {len(controlnet_poses)} image(s) with ControlNet conditioning...")
        saved_files = []

        for idx, (pose_path, pose_img) in enumerate(controlnet_poses, 1):
            norm_prompt, _ = proc.process_prompt(norm_prompt)
            _, inline_params = proc.extract_parameters(prompt_str)
            actual_steps = inline_params.get("steps", args.steps)
            actual_cfg = inline_params.get("cfg_scale", args.cfg)
            actual_width = inline_params.get("width", args.width)
            actual_height = inline_params.get("height", args.height)

            print(f"   [{idx}/{len(controlnet_poses)}] Processing pose: {os.path.basename(pose_path)}")

            # Progress bar for batch processing
            try:
                from tqdm import tqdm
            except ImportError:
                def tqdm(iterable=None, **kwargs):
                    return iterable

            # Demo mode for ControlNet poses (inline implementation to avoid gradio import)
            if args.engine == "demo":
                from PIL import Image, ImageDraw
                actual_seed = -1 + idx  # Deterministic seed per pose
                img = Image.new("RGB", (actual_width, actual_height), color=(20, 30, 48))
                draw = ImageDraw.Draw(img)
                # Simple gradient background
                for y in range(actual_height):
                    t = y / max(1, actual_height)
                    r, g, b = int(35 * (1-t)), int(20 * (1-t)), int(60 * t)
                    draw.line([(0, y), (actual_width, y)], fill=(r, g, b))
                # Overlay text
                draw.text((20, 20), f"NEURONGEN CONTROLNET DEMO #{idx}", fill=(180, 190, 255))
                out_file = output_path.replace(".png", f"_{idx}.png")
                b_gen._save_image_fast(img, Path(out_file))
                saved_files.append(str(out_file))
                print(f"      [OK] Saved: {out_file}")
            else:
                # WebUI/ComfyUI API generation with ControlNet
                client = StableDiffusionWebUI({"webui": {"url": args.webui_host}, "comfyui": {"url": args.comfy_host}})
                if not client.is_running(args.webui_host):
                    print(f"[ERROR] Automatic1111 is offline at {args.webui_host}")
                    return

                # Progress bar for API generation
                with tqdm(total=1, desc="Generating via API...", leave=False) as pbar:
                    imgs, info_dict, errs = client.generate_images(
                        positive_prompt=norm_prompt,
                        steps=actual_steps,
                        cfg_scale=actual_cfg,
                        width=actual_width,
                        height=actual_height,
                        batch_size=1,
                        seed=args.seed,
                        sampler_name=args.sampler,
                        controlnet_poses=[(pose_path, pose_img)]  # Single pose for this iteration
                    )
                    pbar.update(1)
                
                if errs:
                    print(f"[ERROR] Generation failed: {', '.join(errs)}")
                    continue

                for img in imgs:
                    out_file = output_path.replace(".png", f"_{idx}.png")
                    b_gen._save_image_fast(img, Path(out_file))
                    saved_files.append(str(out_file))
                    print(f"      [OK] Saved: {out_file}")

        if saved_files:
            # Save metadata for ControlNet generation (separate from main block)
            meta_data = {
                "engine": args.engine,
                "timestamp": datetime.now().isoformat(),
                "prompt": norm_prompt,
                "controlnet": {
                    "mode": "single" if len(controlnet_poses) == 1 else "batch",
                    "poses": [os.path.basename(p[0]) for p in controlnet_poses]
                },
                "parameters": {
                    "steps": actual_steps,
                    "cfg": actual_cfg,
                    "width": actual_width,
                    "height": actual_height,
                    "seed": args.seed
                },
                "output_files": saved_files
            }
            meta_file = output_path.replace(".png", "_metadata.json")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
            print(f"\n[DONE] Generated {len(saved_files)} image(s). Metadata: {meta_file}")
        return  # Early exit after ControlNet generation

    # Inline demo implementation to avoid gradio import issues
    if args.engine == "demo":
        from PIL import Image, ImageDraw
        rng = random.Random(int(time.time() * 1000) % (2**31))
        saved = []
        for i in range(1, args.batch_size + 1):
            # Create aesthetic demo artwork
            img = Image.new("RGB", (width, height), color=(14, 15, 24))
            draw = ImageDraw.Draw(img)
            # Gradient background
            c1 = (rng.randint(30, 70), rng.randint(15, 40), rng.randint(80, 140))
            c2 = (rng.randint(10, 25), rng.randint(10, 25), rng.randint(30, 60))
            for y in range(height):
                t = y / max(1, height)
                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            # Text overlay
            draw.text((20, 20), f"NEURONGEN DEMO #{i}", fill=(180, 190, 255))
            out_file = output_path.replace(".png", f"_{i}.png") if args.batch_size > 1 else output_path
            b_gen._save_image_fast(img, Path(out_file))
            saved.append(out_file)
            print(f"   [OK] Saved: {out_file}")

    elif args.engine == "diffusers":
        saved = generate_image_diffusers(
            prompt=norm_prompt,
            output_path=output_path,
            steps=steps,
            cfg_scale=cfg,
            width=width,
            height=height,
            seed=seed
        )

    else:
        # WebUI or ComfyUI
        client = StableDiffusionWebUI({"webui": {"url": args.webui_host}, "comfyui": {"url": args.comfy_host}})
        if not client.is_running(args.webui_host):
            print(f"[ERROR] Automatic1111 is offline at {args.webui_host}")
            print("   Start Automatic1111 with --api, or run with '--engine demo' for instant testing.")
            return

        # Pass ControlNet poses if available (for non-batch mode)
        cn_poses = controlnet_poses[0] if controlnet_poses and len(controlnet_poses) == 1 else None
        imgs, info, errs = client.generate_images(
            positive_prompt=norm_prompt,
            steps=steps,
            cfg_scale=cfg,
            width=width,
            height=height,
            batch_size=args.batch_size,
            seed=seed,
            sampler_name=args.sampler,
            controlnet_poses=cn_poses
        )
        if errs:
            print(f"[ERROR] Generation failed: {', '.join(errs)}")
            return

        saved = []
        for i, img in enumerate(imgs, 1):
            out_file = output_path.replace(".png", f"_{i}.png") if len(imgs) > 1 else output_path
            b_gen._save_image_fast(img, Path(out_file))
            saved.append(out_file)
            print(f"   [OK] Saved: {out_file}")

    # Metadata persistence
    if saved:
        meta_data = {
            "engine": args.engine,
            "timestamp": datetime.now().isoformat(),
            "prompt": norm_prompt,
            "parameters": {
                "steps": steps,
                "cfg": cfg,
                "width": width,
                "height": height,
                "seed": seed,
                "batch_size": args.batch_size
            },
            "output_files": saved
        }
        # Add ControlNet info if used
        if controlnet_poses:
            meta_data["controlnet"] = {
                "mode": "single" if len(controlnet_poses) == 1 else "batch",
                "poses": [os.path.basename(p[0]) for p in controlnet_poses]
            }
        
        meta_file = output_path.replace(".png", "_metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        print(f"\n[DONE] Generated {len(saved)} image(s). Metadata: {meta_file}")


if __name__ == "__main__":
    main()