#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script: Photorealistic OnlyFans Style Generation with Juggernaut XL v9
===========================================================================
Tests the pipeline end-to-end using curated OnlyFans presets.

Usage:
    python tests/test_photorealistic.py --engine demo
    python tests/test_photorealistic.py --engine webui_api
"""

import argparse
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_processor import PromptProcessor
from demo_renderer import create_demo_artwork


def create_test_prompt(archetype="slavic_blonde", scenario="bedroom_morning") -> tuple[str, str]:
    """Generates a photorealistic OnlyFans-style prompt using presets."""
    
    # Character archetypes (tuned for realism and sex appeal)
    ARCHETYPES = {
        "slavic_blonde": {
            "character": "21yo slavic woman, athletic toned body with visible abs, natural blonde wavy hair cascading over shoulders, piercing blue eyes, cute soft seductive smile, perfect white teeth, symmetrical face, soft cheekbones, natural lip shape",
        },
        "latina_brunette": {
            "character": "22yo latina woman, tan smooth skin with sun-kissed glow, voluptuous hourglass figure with wide hips and toned waist, dark brown voluminous hair cascading down back, hazel eyes, alluring confident look, plump natural lips",
        },
        "nordic_redhead": {
            "character": "20yo caucasian woman, natural ginger red hair in loose waves framing face, light green eyes with dramatic catchlights, pale porcelain skin with subtle freckles across nose and shoulders, slim fit physique, defined collarbones",
        },
    }

    # Scenarios (photographic settings)
    SCENARIOS = {
        "bedroom_morning": {
            "setting": "luxury modern bedroom with messy silk sheets in background, soft golden hour morning sunlight streaming through window creating dramatic rim lighting on skin and hair",
            "outfit": "wearing an unbuttoned oversized silk white shirt revealing delicate black lace bralette underneath, top button undone exposing cleavage",
            "camera": "iPhone selfie perspective held at arm's length, natural slight motion blur for candid feel, authentic casual snap aesthetic",
        },
        "mirror_selfie": {
            "setting": "clean modern bathroom with marble tiles and gold fixtures, ambient ring light creating flattering highlights on skin and hair, clear glass reflection showing depth",
            "outfit": "wearing matching ribbed seamless sports crop top revealing toned midriff and navel, cheeky micro shorts sitting low on hips emphasizing hip curve",
            "camera": "holding smartphone taking mirror selfie from slight angle to emphasize curves, realistic phone case visible in reflection",
        },
        "bikini_poolside": {
            "setting": "infinity pool overlooking tropical ocean with crystal clear water, vibrant sunny day, palm tree shadows creating dappled lighting on skin",
            "outfit": "wearing skimpy metallic micro bikini that barely covers essentials, droplets of seawater glistening on sun-kissed tan skin",
            "camera": "professional candid photography aesthetic, 85mm portrait lens f/1.2, natural highlights and shadows emphasizing body contours",
        },
    }

    # Photographic realism boosters (critical for OnlyFans style)
    REALISM_BOOSTERS = (
        "(raw photo, candid amateur photography, 8k uhd, dslr, high resolution:1.2), "
        "(natural detailed skin texture with visible pores and subtle imperfections, goosebumps on arms:1.1), "
        "(soft natural lighting creating dramatic shadows across curves, photorealistic catchlights in eyes:1.05)"
    )

    # Negative prompt (eliminates AI plastic look)
    NEGATIVE_PROMPT = (
        "(plastic skin, airbrushed, wax figure, CGI, 3D render, cartoon, anime, illustration:1.4), "
        "(extra fingers, deformed hands, fused fingers, mutated limbs, cross-eyed, malformed eyes:1.4), "
        "(worst quality, low quality, normal quality:1.3), "
        "(monochrome, grayscale, bad anatomy, bad proportions, unnatural body, distorted features:1.2), "
        "(overly symmetrical face, doll-like features, artificial lighting, studio perfection:1.1)"
    )

    arch = ARCHETYPES.get(archetype, ARCHETYPES["slavic_blonde"])
    scen = SCENARIOS.get(scenario, SCENARIOS["bedroom_morning"])

    positive_prompt = (
        f"{REALISM_BOOSTERS}, "
        f"{arch['character']}, "
        f"{scen['outfit']}, "
        f"{scen['setting']}, "
        f"{scen['camera']}, "
        "(masterpiece, photorealistic:1.2)"
    )

    return positive_prompt.strip(), NEGATIVE_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Photorealistic OnlyFans Style Test Generator")
    parser.add_argument("--engine", choices=["demo", "webui", "webui_api", "diffusers", "local"], default="demo", help="Engine to use")
    parser.add_argument("--archetype", default="slavic_blonde", help="Character archetype")
    parser.add_argument("--scenario", default="bedroom_morning", help="Scenario/setting")
    parser.add_argument("--hires_fix", action="store_true", help="Enable High-Resolution Fix for pore-level skin texture")
    args = parser.parse_args()

    if args.engine == "webui":
        args.engine = "webui_api"
    elif args.engine == "local":
        args.engine = "diffusers"

    # Generate prompt
    positive_prompt, negative_prompt = create_test_prompt(args.archetype, args.scenario)

    print("\n" + "=" * 70)
    print("NEURONGEN PHOTOREALISTIC TEST GENERATOR")
    print("=" * 70)
    print(f"Archetype: {args.archetype}")
    print(f"Scenario: {args.scenario}")
    print("-" * 70)

    # Process prompt
    proc = PromptProcessor()
    norm_prompt, _ = proc.process_prompt(positive_prompt)
    _, inline_params = proc.extract_parameters(positive_prompt)

    steps = inline_params.get("steps", 30)
    cfg_scale = inline_params.get("cfg_scale", 6.5)
    width = inline_params.get("width", 512)
    height = inline_params.get("height", 768)

    print(f"\nPrompt (first 100 chars): {norm_prompt[:100]}...")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {steps} | CFG Scale: {cfg_scale}")
    print("-" * 70)

    # Save prompt to file for reference
    output_dir = Path("results/test_photorealistic")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / f"{args.archetype}_{args.scenario}_prompt.txt"
    
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(f"# {args.archetype} - {args.scenario}\n\n")
        f.write(f"Positive:\n{positive_prompt}\n\n")
        f.write(f"Negative:\n{negative_prompt}")

    print(f"\nPrompt saved to: {prompt_file}")

    if args.engine == "demo":
        from PIL import Image, ImageDraw
        
        print("\n[INFO] Rendering demo preview...")
        
        img = Image.new("RGB", (width, height), color=(14, 15, 24))
        draw = ImageDraw.Draw(img)

        # Gradient background
        c1 = (30, 60, 90)
        c2 = (20, 40, 70)
        for y in range(height):
            t = y / max(1, height)
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Text overlay
        text = f"NEURONGEN PHOTOREALISTIC TEST\n{args.archetype} | {args.scenario}"
        font_size = max(16, height // 40)
        draw.text((20, 20), text, fill=(180, 190, 255))

        output_path = output_dir / f"{args.archetype}_{args.scenario}_demo.png"
        img.save(output_path, format="PNG", compress_level=1)
        
        print(f"\n[OK] Demo preview saved: {output_path}")
        print("\nTo generate real images on RTX 4070 Ti:")
        print(f'  python generate.py --prompt "{prompt_file}" --engine diffusers')

    elif args.engine == "diffusers":
        from generate import generate_image_diffusers
        print("\n[INFO] Generating with local PyTorch Diffusers (RTX 4070 Ti)...")
        output_path = output_dir / f"{args.archetype}_{args.scenario}_diffusers.png"
        generated = generate_image_diffusers(
            prompt=norm_prompt,
            output_path=str(output_path),
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            seed=42,
            negative_prompt=negative_prompt,
            hires_fix=args.hires_fix
        )
        if generated:
            print(f"[OK] Saved: {generated[0]}")
            meta_data = {
                "engine": "diffusers",
                "archetype": args.archetype,
                "scenario": args.scenario,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "prompt": norm_prompt,
                "negative_prompt": negative_prompt,
                "parameters": {
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "width": width,
                    "height": height,
                    "seed": 42
                },
                "output_files": [str(output_path)]
            }
            with open(output_dir / f"{args.archetype}_{args.scenario}_metadata.json", "w", encoding="utf-8") as f:
                __import__("json").dump(meta_data, f, indent=2)

    elif args.engine == "webui_api":
        from config import get_config
        from webui_api import StableDiffusionWebUI
        
        cfg = get_config()
        client = StableDiffusionWebUI(cfg)
        
        if not client.is_running(cfg["webui"]["url"]):
            print(f"[ERROR] WebUI is offline at {cfg['webui']['url']}")
            return

        print("\n[INFO] Generating with WebUI API...")
        
        imgs, info_dict, errs = client.generate_images(
            positive_prompt=norm_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            batch_size=1,
            seed=-1,
        )

        if errs:
            print(f"[ERROR] Generation failed: {', '.join(errs)}")
            return

        for i, img in enumerate(imgs, 1):
            output_path = output_dir / f"{args.archetype}_{args.scenario}_v{i}.png"
            img.save(output_path, format="PNG", compress_level=1)
            print(f"[OK] Saved: {output_path}")

        # Save metadata
        meta_data = {
            "engine": "webui_api",
            "archetype": args.archetype,
            "scenario": args.scenario,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "prompt": norm_prompt[:200],
            "parameters": {
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "seed": -1
            },
            "output_files": [str(output_path)]
        }

        with open(output_dir / f"{args.archetype}_{args.scenario}_metadata.json", "w", encoding="utf-8") as f:
            __import__("json").dump(meta_data, f, indent=2)


class TestPhotorealistic(unittest.TestCase):
    """Unit tests for photorealistic prompt presets and demo rendering."""

    def test_01_prompt_creation_defaults(self):
        pos, neg = create_test_prompt()
        self.assertIn("slavic woman", pos)
        self.assertIn("bedroom", pos)
        self.assertIn("plastic skin", neg)

    def test_02_all_archetypes_and_scenarios(self):
        archetypes = ["slavic_blonde", "latina_brunette", "nordic_redhead"]
        scenarios = ["bedroom_morning", "mirror_selfie", "bikini_poolside"]
        for arch in archetypes:
            for scen in scenarios:
                pos, neg = create_test_prompt(arch, scen)
                self.assertGreater(len(pos), 50)
                self.assertGreater(len(neg), 30)

    def test_03_prompt_processor_parsing(self):
        proc = PromptProcessor()
        pos, _ = create_test_prompt("latina_brunette", "mirror_selfie")
        norm, tokens = proc.process_prompt(pos)
        self.assertTrue(len(tokens) > 0)
        self.assertIn("latina woman", norm)

    def test_04_demo_artwork_render(self):
        img = create_demo_artwork(
            prompt="21yo slavic woman, bedroom selfie",
            width=512,
            height=768,
            steps=25,
            cfg=7.5,
            seed=42,
            index=1
        )
        self.assertEqual(img.size, (512, 768))


if __name__ == "__main__":
    if len(sys.argv) > 1 and any(arg.startswith("--") for arg in sys.argv[1:]):
        main()
    else:
        unittest.main()