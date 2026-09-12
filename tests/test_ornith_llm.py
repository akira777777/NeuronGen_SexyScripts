#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script: Ornith-1.5 LLM Integration for Local Prompt Enhancement
=====================================================================
Tests the local LLM fallback when Grok API key is unavailable.

Usage:
    python tests/test_ornith_llm.py --engine demo
    python tests/test_ornith_llm.py --test-only-grok  # Tests Grok client without generation
"""

import argparse
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from grok_client import GrokClient


def test_grok_api():
    """Test Grok API integration (requires valid API key)."""
    print("=" * 70)
    print("TEST: Grok API Integration")
    print("=" * 70)

    client = GrokClient(api_key="test_invalid_key_for_testing_only")
    
    if not client.is_configured():
        print("[SKIP] No API key configured.")
        return
    
    try:
        pos, neg, caption = client.enhance_prompt(
            "beautiful girl selfie in silk shirt",
            style_preset="OnlyFans Photorealistic Selfie"
        )
        
        print(f"[OK] Grok enhancement complete!")
        print(f"   Positive prompt length: {len(pos)} chars")
        print(f"   Negative prompt length: {len(neg)} chars")
        print(f"   Caption preview: {caption[:80]}...")
        
    except Exception as e:
        print(f"[ERROR] Grok API test failed: {e}")


def test_ornith_local_llm():
    """Test local Ornith-1.5 LLM fallback."""
    print("\n" + "=" * 70)
    print("TEST: Local Ornith-1.5 LLM Integration")
    print("=" * 70)

    # Create client without API key to force local LLM fallback
    client = GrokClient(api_key=None)
    
    if client.is_configured():
        print("[SKIP] API key is configured, skipping local LLM test.")
        return
    
    try:
        pos, neg, caption = client.enhance_prompt(
            "beautiful girl selfie in silk shirt",
            style_preset="OnlyFans Photorealistic Selfie"
        )
        
        print(f"[OK] Local Ornith enhancement complete!")
        print(f"   Positive prompt length: {len(pos)} chars")
        print(f"   Negative prompt length: {len(neg)} chars")
        print(f"   Caption preview: {caption[:80]}...")
        
    except FileNotFoundError as e:
        print(f"[INFO] Ornith model not found (expected if not downloaded): {e}")
        print("[INFO] Run: python generate.py download-ornith")
    except Exception as e:
        print(f"[ERROR] Local LLM test failed: {type(e).__name__}: {e}")


def test_demo_generation():
    """Test demo generation with Grok-enhanced prompts."""
    from demo_renderer import create_demo_artwork
    
    print("\n" + "=" * 70)
    print("TEST: Demo Generation with AI Prompt Enhancement")
    print("=" * 70)

    # Test with Grok API (if configured) or local LLM fallback
    client = GrokClient()
    
    base_prompt = "beautiful girl selfie in silk shirt"
    
    if client.is_configured():
        print("[INFO] Using Grok API for prompt enhancement...")
        pos, neg, caption = client.enhance_prompt(
            base_prompt,
            style_preset="OnlyFans Photorealistic Selfie"
        )
    else:
        print("[INFO] Using local Ornith LLM fallback...")
        try:
            pos, neg, caption = client.enhance_prompt(
                base_prompt,
                style_preset="OnlyFans Photorealistic Selfie"
            )
        except Exception as e:
            print(f"[WARN] Enhancement failed ({e}), using default prompt.")
            pos = base_prompt
            neg = "(worst quality, low quality:1.3)"
            caption = "✨ New drop! 💕"

    # Generate demo image with enhanced prompt
    img = create_demo_artwork(
        prompt=pos,
        width=512,
        height=768,
        steps=10,
        cfg=6.5,
        seed=-1,
        index=1
    )

    # Save to test results folder
    output_dir = Path("results") / "test_ornith_llm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"demo_with_ai_prompt_{client.api_key or 'local'}_1.png"
    img.save(output_path, format="PNG", compress_level=1)

    print(f"[OK] Demo image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ornith LLM Integration Tests")
    parser.add_argument("--engine", choices=["demo"], default="demo", help="Generation engine mode")
    parser.add_argument("--test-only-grok", action="store_true", help="Test Grok client only, no generation")

    args = parser.parse_args()

    if args.test_only_grok:
        test_grok_api()
    else:
        test_grok_api()
        test_ornith_local_llm()
        
        if args.engine == "demo":
            test_demo_generation()


if __name__ == "__main__":
    main()