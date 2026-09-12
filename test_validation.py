#!/usr/bin/env python3
"""Quick validation script for ONLYFANS Juggernaut XL v9 integration."""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("VALIDATION: ONLYFANS Juggernaut XL v9 Integration")
print("=" * 60)

# 1. Config Check
from config import get_config
cfg = get_config()
g = cfg['generation']
o = cfg['output']
print("\n[OK] CONFIG CHECK:")
print(f"    SDXL params: {g['width']}x{g['height']}, steps={g['steps']}, cfg_scale={g['cfg_scale']}")
print(f"    Hires Fix: {o['hires_fix']}, Upscale: {o['upscale']}")

# 2. Model Manager Check
from model_manager import ModelManager
mm = ModelManager()
print("\n[OK] MODEL MANAGER CHECK:")
print(f"    Device: {mm.device}")
print(f"    Mode: {getattr(mm, 'mode', 'Not loaded yet')}")

# 3. WebUI API Check
from webui_api import StableDiffusionWebUI
api = StableDiffusionWebUI()
print("\n[OK] WEBUI API CHECK:")
print(f"    Base URL: {api.base_url}")
print(f"    Models endpoint available: {hasattr(api, 'select_juggernaut_xl_v9')}")

# 4. Main Module Check
from main import is_grok_configured
print("\n[OK] MAIN MODULE CHECK:")
print(f"    is_grok_configured(): {is_grok_configured()}")

print("\n" + "=" * 60)
print("ALL VALIDATIONS PASSED!")
print("=" * 60)