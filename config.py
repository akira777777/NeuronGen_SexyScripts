"""
Configuration management for NeuronGen.
Handles loading/saving settings to JSON and providing defaults.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Optional


CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILE = CONFIG_DIR / "generation_config.json"


_CACHED_CONFIG: Optional[dict] = None


def ensure_config_dir():
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config(force_reload: bool = False) -> dict:
    """Load configuration from JSON file with in-memory caching."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not force_reload:
        return copy.deepcopy(_CACHED_CONFIG)

    ensure_config_dir()
    
    default_config = {
        "webui": {
            "url": "http://127.0.0.1:7860",  # Default A1111 WebUI address
            "api_key": None,  # Optional API key for secured instances
            "timeout": 300,    # Request timeout in seconds
            "enabled": True,
        },
        "generation": {
            "steps": 40,       # Juggernaut XL v9: more steps for ultra skin detail
            "cfg_scale": 7.5,  # CFG scale optimized for photorealism (Juggernaut XL standard)
            "width": 1024,     # SDXL native resolution
            "height": 1536,    # SDXL vertical format (ONLYFANS portrait optimization)
            "batch_size": 5,   # Batch generation for efficiency
            "seed": -1,        # -1 for random seed
            "sampler": "DPM++ SDE Karras",  # Best sampler for photorealism with XL
            "sampler_name": "DPM++ SDE Karras",
            "model": None,     # Auto-detect Juggernaut XL v9 from checkpoint dir
        },
        "prompts": {
            "positive_separator": ",",      # Token separator for positive prompt
            "negative_separator": "|",      # Token separator for negative prompt  
            "supports_weights": True,       # Enable weight syntax like 1.5:keyword
            "default_negative_prompt": "(plastic skin, airbrushed, wax figure, CGI, 3D render, cartoon, anime, illustration:1.4), (extra fingers, six fingers, fused fingers, mutated fingers, elongated fingers, deformed hands, poorly drawn hands, floating hands, interlocking fingers, overlapping hands gripping object, missing fingers, mutated limbs, cross-eyed, malformed eyes:1.45), (worst quality, low quality, normal quality:1.3), (monochrome, grayscale, bad anatomy, bad proportions, unnatural body, distorted features:1.2), (overly symmetrical face, doll-like features, artificial lighting, studio perfection:1.1)",
        },
        "output": {
            "save_dir": str(Path(__file__).parent / "results"),
            "directory": "results",
            "filename_format": "{width}_{height}_{seed}_{timestamp}.png",  # Include resolution for tracking
            "overwrite": False,
            "upscale": True,      # Enable ESRGAN upscale pipeline by default (ONLYFANS quality)
            "hires_fix": True,    # Enable 2-pass High-Res Fix for Juggernaut XL v9
            "onlyfans_optimized": True  # Flag for OF-specific presets
        }
    }
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                
                # If generation_config.json has webui_api, sync with webui
                if "webui_api" in saved_config:
                    webui_data = saved_config["webui_api"]
                    if "host" in webui_data:
                        default_config["webui"]["url"] = webui_data["host"]
                    if "timeout" in webui_data:
                        default_config["webui"]["timeout"] = webui_data["timeout"]
                    if "enabled" in webui_data:
                        default_config["webui"]["enabled"] = webui_data["enabled"]
                        
                for key, val in saved_config.items():
                    if isinstance(val, dict):
                        if key not in default_config:
                            default_config[key] = {}
                        default_config[key].update(val)
                    else:
                        default_config[key] = val
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load config from {CONFIG_FILE}. Using defaults.")
    
    # Ensure webui_api and webui are synchronized
    default_config["webui_api"] = {
        "host": default_config["webui"].get("url", "http://127.0.0.1:7860"),
        "timeout": default_config["webui"].get("timeout", 300),
        "enabled": default_config["webui"].get("enabled", True),
    }
    
    _CACHED_CONFIG = default_config
    return copy.deepcopy(_CACHED_CONFIG)


def save_config(config: dict) -> bool:
    """Save configuration to JSON file and update cache."""
    global _CACHED_CONFIG
    ensure_config_dir()
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        _CACHED_CONFIG = copy.deepcopy(config)
        return True
    except (IOError, TypeError, OSError) as e:
        print(f"❌ Error saving config to {CONFIG_FILE}: {e}")
        return False



def get_profile(name: str = "realism") -> dict:
    """Return a named generation profile from config (speed/realism)."""
    profiles = get_config().get("profiles") or {}
    raw = (name or "realism").strip().lower()
    # Gradio labels may include emoji: "⚡ Speed", "🌟 Realism / Quality"
    if "speed" in raw or "fast" in raw or "скорост" in raw:
        key = "speed"
    elif "realism" in raw or "quality" in raw or "качеств" in raw or "реализм" in raw:
        key = "realism"
    else:
        key = raw
    return dict(profiles.get(key) or profiles.get("realism") or {})


def apply_profile_prompt(prompt: str, profile: Optional[dict] = None) -> str:
    """Drop ban_positive_tokens from a prompt for the given profile."""
    if not prompt:
        return ""
    prof = profile if profile is not None else get_profile("realism")
    banned = [t.casefold() for t in (prof.get("ban_positive_tokens") or [])]
    if not banned:
        return prompt
    try:
        from prompts.onlyfans_presets import strip_beauty_tokens
        prompt = strip_beauty_tokens(prompt)
    except Exception:
        pass
    parts = [p.strip() for p in prompt.split(",")]
    kept = []
    for part in parts:
        if not part:
            continue
        low = part.casefold()
        if any(b in low for b in banned):
            continue
        kept.append(part)
    return ", ".join(kept)


def get_config(force_reload: bool = False) -> dict:
    """Get current configuration (returns cached copy or loads if needed)."""
    return load_config(force_reload=force_reload)

