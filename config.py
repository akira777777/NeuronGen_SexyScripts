"""
Configuration management for NeuronGen.
Handles loading/saving settings to JSON and providing defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILE = CONFIG_DIR / "generation_config.json"


def ensure_config_dir():
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load configuration from JSON file. Returns defaults if file not found or invalid."""
    ensure_config_dir()
    
    default_config = {
        "webui": {
            "url": "http://127.0.0.1:7860",  # Default A1111 WebUI address
            "api_key": None,  # Optional API key for secured instances
            "timeout": 300,    # Request timeout in seconds
            "enabled": True,
        },
        "generation": {
            "steps": 25,       # Default generation steps
            "cfg_scale": 7.5,  # CFG scale (guidance)
            "width": 896,      # Output width
            "height": 1152,    # Output height
            "batch_size": 1,   # Number of images per prompt request
            "seed": -1,        # -1 for random seed
            "sampler": "DPM++ 2M Karras",
            "sampler_name": "DPM++ 2M Karras",
            "model": None,     # Specific checkpoint to use (None = WebUI default)
        },
        "prompts": {
            "positive_separator": ",",      # Token separator for positive prompt
            "negative_separator": "|",      # Token separator for negative prompt  
            "supports_weights": True,       # Enable weight syntax like 1.5:keyword
            "default_negative_prompt": "(plastic skin, airbrushed, wax figure, CGI, 3D render, cartoon, anime, illustration:1.4), (extra fingers, deformed hands, fused fingers, mutated limbs, cross-eyed, malformed eyes:1.4), (worst quality, low quality, normal quality:1.3), (monochrome, grayscale, bad anatomy, bad proportions, unnatural body, distorted features:1.2), (overly symmetrical face, doll-like features, artificial lighting, studio perfection:1.1)",
        },
        "output": {
            "save_dir": Path(__file__).parent / "results",
            "directory": "results",
            "filename_format": "{seed}_{timestamp}.png",
            "overwrite": False,
            "upscale": False  # Enable ESRGAN upscale pipeline
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
    
    return default_config


def save_config(config: dict) -> bool:
    """Save configuration to JSON file."""
    ensure_config_dir()
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"❌ Error saving config to {CONFIG_FILE}: {e}")
        return False


def get_config() -> dict:
    """Get current configuration (loads if needed)."""
    return load_config()
