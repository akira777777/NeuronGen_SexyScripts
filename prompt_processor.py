"""
High-Performance Prompt Processing Module for NeuronGen.
Features:
- Pre-compiled regex tokenizers
- LRU cached token parsing and normalization
- Zero-copy parameter extraction (:steps:, :cfg:, :width:, :height:, :seed:)
- Full support for A1111/ComfyUI weight syntax
"""

import re
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional

# Pre-compiled regular expressions for maximum throughput
RE_STEPS = re.compile(r':steps:\s*(\d+)', re.IGNORECASE)
RE_CFG = re.compile(r':cfg:\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
RE_WIDTH = re.compile(r':width:\s*(\d+)', re.IGNORECASE)
RE_HEIGHT = re.compile(r':height:\s*(\d+)', re.IGNORECASE)
RE_SEED = re.compile(r':seed:\s*(\d+)', re.IGNORECASE)
RE_CLEAN_SEPARATORS = re.compile(r'[,|]\s*$')
RE_MULTI_SPACES = re.compile(r'\s{2,}')

# Token weight matching patterns
RE_PAREN_WEIGHT = re.compile(r'^\((.+?):(\d+(?:\.\d+)?)\)$')
RE_PREFIX_WEIGHT = re.compile(r'^(\d+(?:\.\d+)?):(.+)$')
RE_PAREN = re.compile(r'^\((.+?)\)$')
RE_BRACKET = re.compile(r'^\[(.+?)\]$')


@lru_cache(maxsize=4096)
def _fast_parse_token(text: str) -> Tuple[str, str, float, str]:
    """Cached parsing of single token. Returns (token_str, keyword, weight, format_type)."""
    text = text.strip()
    
    # (keyword:1.2)
    m = RE_PAREN_WEIGHT.match(text)
    if m:
        kw = m.group(1).strip()
        w = float(m.group(2))
        return text, kw, w, "parentheses_weight"

    # 1.2:keyword
    m = RE_PREFIX_WEIGHT.match(text)
    if m:
        w = float(m.group(1))
        kw = m.group(2).strip()
        return f"({kw}:{w})", kw, w, "prefix_weight"

    # (keyword) -> 1.1
    m = RE_PAREN.match(text)
    if m:
        kw = m.group(1).strip()
        return text, kw, 1.1, "parentheses"

    # [keyword] -> 0.9
    m = RE_BRACKET.match(text)
    if m:
        kw = m.group(1).strip()
        return text, kw, 0.9, "bracket"

    return text, text, 1.0, "standard"


class PromptProcessor:
    """Processes prompts for WebUI, ComfyUI, and local pipelines."""

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get("prompts", {}) if isinstance(config, dict) else {}
        self.config = cfg
        self.positive_separator = cfg.get("positive_separator", ",")
        self.negative_separator = cfg.get("negative_separator", "|")
        self.supports_weights = cfg.get("supports_weights", True)
        self.default_negative = cfg.get(
            "default_negative_prompt",
            "(worst quality, low quality:1.3), (normalized:0.689)"
        )

    def extract_parameters(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Fast extraction of inline parameters like :steps:30, :cfg:7.5, :width:896.
        Returns:
            Tuple of (clean_prompt, parameters_dict)
        """
        params: Dict[str, Any] = {}
        clean_text = prompt

        m = RE_STEPS.search(clean_text)
        if m:
            params["steps"] = int(m.group(1))
            clean_text = clean_text[:m.start()] + clean_text[m.end():]

        m = RE_CFG.search(clean_text)
        if m:
            params["cfg_scale"] = float(m.group(1))
            clean_text = clean_text[:m.start()] + clean_text[m.end():]

        m = RE_WIDTH.search(clean_text)
        if m:
            params["width"] = int(m.group(1))
            clean_text = clean_text[:m.start()] + clean_text[m.end():]

        m = RE_HEIGHT.search(clean_text)
        if m:
            params["height"] = int(m.group(1))
            clean_text = clean_text[:m.start()] + clean_text[m.end():]

        m = RE_SEED.search(clean_text)
        if m:
            params["seed"] = int(m.group(1))
            clean_text = clean_text[:m.start()] + clean_text[m.end():]

        clean_text = RE_CLEAN_SEPARATORS.sub('', clean_text.strip())
        clean_text = RE_MULTI_SPACES.sub(' ', clean_text).strip()

        return clean_text, params

    def process_prompt(self, prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a prompt string, normalizing weights and extracting tokens.
        """
        if not prompt or not isinstance(prompt, str):
            return "", []

        clean_prompt, _ = self.extract_parameters(prompt)

        raw_parts = [p.strip() for p in clean_prompt.split(self.positive_separator) if p.strip()]
        
        tokens_with_weights = []
        normalized_parts = []

        for part in raw_parts:
            tok_str, kw, w, fmt = _fast_parse_token(part)
            tokens_with_weights.append({
                "token": tok_str,
                "keyword": kw,
                "weight": w,
                "format": fmt
            })
            normalized_parts.append(tok_str)

        normalized = ", ".join(normalized_parts)
        return normalized, tokens_with_weights

    def parse_discord_or_midjourney_prompt(self, text: str) -> Dict[str, Any]:
        """
        Parses prompts copied from Discord (e.g. Midjourney / Bot generations).
        Handles:
        - '/imagine prompt: ...'
        - Aspect ratios '--ar 9:16', '--ar 2:3', '--ar 3:4', '--ar 1:1' -> mapped to SD resolutions
        - Midjourney flags: --v, --s, --stylize, --q, --c, --chaos, --no (negative prompt)
        """
        result = {
            "positive_prompt": text,
            "negative_prompt": self.default_negative,
            "width": 512,
            "height": 768,
            "cfg_scale": 7.0,
            "steps": 25,
            "flags_detected": []
        }

        clean = text.strip()
        # Remove Discord prefix
        if clean.lower().startswith("/imagine prompt:"):
            clean = clean[len("/imagine prompt:"):].strip()
        elif clean.lower().startswith("/imagine"):
            clean = clean[len("/imagine"):].strip()

        # Extract --no (negative prompt)
        no_match = re.search(r'--no\s+([^--]+)', clean, re.IGNORECASE)
        if no_match:
            neg_content = no_match.group(1).strip()
            result["negative_prompt"] = f"{self.default_negative}, {neg_content}"
            clean = clean[:no_match.start()] + clean[no_match.end():]
            result["flags_detected"].append(f"--no {neg_content}")

        # Extract aspect ratio --ar
        ar_match = re.search(r'--ar\s+([0-9]+:[0-9]+)', clean, re.IGNORECASE)
        if ar_match:
            ar = ar_match.group(1).strip()
            result["flags_detected"].append(f"--ar {ar}")
            if ar in ["9:16", "2:3", "3:4", "4:5"]:
                result["width"] = 512
                result["height"] = 768
            elif ar in ["16:9", "3:2", "4:3"]:
                result["width"] = 768
                result["height"] = 512
            elif ar in ["1:1"]:
                result["width"] = 512
                result["height"] = 512
            clean = clean[:ar_match.start()] + clean[ar_match.end():]

        # Strip remaining Midjourney flags (--v 6, --s 250, --c 10, etc.)
        flag_matches = re.findall(r'--[a-zA-Z0-9_\.-]+(?:\s+[a-zA-Z0-9_\.-]+)?', clean)
        for fm in flag_matches:
            result["flags_detected"].append(fm)
            clean = clean.replace(fm, "")

        # Clean excess spaces and trailing commas
        clean = RE_MULTI_SPACES.sub(" ", clean).strip()
        clean = re.sub(r'[,;\s]+$', '', clean).strip()

        result["positive_prompt"] = clean
        return result

    def get_separator(self, prompt_type: str) -> str:
        if prompt_type.lower() == "negative":
            return self.negative_separator
        return self.positive_separator
