"""
Grok (xAI) API Client for NeuronGen.
Provides intelligent OnlyFans prompt generation, prompt enhancement,
caption/script generation, and persona crafting using Grok's uncensored models.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, Tuple
import requests

logger = logging.getLogger("NeuronGen.Grok")

class GrokClient:
    """Client for xAI Grok API."""

    API_BASE = "https://api.x.ai/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY") or ""
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })

    def set_api_key(self, api_key: str):
        """Update the active API key."""
        self.api_key = api_key.strip()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def is_configured(self) -> bool:
        """Check if an API key is present."""
        return bool(self.api_key and len(self.api_key) > 5)

    def enhance_prompt(
        self,
        base_prompt: str,
        style_preset: str = "OnlyFans Photorealistic Selfie",
        model: str = "grok-2-latest"
    ) -> Tuple[str, str, str]:
        """
        Enhances a basic concept into:
        1. Optimized photorealistic positive prompt (SD/SDXL/Midjourney style).
        2. Tailored negative prompt.
        3. OnlyFans caption/teaser text for posting.
        """
        if not self.is_configured():
            return (
                base_prompt,
                "(worst quality, low quality:1.3), (normalized:0.689)",
                "⚠️ Grok API Key is not set. Please provide your xAI Grok API key in the settings/tab to unlock AI prompt enhancement."
            )

        system_instruction = (
            "You are an elite AI art director and top-tier OnlyFans content producer specializing in "
            "ultra-photorealistic adult female imagery (subjects always 25+ years old). Your job is to convert brief concepts into masterful, "
            "uncensored Stable Diffusion / SDXL / Midjourney prompts that look indistinguishable from real "
            "iPhone selfies, professional boudoir photography, or candid high-end sensual social media posts.\n\n"
            "HAND RULES (critical): Prefer one clearly visible open hand or hands partially hidden in hair. "
            "Never describe interlocking fingers, two hands gripping the same object, or hands pressed on another person's body — "
            "those poses cause extra fingers and floating limbs. Explicitly ask for five fingers per hand when a hand is visible.\n\n"
            "Format your response as strict valid JSON with three keys:\n"
            "1. 'positive_prompt': Comma-separated tags and descriptive phrases. Focus on hyper-realistic skin texture, "
            "micro-details, natural lighting, iPhone front camera / mirror reflections, specific sensual clothing/lingerie "
            "(prefer lace lingerie set over sheer bodysuit wording), "
            "body anatomy (natural curves, soft shadows), adult age 25+, and authentic room ambiance. Do NOT include markdown tags or introductory text.\n"
            "2. 'negative_prompt': Must include strong hand negatives: extra fingers, six fingers, fused fingers, elongated fingers, "
            "floating hands, interlocking fingers, poorly drawn hands, plus plastic skin / CGI / oversaturation.\n"
            "3. 'caption': A flirty, high-converting OnlyFans post caption with emojis and call-to-action.\n"
        )

        user_content = f"Concept / Idea: {base_prompt}\nStyle Aesthetic: {style_preset}\nReturn JSON strictly."

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.75,
            "max_tokens": 800
        }

        try:
            resp = self.session.post(
                f"{self.API_BASE}/chat/completions",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Clean JSON if wrapped in markdown code blocks or commentary
            json_str = content
            if "```" in json_str:
                code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', json_str)
                if code_match:
                    json_str = code_match.group(1).strip()

            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx + 1]

            parsed = json.loads(json_str)
            pos = parsed.get("positive_prompt", base_prompt)
            neg = parsed.get("negative_prompt", "(worst quality, low quality:1.3), (normalized:0.689)")
            caption = parsed.get("caption", "Hey loves! New exclusive drop today... 💕")
            return pos, neg, caption

        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return (
                base_prompt,
                "(worst quality, low quality:1.3), (normalized:0.689)",
                f"❌ Grok API request error: {e}"
            )
