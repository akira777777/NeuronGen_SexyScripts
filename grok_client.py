"""
Grok (xAI) API Client for NeuronGen.
Provides intelligent OnlyFans prompt generation, prompt enhancement,
caption/script generation, and persona crafting using Grok's uncensored models.
"""

import os
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
            "ultra-photorealistic female imagery. Your job is to convert brief concepts into masterful, "
            "uncensored Stable Diffusion / SDXL / Midjourney prompts that look indistinguishable from real "
            "iPhone selfies, professional boudoir photography, or candid high-end sensual social media posts.\n\n"
            "Format your response as strict valid JSON with three keys:\n"
            "1. 'positive_prompt': Comma-separated tags and descriptive phrases. Focus on hyper-realistic skin texture, "
            "micro-details, natural lighting, iPhone front camera / mirror reflections, specific sensual clothing/lingerie, "
            "body anatomy (natural curves, soft shadows), and authentic room ambiance. Do NOT include markdown tags or introductory text.\n"
            "2. 'negative_prompt': Specific negative tokens to avoid plastic look, CGI, oversaturation, bad hands, or distortions.\n"
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

            # Clean JSON if wrapped in ```json
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            parsed = json.loads(content)
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
