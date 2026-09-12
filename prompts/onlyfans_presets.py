"""
Curated prompt presets specifically designed for photorealistic OnlyFans model generation.
Includes detailed photography tags, skin textures, lighting, outfits, poses, and character archetypes.

Hand hygiene (from gen stress tests): interlocking hands, object grips, and hands-on-partner
poses fail often. Prefer open hands, hands-in-hair occlusion, or crop hands out of frame.
"""
from typing import Tuple, Dict, Any, Optional
import re

REALISM_BAN_TOKENS = (
    "perfect skin",
    "airbrushed",
    "flawless glass skin",
    "beauty retouch",
    "porcelain glass skin",
    "natural porcelain skin with visible pores",
)


def strip_beauty_tokens(prompt: str) -> str:
    """Remove polish tokens that push plastic/beauty look (Realism / Quality profile)."""
    if not prompt:
        return prompt
    out = prompt
    for tok in REALISM_BAN_TOKENS:
        out = re.sub(re.escape(tok), "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s*,\s*,+", ", ", out)
    return out.strip(" ,")



OF_CHARACTER_ARCHETYPES = {
    "slavic_blonde": {
        "name": "Slavic Blonde (Sporty / Glamour)",
        "character": "25yo slavic woman, athletic toned body with visible abs, natural blonde wavy hair cascading over shoulders, piercing blue eyes, cute soft seductive smile, natural lip shape, slight natural facial asymmetry, soft cheekbones",
    },
    "latina_brunette": {
        "name": "Latina Brunette (Curvy / Exotic)",
        "character": "26yo latina woman, tan smooth skin with sun-kissed glow, voluptuous hourglass figure with wide hips and toned waist, dark brown voluminous hair cascading down back, hazel eyes, alluring confident look, plump natural lips",
    },
    "nordic_redhead": {
        "name": "Nordic Redhead (Milky Skin / Freckles)",
        "character": "27yo caucasian woman, natural ginger red hair in loose waves framing face, light green eyes with dramatic catchlights, pale porcelain skin with subtle freckles across nose and shoulders, slim fit physique, defined collarbones",
    },
    "asian_petite": {
        "name": "Asian Petite (Soft / Aesthetic)",
        "character": "25yo east asian korean beauty, natural porcelain skin with visible pores, long sleek black hair, brown almond eyes, soft subtle seductive smile, slim toned physique with delicate curves",
    }
}

# Pose notes that reduce hand/anatomy failure rate (append to camera when building prompts)
HAND_SAFE_HINTS = (
    "one hand visible with five clear fingers, relaxed open palm or resting on hip, "
    "avoid interlocking fingers, avoid gripping glass or phone with both hands overlapping"
)

OF_SCENARIOS = {
    "bedroom_morning": {
        "name": "Morning Bedroom Selfie (Candid / Intimate)",
        "setting": "luxury modern bedroom, messy silk sheets in background, soft golden hour morning sunlight streaming through window creating dramatic rim lighting",
        "outfit": "wearing silk oversized open unbuttoned white shirt revealing delicate black lace bralette underneath, top button undone exposing cleavage",
        "camera": "iPhone selfie perspective, phone mostly out of frame, authentic casual snap aesthetic, soft window light, " + HAND_SAFE_HINTS,
        "hand_risk": "medium",
    },
    "mirror_selfie": {
        "name": "Bathroom Mirror Selfie (High Engagement)",
        "setting": "clean modern luxury bathroom, marble tiles, ambient ring light, warm vanity mirror reflection showing depth",
        "outfit": "wearing matching ribbed seamless sports crop top revealing toned midriff and navel, cheeky micro shorts sitting low on hips emphasizing hip curve",
        # Holding phone in mirror is high hand-risk; keep phone edge cropped / single clear hand
        "camera": "mirror selfie, smartphone held at edge of frame showing only one hand with five fingers, sharp reflection focus, authentic social media style, " + HAND_SAFE_HINTS,
        "hand_risk": "high",
    },
    "bikini_poolside": {
        "name": "Luxury Poolside / Vacation (Glamour)",
        "setting": "infinity pool overlooking tropical ocean, crystal clear water, vibrant sunny day, palm tree shadow creating dappled lighting on skin",
        "outfit": "wearing skimpy metallic micro bikini that barely covers essentials, droplets of water glistening on sun-kissed tan skin",
        "camera": "professional candid photography, 85mm portrait lens f/1.4, shallow depth of field, arms relaxed at sides or one hand in hair, " + HAND_SAFE_HINTS,
        "hand_risk": "low",
    },
    "boudoir_studio": {
        "name": "Sensual Boudoir Studio (Premium Content)",
        "setting": "dimly lit dark luxury suite, sheer drapes, moody cinematic neon rim lighting, warm shadows across body curves",
        # "sheer bodysuit" can trip some APIs; prefer lace lingerie set
        "outfit": "wearing black floral lace lingerie set, matching garter belt, sheer thigh-high stockings",
        "camera": "studio fashion editorial portrait, 50mm prime lens f/1.4, hands resting on thighs or in hair, ultra-detailed skin pores, Hasselblad aesthetic, creamy bokeh, " + HAND_SAFE_HINTS,
        "hand_risk": "low",
    },
    "gym_workout": {
        "name": "Fitness & Gym Aesthetic (Lifestyle)",
        "setting": "aesthetic modern gym interior, workout equipment in background, bright overhead flattering LED lighting",
        "outfit": "wearing skin-tight spandex yoga leggings, matching sports crop top, light natural sweat sheen on collarbone and cleavage",
        "camera": "front camera selfie resting on gym bench, natural candid posture, hands resting on thighs, relatable gym influencer aesthetic, " + HAND_SAFE_HINTS,
        "hand_risk": "medium",
    }
}

# Shared hand/anatomy negatives from stress tests (interlock/grip/extra digits)
HAND_ANATOMY_NEGATIVES = (
    "(extra fingers, six fingers, fused fingers, mutated fingers, elongated fingers, "
    "deformed hands, poorly drawn hands, floating hands, disconnected limbs, "
    "interlocked fingers, overlapping hands gripping object, missing fingers:1.45)"
)

# Juggernaut XL recommended negative prompt
JUGGERNAUT_NEGATIVE_PROMPTS = (
    "(worst quality, low quality, normal quality, lowres:1.2), "
    "(monochrome, grayscale:1.1), "
    "(3d render, cgi, cartoon, anime, illustration, sketch, drawing, painting:1.2), "
    f"{HAND_ANATOMY_NEGATIVES}, "
    "deformed, bad anatomy, missing limbs, extra limbs, poorly drawn face, disfigured, blurry, oversaturated, "
    "plastic skin, airbrushed, unnatural body proportions, watermark, text, signature"
)

# Realistic Vision V5.1 (SD 1.5) calibrated negative prompt
REALISTIC_VISION_NEGATIVE_PROMPTS = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), "
    "(worst quality, low quality, normal quality:1.3), "
    "(plastic skin, airbrushed, wax figure:1.25), "
    f"{HAND_ANATOMY_NEGATIVES}, "
    "(monochrome, grayscale, bad anatomy, bad proportions, unnatural body, distorted features:1.2), "
    "text, close up, cropped, out of frame, duplicate, morbid, mutilated, blurry, dehydrated, cloned face, "
    "watermark, signature, over-saturated"
)

OF_NEGATIVE_PROMPTS = REALISTIC_VISION_NEGATIVE_PROMPTS

# Photographic realism triggers
JUGGERNAUT_REALISM_BOOSTERS = (
    "candid cinematic photograph, authentic skin texture with visible micro-pores, "
    "natural lighting, 35mm film photography, f/1.8 shallow depth of field, photorealistic"
)

REALISTIC_VISION_REALISM_BOOSTERS = (
    "RAW photo, candid 35mm portrait photograph, authentic natural skin texture with visible pores and subtle imperfections, "
    "goosebumps on arms, soft natural lighting casting dramatic shadows across body contours, photorealistic catchlights in eyes, "
    "8k uhd, dslr, high quality, Fujifilm XT3, shallow depth of field, creamy bokeh"
)

OF_REALISM_BOOSTERS = REALISTIC_VISION_REALISM_BOOSTERS


def build_of_prompt(
    archetype_key: str = "slavic_blonde",
    scenario_key: str = "bedroom_morning",
    style: str = "realistic_vision"
) -> Tuple[str, str]:
    """
    Generates an ultra-realistic prompt tuned for photorealistic pipelines.
    Supports 'realistic_vision' (SD 1.5) and 'juggernaut' / 'sdxl'.
    """
    arch = OF_CHARACTER_ARCHETYPES.get(archetype_key, OF_CHARACTER_ARCHETYPES["slavic_blonde"])
    scen = OF_SCENARIOS.get(scenario_key, OF_SCENARIOS["bedroom_morning"])

    if style.lower() in ["realistic_vision", "sd15", "sd1.5", "default"]:
        positive = (
            f"{REALISTIC_VISION_REALISM_BOOSTERS}, "
            f"{arch['character']}, "
            f"{scen['outfit']}, "
            f"{scen['setting']}, "
            f"{scen['camera']}, "
            f"(hyperrealistic skin rendering, seductive sensual atmosphere:1.15), (masterpiece, best quality:1.2)"
        )
        return strip_beauty_tokens(positive), REALISTIC_VISION_NEGATIVE_PROMPTS
    elif style.lower() in ["juggernaut", "sdxl"]:
        positive = (
            f"candid photograph of a {arch['character']}, "
            f"{scen['outfit']}, {scen['setting']}, {scen['camera']}, "
            f"natural skin texture, visible pores, soft ambient lighting, "
            f"shot on 35mm lens, f/1.8, bokeh, photorealistic editorial quality"
        )
        return strip_beauty_tokens(positive), JUGGERNAUT_NEGATIVE_PROMPTS
    else:
        positive = (
            f"{OF_REALISM_BOOSTERS}, "
            f"{arch['character']}, "
            f"{scen['outfit']}, "
            f"{scen['setting']}, "
            f"{scen['camera']}, "
            f"photorealistic"
        )
        return strip_beauty_tokens(positive), OF_NEGATIVE_PROMPTS


def scenario_hand_risk(scenario_key: str) -> str:
    """Return low|medium|high hand-failure risk for a scenario (for UI / batch filters)."""
    scen = OF_SCENARIOS.get(scenario_key, {})
    return scen.get("hand_risk", "medium")
