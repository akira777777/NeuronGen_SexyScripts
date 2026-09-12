"""
Curated prompt presets specifically designed for photorealistic OnlyFans model generation.
Includes detailed photography tags, skin textures, lighting, outfits, poses, and character archetypes.
"""

OF_CHARACTER_ARCHETYPES = {
    "slavic_blonde": {
        "name": "Slavic Blonde (Sporty / Glamour)",
        "character": "21yo slavic woman, athletic toned body, natural blonde wavy hair, blue eyes, cute soft smile, natural lip shape, symmetrical face, soft cheekbones",
    },
    "latina_brunette": {
        "name": "Latina Brunette (Curvy / Exotic)",
        "character": "22yo latina woman, tan smooth skin, hourglass figure, wide hips, dark brown voluminous hair, hazel eyes, alluring confident look, plump lips",
    },
    "nordic_redhead": {
        "name": "Nordic Redhead (Milky Skin / Freckles)",
        "character": "20yo caucasian woman, natural ginger red hair, light green eyes, subtle cute nose freckles, pale porcelain skin, slim fit physique",
    },
    "asian_petite": {
        "name": "Asian Petite (Soft / Aesthetic)",
        "character": "21yo east asian korean beauty, flawless glass skin, long sleek black hair, brown almond eyes, soft subtle smile, slim fit physique",
    }
}

OF_SCENARIOS = {
    "bedroom_morning": {
        "name": "Morning Bedroom Selfie (Candid / Intimate)",
        "setting": "luxury modern bedroom, messy silk sheets, soft golden hour morning sunlight streaming through window",
        "outfit": "wearing silk oversized open unbuttoned white shirt, delicate lace bralette",
        "camera": "iPhone selfie perspective, arm extended, natural slight motion blur, authentic casual snap, realistic flash reflection",
    },
    "mirror_selfie": {
        "name": "Bathroom Mirror Selfie (High Engagement)",
        "setting": "clean modern bathroom, marble tiles, ambient ring light, warm vanity mirror reflections",
        "outfit": "wearing matching ribbed seamless sports bra and micro shorts, showing flat stomach and toned abs",
        "camera": "holding smartphone taking mirror selfie, sharp reflection focus, realistic phone case, authentic social media style",
    },
    "bikini_poolside": {
        "name": "Luxury Poolside / Vacation (Glamour)",
        "setting": "infinity pool overlooking tropical ocean, crystal clear water, vibrant sunny day, palm tree shadow",
        "outfit": "wearing skimpy metallic micro bikini, droplets of water glistening on skin, sun-kissed tan skin",
        "camera": "professional candid photography, 85mm portrait lens, f/1.8 shallow depth of field, natural highlights",
    },
    "boudoir_studio": {
        "name": "Sensual Boudoir Studio (Premium Content)",
        "setting": "dimly lit dark aesthetic luxury suite, sheer drapes, moody cinematic neon rim lighting, warm shadows",
        "outfit": "wearing sheer black floral lace lingerie bodysuit, garter belt, thigh high sheer stockings",
        "camera": "studio fashion editorial portrait, 50mm prime lens, ultra-detailed skin pores, Hasselblad aesthetic",
    },
    "gym_workout": {
        "name": "Fitness & Gym Aesthetic (Lifestyle)",
        "setting": "aesthetic modern gym interior, workout equipment in background, bright overhead LED lighting",
        "outfit": "wearing skin-tight spandex yoga leggings, sports crop top, light natural sweat sheen on collarbone",
        "camera": "front camera selfie resting on gym bench, natural candid posture, relatable gym influencer look",
    }
}

# Juggernaut XL recommended negative prompt
JUGGERNAUT_NEGATIVE_PROMPTS = (
    "(worst quality, low quality, normal quality, lowres:1.2), "
    "(monochrome, grayscale:1.1), "
    "(3d render, cgi, cartoon, anime, illustration, sketch, drawing, painting:1.2), "
    "deformed, bad anatomy, bad hands, missing limbs, extra limbs, extra fingers, mutated fingers, "
    "fused fingers, poorly drawn hands, poorly drawn face, disfigured, blurry, oversaturated, "
    "plastic skin, airbrushed, unnatural body proportions, watermark, text, signature"
)

# Standard SD 1.5 negative prompt fallback
OF_NEGATIVE_PROMPTS = JUGGERNAUT_NEGATIVE_PROMPTS

# Photographic realism triggers for Juggernaut XL
JUGGERNAUT_REALISM_BOOSTERS = (
    "candid cinematic photograph, authentic skin texture with visible micro-pores, "
    "natural lighting, 35mm film photography, f/1.8 shallow depth of field, photorealistic"
)

OF_REALISM_BOOSTERS = JUGGERNAUT_REALISM_BOOSTERS


def build_of_prompt(
    archetype_key: str = "slavic_blonde",
    scenario_key: str = "bedroom_morning",
    style: str = "juggernaut"
) -> Tuple[str, str]:
    """
    Generates an ultra-realistic prompt tuned for Juggernaut XL and photorealistic pipelines.
    Avoids tag-soup and brackets that hurt SDXL image quality.
    """
    arch = OF_CHARACTER_ARCHETYPES.get(archetype_key, OF_CHARACTER_ARCHETYPES["slavic_blonde"])
    scen = OF_SCENARIOS.get(scenario_key, OF_SCENARIOS["bedroom_morning"])

    if style.lower() in ["juggernaut", "sdxl"]:
        positive = (
            f"candid photograph of a {arch['character']}, "
            f"{scen['outfit']}, {scen['setting']}, {scen['camera']}, "
            f"natural skin texture, visible pores, soft ambient lighting, "
            f"shot on 35mm lens, f/1.8, bokeh, photorealistic editorial quality"
        )
        return positive, JUGGERNAUT_NEGATIVE_PROMPTS
    else:
        positive = (
            f"{JUGGERNAUT_REALISM_BOOSTERS}, "
            f"{arch['character']}, "
            f"{scen['outfit']}, "
            f"{scen['setting']}, "
            f"{scen['camera']}, "
            f"photorealistic"
        )
        return positive, OF_NEGATIVE_PROMPTS
