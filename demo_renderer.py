"""
NeuronGen Demo Renderer.
Provides standalone aesthetic preview rendering for demo/test modes
without importing Gradio, FastAPI, or web dependencies.
"""

import time
import random
from PIL import Image, ImageDraw


def create_demo_artwork(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    index: int = 1
) -> Image.Image:
    """Creates an aesthetic cyberpunk/synthwave stylized preview render for demo/testing."""
    actual_seed = seed if seed >= 0 else (int(time.time() * 1000) + index) % (2**31)
    rng = random.Random(actual_seed)

    # Base canvas
    img = Image.new("RGB", (width, height), color=(14, 15, 24))
    draw = ImageDraw.Draw(img)

    # Draw gradient background
    c1 = (rng.randint(30, 70), rng.randint(15, 40), rng.randint(80, 140))
    c2 = (rng.randint(10, 25), rng.randint(10, 25), rng.randint(30, 60))
    for y in range(height):
        t = y / max(1, height)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Perspective grid lines
    horizon = int(height * 0.55)
    grid_color = (130, 70, 220)
    for i in range(-12, 13):
        x_bottom = int(width / 2 + i * (width / 10))
        draw.line([(width // 2, horizon), (x_bottom, height)], fill=grid_color, width=1)
    for gy in range(horizon, height, max(15, (height - horizon) // 10)):
        draw.line([(0, gy), (width, gy)], fill=grid_color, width=1)

    # Glowing geometric aesthetic element
    center_x, center_y = width // 2, int(height * 0.42)
    radius = min(width, height) // 4
    for r_offset in range(radius, 0, -8):
        glow_alpha = int(255 * (1 - r_offset / radius))
        glow_color = (min(255, 180 + glow_alpha // 3), min(255, 60 + glow_alpha // 4), min(255, 230))
        draw.ellipse(
            [(center_x - r_offset, center_y - r_offset), (center_x + r_offset, center_y + r_offset)],
            outline=glow_color,
            width=2
        )

    # Stylish framing border
    pad = 24
    draw.rectangle([(pad, pad), (width - pad, height - pad)], outline=(190, 110, 255), width=3)
    draw.rectangle([(pad + 6, pad + 6), (width - pad - 6, height - pad - 6)], outline=(70, 40, 110), width=1)

    # Corner brackets
    corner_len = 30
    for cx, cy, dx, dy in [
        (pad, pad, 1, 1),
        (width - pad, pad, -1, 1),
        (pad, height - pad, 1, -1),
        (width - pad, height - pad, -1, -1)
    ]:
        draw.line([(cx, cy), (cx + dx * corner_len, cy)], fill=(255, 170, 255), width=5)
        draw.line([(cx, cy), (cx, cy + dy * corner_len)], fill=(255, 170, 255), width=5)

    # Text overlays
    draw.text((pad + 20, pad + 20), "NEURONGEN STUDIO v2.3", fill=(255, 210, 255))
    draw.text((pad + 20, pad + 45), f"DEMO RENDER #{index} | SEED: {actual_seed}", fill=(200, 180, 255))
    draw.text((pad + 20, height - pad - 65), f"RES: {width}x{height} | STEPS: {steps} | CFG: {cfg}", fill=(180, 180, 220))

    # Prompt snippet (cleaned)
    clean_p = prompt.replace("\n", " ")[:80]
    draw.text((pad + 20, height - pad - 35), f"PROMPT: \"{clean_p}...\"", fill=(240, 240, 255))

    return img
