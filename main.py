"""
NeuronGen Web Studio - Interactive Web UI for Stable Diffusion generation.
Provides Single Generation, Batch Processing, Preset Prompts, Engine Selection, and Diagnostics.
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Patch gradio_client for pydantic 2.x compatibility with boolean schemas
try:
    import gradio_client.utils as gc_utils
    _orig_json_to_py = gc_utils._json_schema_to_python_type
    gc_utils._json_schema_to_python_type = lambda s, d: "Any" if isinstance(s, bool) else _orig_json_to_py(s, d)
    _orig_get_type = gc_utils.get_type
    gc_utils.get_type = lambda s: "Any" if isinstance(s, bool) else _orig_get_type(s)
except Exception:
    pass

# Patch huggingface_hub HfFolder removal for Gradio 4.41.0 compatibility
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token(): return None
            @staticmethod
            def save_token(token): pass
            @staticmethod
            def delete_token(): pass
        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

import gradio as gr
import torch

# Project modules
from config import get_config, save_config
from prompt_processor import PromptProcessor
from webui_api import StableDiffusionWebUI
from batch_generator import BatchGenerator
from grok_client import GrokClient

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Studio port
STUDIO_PORT = 7861

# Initialize subsystems
app_config = get_config()
prompt_proc = PromptProcessor(app_config)
batch_gen = BatchGenerator(app_config)


def get_available_presets() -> dict:
    """Scan prompts directory and return dict of preset_name -> prompt_content."""
    presets = {}
    if PROMPTS_DIR.exists():
        for p_file in sorted(PROMPTS_DIR.glob("*.txt")):
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        presets[p_file.name] = content
            except Exception:
                pass
    return presets


def load_preset_content(preset_name: str) -> str:
    presets = get_available_presets()
    return presets.get(preset_name, "")


def get_valid_checkpoints() -> List[Path]:
    """Find valid checkpoint files (>100MB) in models/checkpoints."""
    valid = []
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.glob("*.safetensors"):
            if f.stat().st_size > 100 * 1024 * 1024:
                valid.append(f)
    return valid


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
    draw.text((pad + 20, pad + 20), "NEURONGEN STUDIO v2.1", fill=(255, 210, 255))
    draw.text((pad + 20, pad + 45), f"DEMO RENDER #{index} | SEED: {actual_seed}", fill=(200, 180, 255))
    draw.text((pad + 20, height - pad - 65), f"RES: {width}x{height} | STEPS: {steps} | CFG: {cfg}", fill=(180, 180, 220))
    
    # Prompt snippet (cleaned)
    clean_p = prompt.replace("\n", " ")[:80]
    draw.text((pad + 20, height - pad - 35), f"PROMPT: \"{clean_p}...\"", fill=(240, 240, 255))

    return img


def check_system_status(webui_url: Optional[str] = None) -> str:
    """Returns a markdown summary of system, device, models, and API status."""
    url = webui_url or app_config.get("webui", {}).get("url", "http://127.0.0.1:7860")
    client = StableDiffusionWebUI({"webui": {"url": url}})
    is_online = client.is_running()

    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only (CUDA unavailable)"

    # Check models
    ckpts = []
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.glob("*.*"):
            size = f.stat().st_size
            size_mb = size / (1024 * 1024)
            if size < 1024 * 1024:
                ckpts.append(f"- ⚠️ `{f.name}`: **Invalid (<1MB)**")
            else:
                ckpts.append(f"- ✅ `{f.name}`: **Ready** ({size_mb:.1f} MB)")
    if not ckpts:
        ckpts.append("- *No `.safetensors` checkpoints currently in `models/checkpoints/`*")

    status_md = f"""
### 🖥️ Hardware & Device
- **PyTorch Version:** `{torch.__version__}`
- **Active Device:** `{device_name}`
- **CUDA Available:** `{cuda_avail}`

### 🌐 Automatic1111 WebUI API Target
- **Target URL:** `{url}`
- **Connection Status:** {"🟢 **CONNECTED (Online)**" if is_online else "🔴 **OFFLINE (Not reachable)**"}
> *Note: Automatic1111 usually runs on port 7860. This NeuronGen Studio runs on port {STUDIO_PORT}.*

### 📦 Local Checkpoints Status
{chr(10).join(ckpts)}
"""
    return status_md


def generate_single_ui(
    engine: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    batch_size: int,
    seed: int,
    sampler_name: str,
    webui_host: str,
    progress=gr.Progress()
) -> Tuple[List[Image.Image], str]:
    """Handles generation across the selected engine."""
    if not prompt or not prompt.strip():
        return [], "⚠️ Please enter a prompt."

    progress(0.1, desc="Parsing prompt parameters...")

    # Extract any inline params
    norm_prompt, tokens = prompt_proc.process_prompt(prompt)
    _, inline_params = prompt_proc.extract_parameters(prompt)

    actual_steps = inline_params.get("steps", steps)
    actual_cfg = inline_params.get("cfg_scale", cfg_scale)
    actual_width = inline_params.get("width", width)
    actual_height = inline_params.get("height", height)
    actual_seed = seed if seed >= 0 else inline_params.get("seed", -1)
    actual_batch = max(1, int(batch_size))

    # ENGINE 1: Demo & Test Mode (Instant Preview)
    if "Demo" in engine or "Test" in engine:
        progress(0.4, desc="Rendering demo artwork...")
        images = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx in range(1, actual_batch + 1):
            demo_img = create_demo_artwork(
                prompt=norm_prompt,
                width=int(actual_width),
                height=int(actual_height),
                steps=int(actual_steps),
                cfg=float(actual_cfg),
                seed=int(actual_seed) + idx if actual_seed >= 0 else -1,
                index=idx
            )
            filename = f"demo_{timestamp}_{idx}.png"
            filepath = RESULTS_DIR / filename
            demo_img.save(filepath, format="PNG")
            images.append(demo_img)

            # Metadata
            meta_path = RESULTS_DIR / f"demo_{timestamp}_{idx}_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "engine": "Demo / Test Preview Mode",
                    "timestamp": datetime.now().isoformat(),
                    "prompt": norm_prompt,
                    "negative_prompt": negative_prompt,
                    "parameters": {
                        "steps": actual_steps,
                        "cfg_scale": actual_cfg,
                        "width": actual_width,
                        "height": actual_height,
                        "seed": actual_seed,
                        "batch_size": actual_batch,
                        "sampler": sampler_name
                    }
                }, f, indent=2)

        progress(1.0, desc="Done!")
        msg = f"✨ **Demo Render Complete!** Generated {len(images)} preview image(s).\nSaved to: `{RESULTS_DIR}`"
        return images, msg

    # ENGINE 2: Local Diffusers Pipeline
    elif "Local Diffusers" in engine:
        progress(0.2, desc="Checking local checkpoint files...")
        ckpts = get_valid_checkpoints()
        if not ckpts:
            return [], (
                "⚠️ **No valid `.safetensors` model found in `models/checkpoints/`!**\n\n"
                "To use Local Diffusers:\n"
                "1. Download a model (e.g. SDXL or RealisticVision) and place the `.safetensors` file into `models/checkpoints/`.\n"
                "2. Or switch the **Generation Engine** dropdown to **'Demo & Test Mode (Instant Preview)'**."
            )

        progress(0.4, desc="Loading PyTorch diffusers pipeline...")
        try:
            from generate import generate_image_diffusers
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(RESULTS_DIR / f"diffusers_{timestamp}.png")

            generated = generate_image_diffusers(
                prompt=norm_prompt,
                output_path=out_path,
                steps=int(actual_steps),
                cfg_scale=float(actual_cfg),
                width=int(actual_width),
                height=int(actual_height),
                seed=int(actual_seed),
                checkpoint_path=str(ckpts[0]),
                negative_prompt=neg_prompt
            )

            if generated:
                imgs = [Image.open(p) for p in generated]
                return imgs, f"✅ Generated {len(imgs)} image(s) via Diffusers!\nSaved to: `{RESULTS_DIR}`"
            else:
                return [], "❌ Local diffusers generation failed. Check console logs for details."
        except Exception as e:
            return [], f"❌ Error loading diffusers pipeline: {e}"

    # ENGINE 3: Automatic1111 WebUI API
    else:
        target_url = (webui_host or "http://127.0.0.1:7860").strip().rstrip("/")
        
        # Guard against self-referencing port
        if f":{STUDIO_PORT}" in target_url:
            return [], (
                f"⚠️ **Configuration Error**: Target URL `{target_url}` is pointing to NeuronGen Studio itself!\n"
                f"Automatic1111 WebUI runs separately, usually at `http://127.0.0.1:7860`.\n"
                f"If you don't have Automatic1111 running, change the **Generation Engine** to **'Demo & Test Mode'**."
            )

        progress(0.3, desc=f"Connecting to WebUI API at {target_url}...")
        client = StableDiffusionWebUI({"webui": {"url": target_url}})

        if not client.is_running():
            return [], (
                f"❌ **Cannot connect to Automatic1111 at `{target_url}`!**\n\n"
                f"**To resolve:**\n"
                f"1. Make sure your Automatic1111 WebUI is started with the `--api` flag (e.g. `webui.bat --api`).\n"
                f"2. Verify the URL is correct (default: `http://127.0.0.1:7860`).\n\n"
                f"💡 **Don't have Automatic1111 running right now?**\n"
                f"Switch the **Generation Engine** dropdown above to **'Demo & Test Mode (Instant Preview)'** to test the studio right away!"
            )

        progress(0.5, desc="WebUI API generating image(s)...")
        images, info_dict, errors = client.generate_images(
            positive_prompt=norm_prompt,
            negative_prompt=negative_prompt,
            steps=int(actual_steps),
            cfg_scale=float(actual_cfg),
            width=int(actual_width),
            height=int(actual_height),
            batch_size=actual_batch,
            seed=int(actual_seed),
            sampler_name=sampler_name
        )

        if errors:
            error_report = "\n".join(errors)
            return [], f"❌ **Generation Error:**\n{error_report}"

        progress(0.8, desc="Saving results to disk...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, img in enumerate(images, 1):
            filename = f"webui_{timestamp}_{i}.png"
            filepath = RESULTS_DIR / filename
            img.save(filepath, format="PNG")

        progress(1.0, desc="Done!")
        info_text = f"✅ Generated {len(images)} image(s) via WebUI API!\nSaved to: `{RESULTS_DIR}`"
        return images, info_text


def build_interface() -> gr.Blocks:
    """Build and return the Gradio UI Blocks application."""
    presets = get_available_presets()
    preset_choices = ["None"] + list(presets.keys())

    custom_theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="pink",
        neutral_hue="slate"
    )

    with gr.Blocks(title="NeuronGen Studio v2.1", theme=custom_theme) as demo:
        gr.Markdown(
            """
            # 🎨 NeuronGen Studio v2.1
            ### High Performance Neural Image Generation Suite
            """
        )

        with gr.Tabs():
            with gr.TabItem("🖼️ Single / Studio Generation"):
                with gr.Row():
                    with gr.Column(scale=5):
                        # Engine selector & local hardware detection
                        valid_ckpts = get_valid_checkpoints()
                        default_engine = "Local Diffusers Pipeline (PyTorch)" if valid_ckpts else "Demo & Test Mode (Instant Preview)"

                        engine_dropdown = gr.Dropdown(
                            choices=[
                                "Local Diffusers Pipeline (PyTorch)",
                                "Automatic1111 WebUI API (Remote or Local)",
                                "Demo & Test Mode (Instant Preview)"
                            ],
                            value=default_engine,
                            label="⚡ Generation Engine",
                            info="Local Diffusers runs directly on your RTX 4070 Ti using models in models/checkpoints/."
                        )

                        if valid_ckpts:
                            gr.Markdown(f"🟢 **Local Checkpoint Loaded:** `{valid_ckpts[0].name}` on **RTX 4070 Ti (CUDA)**")

                        # Grok OnlyFans AI Assistant
                        with gr.Accordion("✨ Grok AI (xAI) - OnlyFans Prompt Architect & Captions", open=False):
                            gr.Markdown("Powered by Grok's uncensored intelligence for hyper-realistic OnlyFans scenes, skin details, lighting, and high-converting post captions.")
                            grok_key_input = gr.Textbox(
                                label="xAI Grok API Key",
                                placeholder="xai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                                type="password",
                                value=os.environ.get("GROK_API_KEY", "") or os.environ.get("XAI_API_KEY", "")
                            )
                            with gr.Row():
                                grok_style_dropdown = gr.Dropdown(
                                    choices=[
                                        "📱 OnlyFans Bedroom / Mirror Selfie (Casual & Intimate)",
                                        "🔥 Sensual Boudoir / Silk & Lace",
                                        "💪 Fitness / Gym Spandex & Leggings",
                                        "🏖️ Exotic Beach / Golden Hour Bikini",
                                        "🏙️ Penthouse Glamour / Luxury Evening",
                                        "🚿 Steamy Shower / Wet Hair Aesthetic"
                                    ],
                                    value="📱 OnlyFans Bedroom / Mirror Selfie (Casual & Intimate)",
                                    label="🎭 Aesthetic Theme"
                                )
                                grok_concept_input = gr.Textbox(
                                    label="Idea / Appearance / Details",
                                    placeholder="e.g. 21yo blonde, piercing blue eyes, playful smirk, sheer white tank top",
                                    lines=1
                                )
                            grok_gen_btn = gr.Button("🔮 Grok AI Generate Prompt & Caption", variant="secondary")
                            grok_caption_box = gr.Textbox(
                                label="📱 OnlyFans Post Caption & PPV Teaser (Ready to Copy)",
                                lines=2,
                                interactive=True
                            )

                        # Discord / Midjourney Prompt Importer
                        with gr.Accordion("💬 Discord / Midjourney Prompt Importer", open=False):
                            gr.Markdown("Paste prompts copied directly from your Discord server ([Сервер plaksavaksa](https://discord.gg/j8mjujwcG)). Automatically converts `/imagine prompt:`, extracts `--ar 9:16`, strips flags, and tunes resolution.")
                            discord_prompt_box = gr.Textbox(
                                label="Paste Discord Generation Prompt",
                                placeholder="/imagine prompt: 1girl, bedroom selfie, silk camisole --ar 9:16 --v 6.0 --s 250 --no ugly, deformed",
                                lines=2
                            )
                            discord_import_btn = gr.Button("📥 Import & Auto-Configure Prompt", variant="secondary")
                            discord_status_msg = gr.Markdown("")

                        preset_dropdown = gr.Dropdown(
                            choices=preset_choices,
                            value="onlyfans_bedroom_selfie.txt" if "onlyfans_bedroom_selfie.txt" in presets else ("best_quality_prompt.txt" if "best_quality_prompt.txt" in presets else "None"),
                            label="📁 Quick Preset Templates"
                        )
                        prompt_input = gr.Textbox(
                            label="Positive Prompt",
                            placeholder="Enter description or keywords...",
                            lines=5,
                            value=presets.get("onlyfans_bedroom_selfie.txt", presets.get("best_quality_prompt.txt", ""))
                        )
                        neg_prompt_input = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value=prompt_proc.default_negative
                        )

                        with gr.Accordion("⚙️ Generation Parameters", open=True):
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=10, maximum=100, value=20, step=1, label="Sampling Steps"
                                )
                                cfg_slider = gr.Slider(
                                    minimum=1.0, maximum=20.0, value=7.0, step=0.5, label="CFG Guidance Scale"
                                )
                            with gr.Row():
                                width_dropdown = gr.Dropdown(
                                    choices=[512, 768, 896, 1024, 1152, 1280],
                                    value=512,
                                    label="Width"
                                )
                                height_dropdown = gr.Dropdown(
                                    choices=[512, 768, 896, 1024, 1152, 1280],
                                    value=768,
                                    label="Height"
                                )
                            with gr.Row():
                                batch_slider = gr.Slider(
                                    minimum=1, maximum=8, value=1, step=1, label="Batch Size"
                                )
                                seed_input = gr.Number(
                                    value=-1, label="Seed (-1 for Random)", precision=0
                                )
                            with gr.Row():
                                sampler_dropdown = gr.Dropdown(
                                    choices=[
                                        "DPM++ 2M Karras",
                                        "Euler a",
                                        "Euler",
                                        "DPM++ SDE Karras",
                                        "DPM++ 2S a Karras",
                                        "DDIM"
                                    ],
                                    value="DPM++ 2M Karras",
                                    label="Sampler"
                                )
                                webui_host_input = gr.Textbox(
                                    label="WebUI Host URL (for Automatic1111 mode)",
                                    value="http://127.0.0.1:7860"
                                )

                        generate_btn = gr.Button("🚀 Generate Images", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        output_gallery = gr.Gallery(
                            label="Generated Artwork",
                            show_label=True,
                            elem_id="gallery",
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height="auto"
                        )
                        output_status = gr.Markdown("Ready for generation.")

            with gr.TabItem("📊 Batch Processing"):
                gr.Markdown("### Run multiple prompts automatically in sequence")
                batch_engine = gr.Dropdown(
                    choices=["Demo & Test Mode", "Automatic1111 WebUI API"],
                    value="Demo & Test Mode",
                    label="Batch Engine"
                )
                batch_prompts_input = gr.Textbox(
                    label="Prompts (One per line)",
                    lines=8,
                    placeholder="prompt 1...\nprompt 2...\nprompt 3...",
                    value="masterpiece, 1girl, cute dress, smiling\nmasterpiece, 1girl, cyberpunk aesthetic, neon glow\nmasterpiece, 1girl, fantasy landscape, magical aura"
                )
                run_batch_btn = gr.Button("⚡ Start Batch Processing", variant="secondary")
                batch_status_md = gr.Markdown("Batch status: Idle")

            with gr.TabItem("🔍 System Status & Diagnostics"):
                status_display = gr.Markdown(check_system_status())
                refresh_status_btn = gr.Button("🔄 Refresh Diagnostics", variant="secondary")

        # Event Handlers
        def on_preset_change(choice):
            if choice == "None":
                return gr.update()
            return gr.update(value=load_preset_content(choice))

        preset_dropdown.change(
            fn=on_preset_change,
            inputs=[preset_dropdown],
            outputs=[prompt_input]
        )

        def on_grok_enhance(api_key, concept, style):
            if not concept:
                concept = "sensual woman, alluring gaze, authentic smartphone photo"
            client = GrokClient(api_key=api_key)
            pos, neg, caption = client.enhance_prompt(base_prompt=concept, style_preset=style)
            return pos, neg, caption

        grok_gen_btn.click(
            fn=on_grok_enhance,
            inputs=[grok_key_input, grok_concept_input, grok_style_dropdown],
            outputs=[prompt_input, neg_prompt_input, grok_caption_box]
        )

        def on_discord_import(raw_text):
            if not raw_text.strip():
                return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Please enter a Discord prompt to import."
            res = prompt_proc.parse_discord_or_midjourney_prompt(raw_text)
            flag_str = f"✅ Extracted & configured! Flags: {', '.join(res['flags_detected'])}" if res['flags_detected'] else "✅ Imported Discord prompt successfully!"
            return res["positive_prompt"], res["negative_prompt"], res["width"], res["height"], flag_str

        discord_import_btn.click(
            fn=on_discord_import,
            inputs=[discord_prompt_box],
            outputs=[prompt_input, neg_prompt_input, width_dropdown, height_dropdown, discord_status_msg]
        )

        generate_btn.click(
            fn=generate_single_ui,
            inputs=[
                engine_dropdown,
                prompt_input,
                neg_prompt_input,
                steps_slider,
                cfg_slider,
                width_dropdown,
                height_dropdown,
                batch_slider,
                seed_input,
                sampler_dropdown,
                webui_host_input
            ],
            outputs=[output_gallery, output_status]
        )

        def on_batch_run(engine_choice, prompts_text, webui_host):
            lines = [l.strip() for l in prompts_text.splitlines() if l.strip()]
            if not lines:
                return "⚠️ Please enter at least one prompt."
            
            if "Demo" in engine_choice:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for i, p in enumerate(lines, 1):
                    img = create_demo_artwork(p, 896, 1152, 25, 7.5, -1, i)
                    img.save(RESULTS_DIR / f"batch_demo_{timestamp}_{i}.png")
                return f"✅ Batch complete! Generated and saved {len(lines)} images to `{RESULTS_DIR}`."
            else:
                client = StableDiffusionWebUI({"webui": {"url": webui_host or "http://127.0.0.1:7860"}})
                if not client.is_running():
                    return f"❌ Automatic1111 is not reachable at {client.base_url}. Switch engine to 'Demo & Test Mode' or launch Automatic1111."
                b_gen = BatchGenerator({"webui": {"url": webui_host or "http://127.0.0.1:7860"}})
                files = b_gen.generate_batch(lines)
                return f"✅ Batch complete! Saved {len(files)} files to `{RESULTS_DIR}`."

        run_batch_btn.click(
            fn=on_batch_run,
            inputs=[batch_engine, batch_prompts_input, webui_host_input],
            outputs=[batch_status_md]
        )

        refresh_status_btn.click(
            fn=check_system_status,
            inputs=[webui_host_input],
            outputs=[status_display]
        )

    return demo


def main():
    demo = build_interface()
    # Find free port starting at 7861
    port = STUDIO_PORT
    import socket
    while port < 7900:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        in_use = sock.connect_ex(('127.0.0.1', port)) == 0
        sock.close()
        if not in_use:
            break
        port += 1

    print(f"\n[INFO] Starting NeuronGen Web Studio on http://127.0.0.1:{port} ...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        show_error=True,
        show_api=False
    )


if __name__ == "__main__":
    main()
