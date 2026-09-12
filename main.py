"""
NeuronGen Web Studio v2.2 - Интерактивная веб-студия генерации изображений.
Простой и понятный интерфейс на русском языке для управления локальной генерацией на RTX 4070 Ti,
внешним Automatic1111 WebUI и тестовым демо-режимом.
"""

import os
import sys
import json
import uuid
import time
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Настройка UTF-8 для корректного вывода в Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Патчи совместимости gradio_client для pydantic 2.x
try:
    import gradio_client.utils as gc_utils
    _orig_json_to_py = gc_utils._json_schema_to_python_type
    gc_utils._json_schema_to_python_type = lambda s, d: "Any" if isinstance(s, bool) else _orig_json_to_py(s, d)
    _orig_get_type = gc_utils.get_type
    gc_utils.get_type = lambda s: "Any" if isinstance(s, bool) else _orig_get_type(s)
except Exception:
    pass

# Патч huggingface_hub HfFolder для совместимости с Gradio 4.x
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
from typing import Optional

# Grok API key from environment (ONLYFANS AI enhancement)
GROK_API_KEY: Optional[str] = os.environ.get("XAI_API_KEY") or None


def is_grok_configured() -> bool:
    """Check if Grok API key is set."""
    return bool(GROK_API_KEY and len(GROK_API_KEY.strip()) > 5)


# Модули проекта
from config import get_config, get_profile, apply_profile_prompt, save_config
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

# Порт студии
STUDIO_PORT = 7861

# Инициализация подсистем
app_config = get_config()
prompt_proc = PromptProcessor(app_config)
batch_gen = BatchGenerator(app_config)

# Человекопонятные русские названия для готовых шаблонов
PRESET_LABELS: Dict[str, str] = {
    "onlyfans_bedroom_selfie.txt": "📱 Селфи в спальне (Шёлковая рубашка, нежный утренний свет)",
    "onlyfans_mirror_selfie.txt": "🪞 Селфи у зеркала в ванной (Спортивный топ, мрамор, ринглайт)",
    "onlyfans_boudoir_studio.txt": "🔥 Чувственный будуар (Кружевное боди, кинематографичный свет)",
    "juggernaut_poolside_bikini.txt": "🏖️ Пляж и бассейн (Бикини, тропический океан, загар)",
    "juggernaut_gym_fitness.txt": "💪 Фитнес в спортзале (Леггинсы, спортивная фигура, зал)",
    "best_quality_prompt.txt": "💎 Шедевр качества (Универсальный ультра-фотореализм)",
    "sexy_variant.txt": "✨ Стильный эстетичный образ (Элегантный стиль)",
    "sexy_variant2.txt": "🌹 Премиум фотосессия (Гламур и детализация)",
    "examples.txt": "📝 Примеры описаний",
}

# Описания типажей внешности для быстрой вставки
ARCHETYPES_RU = {
    "👱‍♀️ Славянка (Блондинка)": "21yo slavic woman, athletic toned body with visible abs, natural blonde wavy hair cascading over shoulders, piercing blue eyes, cute soft seductive smile, symmetrical face",
    "👩 Латина (Загар / Брюнетка)": "22yo latina woman, tan smooth skin with sun-kissed glow, voluptuous hourglass figure, dark brown voluminous hair, hazel eyes, alluring confident look, plump natural lips",
    "👩‍🦰 Скандинавка (Рыжая / Веснушки)": "20yo caucasian woman, natural ginger red hair in loose waves, light green eyes, pale porcelain skin with subtle freckles across nose, slim fit physique",
    "👧 Азиатка (Нежный образ)": "21yo east asian korean beauty, flawless porcelain glass skin, long sleek black hair, brown almond eyes, soft subtle smile, slim fit physique",
}

# Режимы генерации
ENGINE_LOCAL = "⚡ Локальная видеокарта (RTX 4070 Ti — Быстро и автономно)"
ENGINE_WEBUI = "🌐 Внешний Automatic1111 WebUI (Сервер на порту 7860)"
ENGINE_DEMO = "🎨 Демо-режим (Мгновенный тест без нагрузки)"


def get_available_presets() -> Dict[str, str]:
    """Сканирует папку prompts и возвращает словарь: понятное название -> текст промпта."""
    presets = {}
    if PROMPTS_DIR.exists():
        for p_file in sorted(PROMPTS_DIR.glob("*.txt")):
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        label = PRESET_LABELS.get(p_file.name, f"📄 {p_file.name}")
                        presets[label] = content
            except Exception:
                pass
    return presets


def get_valid_checkpoints() -> List[Path]:
    """Поиск файлов моделей (>100MB) в models/checkpoints."""
    valid = []
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.glob("*.safetensors"):
            if f.stat().st_size > 100 * 1024 * 1024:
                valid.append(f)
    return valid


def open_results_folder() -> str:
    """Открывает папку с результатами в проводнике Windows."""
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(RESULTS_DIR.resolve()))
            return "📂 Папка с результатами успешно открыта в проводнике Windows!"
        else:
            return f"📂 Путь к папке: {RESULTS_DIR.resolve()}"
    except Exception as e:
        return f"Не удалось автоматически открыть папку: {e}\nПуть: {RESULTS_DIR.resolve()}"


from demo_renderer import create_demo_artwork



def check_system_status(webui_url: Optional[str] = None) -> str:
    """Возвращает понятную сводку о видеокарте, памяти, моделях и серверах на русском языке."""
    url = webui_url or app_config.get("webui", {}).get("url", "http://127.0.0.1:7860")
    client = StableDiffusionWebUI({"webui": {"url": url}})
    is_online = client.is_running()

    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "Только процессор (CUDA недоступна)"
    
    free_vram_str = "Н/Д"
    total_vram_str = "Н/Д"
    if cuda_avail:
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_vram_str = f"{free_b / (1024**3):.1f} ГБ"
            total_vram_str = f"{total_b / (1024**3):.1f} ГБ"
        except Exception:
            pass

    # Проверка моделей
    ckpts = []
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.glob("*.*"):
            size = f.stat().st_size
            size_mb = size / (1024 * 1024)
            if size < 1024 * 1024:
                ckpts.append(f"- ⚠️ `{f.name}`: Некорректный файл (размер < 1 МБ)")
            else:
                ckpts.append(f"- 🟢 `{f.name}`: **Готова к генерации** ({size_mb:.0f} МБ)")
    if not ckpts:
        ckpts.append("- ℹ️ *В папке `models/checkpoints/` пока нет файлов `.safetensors`*")

    status_md = f"""
### 🖥️ Ваша видеокарта и система
- **Модель видеокарты:** `{device_name}`
- **Ускорение CUDA:** {"🟢 **Активно (Максимальная скорость)**" if cuda_avail else "⚠️ **Выключено**"}
- **Видеопамять (VRAM):** свободно `{free_vram_str}` из `{total_vram_str}`
- **Версия PyTorch:** `{torch.__version__}`

---

### 📦 Установленные нейросетевые модели (Checkpoints)
{chr(10).join(ckpts)}

> 💡 *Локальная генерация на вашей RTX 4070 Ti использует модель из этой папки напрямую без сторонних программ.*

---

### 🌐 Внешний сервер Automatic1111 WebUI
- **Адрес для подключения:** `{url}`
- **Статус соединения:** {"🟢 **ПОДКЛЮЧЕНО (Сервер работает)**" if is_online else "⚪ **НЕ ЗАПУЩЕН (Не требуется для локальной генерации)**"}
> *Подсказка: Если вы хотите генерировать фото локально через встроенную студию, запускать Automatic1111 не нужно.*
"""
    return status_md



def _write_ui_run_log(
    *,
    engine: str,
    positive: str,
    negative: str,
    cfg: float,
    steps: int,
    seed: int,
    width: int,
    height: int,
    batch_size: int,
    profile_name: str,
) -> dict:
    """Persist effective Gradio params for realism QA / UI badge."""
    run_id = uuid.uuid4().hex[:12]
    payload = {
        "run_id": run_id,
        "engine": engine,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "cfg": cfg,
        "steps": steps,
        "seed": seed,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "profile": profile_name,
        "frame_total": batch_size,
        "source": "gradio",
        "timestamp": datetime.now().isoformat(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"run_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def _format_run_badge(run: dict) -> str:
    neg = str(run.get("negative_prompt") or "")
    neg_short = (neg[:96] + "…") if len(neg) > 96 else neg
    return (
        f"\n\n🏷️ **run_id:** `{run.get('run_id')}` | "
        f"CFG `{run.get('cfg')}` | seed `{run.get('seed')}` | "
        f"profile `{run.get('profile')}`\n"
        f"neg: `{neg_short}`"
    )

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
    profile_name: str = "realism",
    progress=gr.Progress()
) -> Tuple[List[Image.Image], str]:
    """Универсальный диспетчер генерации с подробным информированием на русском языке."""
    if not prompt or not prompt.strip():
        return [], "⚠️ **Пожалуйста, введите описание фото (промпт) или выберите готовый шаблон выше!**"

    progress(0.1, desc="Анализ описания и параметров...")

    prof = get_profile(profile_name)
    prompt = apply_profile_prompt(prompt, prof)

    # Grok AI Enhancement (ONLYFANS photorealism optimization)
    if grok_enabled and is_grok_configured():
        progress(0.15, desc="🤖 Улучшение промпта через xAI Grok для фотореализма...")
        try:
            from grok_client import GrokClient
            client = GrokClient(GROK_API_KEY)
            
            # Enhance prompt for ONLYFANS photorealism
            enhanced_pos, _, caption = client.enhance_prompt(
                base_prompt=prompt,
                style_preset="ONLYFANS Photorealistic Selfie",
                model="grok-2-latest"
            )
            
            # Replace original prompt with Grok-enhanced version
            norm_prompt, tokens = prompt_proc.process_prompt(enhanced_pos)
        except Exception as e:
            print(f"[Grok] [WARN] Enhancement failed (using original): {e}")
            norm_prompt, tokens = prompt_proc.process_prompt(prompt)
    else:
        # Извлечение встроенных параметров, если они есть в тексте
        norm_prompt, tokens = prompt_proc.process_prompt(prompt)

    _, inline_params = prompt_proc.extract_parameters(prompt)

    actual_steps = inline_params.get("steps", steps)
    actual_cfg = inline_params.get("cfg_scale", cfg_scale)
    actual_width = inline_params.get("width", width)
    actual_height = inline_params.get("height", height)
    actual_seed = seed if seed >= 0 else inline_params.get("seed", -1)
    actual_batch = max(1, int(batch_size))

    run_meta = _write_ui_run_log(
        engine=engine,
        positive=norm_prompt,
        negative=negative_prompt,
        cfg=float(actual_cfg),
        steps=int(actual_steps),
        seed=int(actual_seed),
        width=int(actual_width),
        height=int(actual_height),
        batch_size=actual_batch,
        profile_name=profile_name,
    )
    badge = _format_run_badge(run_meta)

    # РЕЖИМ 1: Демо-режим (быстрое превью без видеокарты)
    if any(w in engine.lower() for w in ["демо", "demo", "тест"]):
        progress(0.4, desc="Создание тестового демо-арта...")
        images = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx in range(1, actual_batch + 1):
            demo_img = create_demo_artwork(
                prompt=norm_prompt,
                width=int(actual_width),
                height=int(actual_height),
                steps=int(actual_steps),
                cfg=float(actual_cfg),
                seed=int(actual_seed) + idx - 1 if actual_seed >= 0 else -1,
                index=idx
            )
            filename = f"demo_{timestamp}_{idx}.png"
            filepath = RESULTS_DIR / filename
            demo_img.save(filepath, format="PNG")
            images.append(demo_img)

            # Сохранение метаданных
            meta_path = RESULTS_DIR / f"demo_{timestamp}_{idx}_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "engine": "Демо-режим (Тест)",
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
                }, f, indent=2, ensure_ascii=False)

        progress(1.0, desc="Готово!")
        msg = f"✨ **Демо-генерация завершена!** Создано {len(images)} тестовых изображений.\nСохранено в: `{RESULTS_DIR}`" + badge
        return images, msg

    # РЕЖИМ 2: Локальная генерация на видеокарте (PyTorch Diffusers)
    elif any(w in engine.lower() for w in ["локальн", "local", "diffusers"]):
        progress(0.2, desc="Проверка модели в папке models/checkpoints...")
        ckpts = get_valid_checkpoints()
        if not ckpts:
            return [], (
                "⚠️ **Не найдена модель `.safetensors` в папке `models/checkpoints/`!**\n\n"
                "Чтобы запустить генерацию на видеокарте:\n"
                "1. Поместите файл модели (например, Realistic_Vision_V5.1.safetensors) в папку `models/checkpoints/`.\n"
                "2. Либо переключите режим выше на **'🎨 Демо-режим'** для мгновенного теста."
            )

        progress(0.3, desc=f"Загрузка модели {ckpts[0].name} на видеокарту...")
        try:
            from generate import generate_image_diffusers
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            images = []

            for idx in range(1, actual_batch + 1):
                progress(0.3 + (0.6 * (idx / actual_batch)), desc=f"Отрисовка фото {idx} из {actual_batch} на RTX 4070 Ti...")
                out_path = str(RESULTS_DIR / f"diffusers_{timestamp}_{idx}.png")
                current_seed = int(actual_seed) + idx - 1 if actual_seed >= 0 else None

                generated = generate_image_diffusers(
                    prompt=norm_prompt,
                    output_path=out_path,
                    steps=int(actual_steps),
                    cfg_scale=float(actual_cfg),
                    width=int(actual_width),
                    height=int(actual_height),
                    seed=current_seed,
                    checkpoint_path=str(ckpts[0]),
                    negative_prompt=negative_prompt
                )
                if generated:
                    images.append(Image.open(generated[0]))

            if images:
                progress(1.0, desc="Успешно завершено!")
                msg = f"✅ **Успешно сгенерировано {len(images)} фото на видеокарте RTX 4070 Ti!**\nФайлы сохранены в: `{RESULTS_DIR}`" + badge
                return images, msg
            else:
                return [], "❌ Генерация не вернула изображение. Подробности смотрите в окне консоли."
        except Exception as e:
            return [], f"❌ Ошибка генерации на видеокарте: {e}"

    # РЕЖИМ 3: Внешний Automatic1111 WebUI
    else:
        target_url = (webui_host or "http://127.0.0.1:7860").strip().rstrip("/")
        
        if f":{STUDIO_PORT}" in target_url:
            return [], (
                f"⚠️ **Ошибка настройки:** Адрес `{target_url}` указывает на саму эту студию!\n"
                f"Automatic1111 работает на отдельном порту (обычно `http://127.0.0.1:7860`).\n"
                f"Если у вас нет запущенного Automatic1111, переключите режим выше на **'⚡ Локальная видеокарта'**."
            )

        progress(0.3, desc=f"Подключение к Automatic1111 по адресу {target_url}...")
        client = StableDiffusionWebUI({"webui": {"url": target_url}})

        if not client.is_running():
            return [], (
                f"❌ **Не удалось связаться с сервером Automatic1111 по адресу `{target_url}`!**\n\n"
                f"**Как решить:**\n"
                f"1. Если у вас есть Automatic1111 — запустите его с флагом `--api` (файл webui-user.bat).\n"
                f"2. **Если стороннего Automatic1111 нет — просто выберите выше:**\n"
                f"   👉 **'⚡ Локальная видеокарта (RTX 4070 Ti)'** — студия будет генерировать фото напрямую на вашей видеокарте!"
            )

        progress(0.5, desc="Генерация через Automatic1111 API...")
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
            return [], f"❌ **Ошибка генерации в Automatic1111:**\n{error_report}"

        progress(0.8, desc="Сохранение готовых файлов на диск...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, img in enumerate(images, 1):
            filename = f"webui_{timestamp}_{i}.png"
            filepath = RESULTS_DIR / filename
            img.save(filepath, format="PNG")

        progress(1.0, desc="Готово!")
        info_text = f"✅ **Сгенерировано {len(images)} фото через Automatic1111!**\nСохранено в: `{RESULTS_DIR}`" + badge
        return images, info_text


def build_interface() -> gr.Blocks:
    """Создает дружелюбный русскоязычный веб-интерфейс на базе Gradio."""
    presets = get_available_presets()
    preset_choices = ["(Выберите готовый сюжет или напишите свой)"] + list(presets.keys())

    custom_theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="pink",
        neutral_hue="slate"
    )

    with gr.Blocks(title="NeuronGen Studio v2.2", theme=custom_theme) as demo:
        gr.Markdown(
            """
            # 🎨 NeuronGen Studio v2.2
            ### Создание ультра-реалистичных фотографий на вашей видеокарте RTX 4070 Ti
            """
        )

        with gr.Tabs():
            with gr.TabItem("🖼️ Генерация фото (Студия)"):
                with gr.Row():
                    # ЛЕВАЯ КОЛОНКА: Управление и параметры
                    with gr.Column(scale=5):
                        valid_ckpts = get_valid_checkpoints()
                        default_engine = ENGINE_LOCAL if valid_ckpts else ENGINE_DEMO

                        engine_dropdown = gr.Dropdown(
                            choices=[ENGINE_LOCAL, ENGINE_WEBUI, ENGINE_DEMO],
                            value=default_engine,
                            label="⚡ Режим генерации (Движок)",
                            info="Локальный режим работает прямо на вашей RTX 4070 Ti автономно без сторонних программ."
                        )

                        if valid_ckpts:
                            gr.Markdown(f"🟢 **Модель готова к работе:** `{valid_ckpts[0].name}` на **NVIDIA GeForce RTX 4070 Ti (12GB)**")
                        else:
                            gr.Markdown("💡 *Модель не обнаружена. Используется Демо-режим для мгновенного теста.*")

                        # Секция готовых шаблонов сюжета
                        with gr.Group():
                            preset_dropdown = gr.Dropdown(
                                choices=preset_choices,
                                value=list(presets.keys())[0] if presets else "(Выберите сюжет)",
                                label="📁 Готовые сюжеты (Шаблоны в 1 клик)",
                                info="Выберите готовый сценарий, чтобы сразу заполнить описание идеальным фотореалистичным промптом."
                            )

                            archetype_radio = gr.Radio(
                                choices=[
                                    "✨ Свой образ",
                                    "👱‍♀️ Славянка (Блондинка)",
                                    "👩 Латина (Загар / Брюнетка)",
                                    "👩‍🦰 Скандинавка (Рыжая / Веснушки)",
                                    "👧 Азиатка (Нежный образ)"
                                ],
                                value="✨ Свой образ",
                                label="💃 Типаж внешности девушки (быстрая замена)",
                                info="Кликните на нужный типаж, чтобы мгновенно применить его черты к фото."
                            )

                        profile_radio = gr.Radio(
                            choices=["⚡ Speed", "🌟 Realism / Quality"],
                            value="🌟 Realism / Quality",
                            label="Режим качества",
                            info="Realism: CFG 6.0, 28 steps, без beauty/airbrushed токенов"
                        )

                        # Поля ввода промптов
                        prompt_input = gr.Textbox(
                            label="📝 Что нарисовать (Положительное описание / Промпт)",
                            placeholder="Опишите желаемый образ девушки, позу, одежду, свет и окружение (или выберите готовый сюжет выше)...",
                            lines=4,
                            value=list(presets.values())[0] if presets else ""
                        )

                        # Grok AI Enhancement Toggle (ONLYFANS photorealism optimization)
                        grok_enabled = gr.Checkbox(
                            value=is_grok_configured(),
                            label="🤖 Grok AI: Улучшить промпт для фотореализма",
                            info="Использует xAI Grok для генерации идеальных ONLYFANS описаний и постов"
                        )

                        with gr.Accordion("🛡️ Защита от дефектов (Негативный промпт)", open=False):
                            neg_prompt_input = gr.Textbox(
                                label="Что исключить из кадра",
                                lines=2,
                                value=prompt_proc.default_negative,
                                info="Исключает размытость, пластиковую кожу, лишние пальцы, 3D-графику и водяные знаки."
                            )

                        # Быстрый выбор формата кадра в 1 клик
                        aspect_radio = gr.Radio(
                            choices=[
                                "📱 Портрет для телефона (512x768) — Рекомендуется",
                                "📸 Квадрат для соцсетей (512x512)",
                                "🖥️ Горизонтальный кадр (768x512)",
                                "🌟 Высокое HD (896x1152)"
                            ],
                            value="📱 Портрет для телефона (512x768) — Рекомендуется",
                            label="📐 Формат кадра и разрешение",
                            info="Выберите удобный формат в 1 клик — размеры настроятся автоматически."
                        )

                        # Расширенные настройки
                        with gr.Accordion("⚙️ Тонкие настройки генерации", open=False):
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=10, maximum=60, value=25, step=1,
                                    label="Детализация / Шаги (Steps)",
                                    info="20–25 шагов дают отличное качество и быстроту"
                                )
                                cfg_slider = gr.Slider(
                                    minimum=1.0, maximum=15.0, value=7.0, step=0.5,
                                    label="Сила следования описанию (CFG Scale)",
                                    info="7.0 — стандарт естественного реализма"
                                )
                            with gr.Row():
                                width_dropdown = gr.Dropdown(
                                    choices=[512, 768, 896, 1024],
                                    value=512,
                                    label="Ширина (пиксели)"
                                )
                                height_dropdown = gr.Dropdown(
                                    choices=[512, 768, 896, 1024, 1152],
                                    value=768,
                                    label="Высота (пиксели)"
                                )
                            with gr.Row():
                                batch_slider = gr.Slider(
                                    minimum=1, maximum=4, value=1, step=1,
                                    label="Сколько фото создать за раз (Batch Size)"
                                )
                                seed_input = gr.Number(
                                    value=-1,
                                    label="Зерно случайности (Seed)",
                                    info="-1 для нового уникального кадра каждый раз",
                                    precision=0
                                )
                            with gr.Row():
                                sampler_dropdown = gr.Dropdown(
                                    choices=[
                                        "DPM++ 2M Karras",
                                        "Euler a",
                                        "Euler",
                                        "DPM++ SDE Karras",
                                        "DDIM"
                                    ],
                                    value="DPM++ 2M Karras",
                                    label="Алгоритм прорисовки (Sampler)"
                                )
                                webui_host_input = gr.Textbox(
                                    label="Адрес сервера Automatic1111 (если выбран режим сервера)",
                                    value="http://127.0.0.1:7860"
                                )

                        # Большая кнопка запуска
                        generate_btn = gr.Button("🚀 СГЕНЕРИРОВАТЬ ФОТО", variant="primary", size="lg")

                        with gr.Row():
                            random_seed_btn = gr.Button("🎲 Случайный кадр (Seed = -1)", variant="secondary")
                            clear_prompt_btn = gr.Button("🗑️ Очистить описание", variant="secondary")

                    # ПРАВАЯ КОЛОНКА: Результаты и дополнительные инструменты
                    with gr.Column(scale=6):
                        output_gallery = gr.Gallery(
                            label="🖼️ Готовые фотографии",
                            show_label=True,
                            elem_id="gallery",
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height="auto"
                        )
                        output_status = gr.Markdown("Готов к созданию фото. Выберите сюжет и нажмите **'🚀 Сгенерировать фото'**.")
                        
                        open_folder_btn = gr.Button("📂 Открыть папку с готовыми фото в проводнике Windows", variant="secondary")
                        folder_status_msg = gr.Markdown("")

                        # Дополнительный ИИ-помощник Grok
                        with gr.Accordion("✨ ИИ-помощник Grok (Идеи, описания и подписи к постам)", open=False):
                            gr.Markdown("Генерация детализированных описаний внешности, одежды, света и готовых вирусных подписей к постам с помощью Grok (xAI).")
                            grok_key_input = gr.Textbox(
                                label="Ключ xAI Grok API",
                                placeholder="xai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                                type="password",
                                value=os.environ.get("GROK_API_KEY", "") or os.environ.get("XAI_API_KEY", "")
                            )
                            with gr.Row():
                                grok_style_dropdown = gr.Dropdown(
                                    choices=[
                                        "📱 Селфи в спальне / зеркале (Непринуждённое и интимное)",
                                        "🔥 Соблазнительный будуар / Шёлк и кружево",
                                        "💪 Фитнес / Спортивные леггинсы и зал",
                                        "🏖️ Пляж / Бикини на закате",
                                        "🏙️ Пентхаус / Вечерний гламур",
                                        "🚿 Душ / Влажные волосы и капли воды"
                                    ],
                                    value="📱 Селфи в спальне / зеркале (Непринуждённое и интимное)",
                                    label="🎭 Стиль и атмосфера"
                                )
                                grok_concept_input = gr.Textbox(
                                    label="Ваши пожелания (возраст, внешность, одежда)",
                                    placeholder="например: 21 год, блондинка, голубые глаза, белая майка, легкая улыбка",
                                    lines=1
                                )
                            grok_gen_btn = gr.Button("🔮 Сгенерировать промпт и текст для поста", variant="secondary")
                            grok_caption_box = gr.Textbox(
                                label="📱 Готовый текст для публикации / поста (скопируйте в 1 клик):",
                                lines=2,
                                interactive=True
                            )

                        # Импортер из Discord / Midjourney
                        with gr.Accordion("💬 Импорт промптов из Discord / Midjourney", open=False):
                            gr.Markdown("Скопируйте команду прямо из вашего Discord-сервера ([Сервер plaksavaksa](https://discord.gg/j8mjujwcG)). Студия сама удалит команды `/imagine`, флаги `--ar`, `--v` и настроит пропорции кадра!")
                            discord_prompt_box = gr.Textbox(
                                label="Вставьте команду из Discord сюда:",
                                placeholder="/imagine prompt: 1girl, bedroom selfie, silk camisole --ar 9:16 --v 6.0 --no ugly, deformed",
                                lines=2
                            )
                            discord_import_btn = gr.Button("📥 Распознать и применить", variant="secondary")
                            discord_status_msg = gr.Markdown("")

            with gr.TabItem("📊 Пакетная генерация (Серия фото)"):
                gr.Markdown(
                    """
                    ### 📸 Создание серии разных фотографий по списку
                    Вставьте несколько описаний (каждое с новой строки). Студия создаст каждое фото по очереди и сохранит их в папку с результатами.
                    """
                )
                batch_engine = gr.Dropdown(
                    choices=[ENGINE_LOCAL if valid_ckpts else ENGINE_DEMO, ENGINE_DEMO, ENGINE_WEBUI],
                    value=ENGINE_LOCAL if valid_ckpts else ENGINE_DEMO,
                    label="Движок пакетной генерации"
                )
                batch_prompts_input = gr.Textbox(
                    label="Список описаний (одно описание на строке)",
                    lines=8,
                    placeholder="Описание 1...\nОписание 2...\nОписание 3...",
                    value="candid iPhone selfie photograph of a 21yo slavic blonde woman, silk white shirt, bedroom\ncandid photograph of a 22yo brunette woman in fitness spandex, modern gym\ncandid photograph of a 20yo woman in metallic bikini by the infinity pool, sunset"
                )
                with gr.Row():
                    run_batch_btn = gr.Button("⚡ Запустить генерацию всей серии", variant="primary")
                    open_batch_folder_btn = gr.Button("📂 Открыть папку с результатами", variant="secondary")
                batch_status_md = gr.Markdown("Статус серии: Ожидание запуска.")

            with gr.TabItem("🔍 Статус системы и видеокарты"):
                status_display = gr.Markdown(check_system_status())
                refresh_status_btn = gr.Button("🔄 Обновить данные диагностики", variant="secondary")

        # --- ОБРАБОТЧИКИ СОБЫТИЙ (Event Handlers) ---

        # Выбор шаблона сюжета
        def on_preset_change(choice):
            if not choice or choice.startswith("("):
                return gr.update()
            return presets.get(choice, "")

        preset_dropdown.change(
            fn=on_preset_change,
            inputs=[preset_dropdown],
            outputs=[prompt_input]
        )

        # Выбор типажа внешности
        def on_archetype_change(arch_choice, current_prompt):
            if arch_choice == "✨ Свой образ" or arch_choice not in ARCHETYPES_RU:
                return current_prompt
            arch_text = ARCHETYPES_RU[arch_choice]
            # Заменяем или добавляем типаж
            words_to_strip = [
                "21yo slavic woman, athletic toned body with visible abs, natural blonde wavy hair cascading over shoulders, piercing blue eyes, cute soft seductive smile, symmetrical face",
                "22yo latina woman, tan smooth skin with sun-kissed glow, voluptuous hourglass figure, dark brown voluminous hair, hazel eyes, alluring confident look, plump natural lips",
                "20yo caucasian woman, natural ginger red hair in loose waves, light green eyes, pale porcelain skin with subtle freckles across nose, slim fit physique",
                "21yo east asian korean beauty, flawless porcelain glass skin, long sleek black hair, brown almond eyes, soft subtle smile, slim fit physique"
            ]
            new_prompt = current_prompt
            for w in words_to_strip:
                if w in new_prompt:
                    new_prompt = new_prompt.replace(w, arch_text)
                    return new_prompt
            # Если нет совпадений, мягко добавляем в начало
            return f"{arch_text}, {current_prompt.strip()}"

        archetype_radio.change(
            fn=on_archetype_change,
            inputs=[archetype_radio, prompt_input],
            outputs=[prompt_input]
        )

        # Переключение формата кадра
        def on_aspect_change(aspect_choice):
            if "512x768" in aspect_choice:
                return 512, 768
            elif "512x512" in aspect_choice:
                return 512, 512
            elif "768x512" in aspect_choice:
                return 768, 512
            elif "896x1152" in aspect_choice:
                return 896, 1152
            return 512, 768

        aspect_radio.change(
            fn=on_aspect_change,
            inputs=[aspect_radio],
            outputs=[width_dropdown, height_dropdown]
        )

        # Кнопки быстрого управления
        random_seed_btn.click(
            fn=lambda: -1,
            inputs=[],
            outputs=[seed_input]
        )

        clear_prompt_btn.click(
            fn=lambda: "",
            inputs=[],
            outputs=[prompt_input]
        )

        open_folder_btn.click(
            fn=open_results_folder,
            inputs=[],
            outputs=[folder_status_msg]
        )

        open_batch_folder_btn.click(
            fn=open_results_folder,
            inputs=[],
            outputs=[batch_status_md]
        )

        # Интеграция с Grok AI
        def on_grok_enhance(api_key, concept, style):
            if not concept:
                concept = "красивая девушка, нежный взгляд, живое селфи на смартфон"
            client = GrokClient(api_key=api_key)
            pos, neg, caption = client.enhance_prompt(base_prompt=concept, style_preset=style)
            return pos, neg, caption

        grok_gen_btn.click(
            fn=on_grok_enhance,
            inputs=[grok_key_input, grok_concept_input, grok_style_dropdown],
            outputs=[prompt_input, neg_prompt_input, grok_caption_box]
        )

        # Импорт из Discord / Midjourney
        def on_discord_import(raw_text):
            if not raw_text.strip():
                return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Пожалуйста, вставьте текст промпта из Discord."
            res = prompt_proc.parse_discord_or_midjourney_prompt(raw_text)
            flag_str = f"✅ Успешно импортировано! Распознаны флаги: {', '.join(res['flags_detected'])}" if res['flags_detected'] else "✅ Команда из Discord успешно обработана!"
            return res["positive_prompt"], res["negative_prompt"], res["width"], res["height"], flag_str

        discord_import_btn.click(
            fn=on_discord_import,
            inputs=[discord_prompt_box],
            outputs=[prompt_input, neg_prompt_input, width_dropdown, height_dropdown, discord_status_msg]
        )

        # Запуск основной генерации
        
        def on_profile_change(choice: str):
            # get_profile already maps emoji Gradio labels -> speed/realism
            prof = get_profile(choice or "realism")
            return (
                float(prof.get("cfg_scale", 6.0)),
                int(prof.get("steps", 28)),
                str(prof.get("sampler", "DPM++ 2M Karras")),
            )

        profile_radio.change(
            fn=on_profile_change,
            inputs=[profile_radio],
            outputs=[cfg_slider, steps_slider, sampler_dropdown],
        )
        demo.load(
            fn=lambda: on_profile_change("Realism / Quality"),
            inputs=None,
            outputs=[cfg_slider, steps_slider, sampler_dropdown],
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
                webui_host_input,
                profile_radio
            ],
            outputs=[output_gallery, output_status]
        )

        # Запуск пакетной генерации
        def on_batch_run(engine_choice, prompts_text, webui_host):
            lines = [l.strip() for l in prompts_text.splitlines() if l.strip()]
            if not lines:
                return "⚠️ Пожалуйста, укажите хотя бы одно описание."
            
            if any(w in engine_choice.lower() for w in ["демо", "demo", "тест"]):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for i, p in enumerate(lines, 1):
                    img = create_demo_artwork(p, 512, 768, 25, 7.5, -1, i)
                    img.save(RESULTS_DIR / f"batch_demo_{timestamp}_{i}.png")
                return f"✅ **Серия завершена!** Создано {len(lines)} изображений в папку `{RESULTS_DIR}`."
            elif any(w in engine_choice.lower() for w in ["локальн", "local", "diffusers"]):
                ckpts = get_valid_checkpoints()
                if not ckpts:
                    return "⚠️ В папке `models/checkpoints/` нет подходящей модели .safetensors."
                from generate import generate_image_diffusers
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                created = 0
                for i, p in enumerate(lines, 1):
                    out_path = str(RESULTS_DIR / f"batch_local_{timestamp}_{i}.png")
                    res = generate_image_diffusers(
                        prompt=p,
                        output_path=out_path,
                        steps=25,
                        cfg_scale=7.5,
                        width=512,
                        height=768,
                        seed=-1,
                        checkpoint_path=str(ckpts[0]),
                        negative_prompt=prompt_proc.default_negative
                    )
                    if res:
                        created += 1
                return f"✅ **Серия завершена!** Сгенерировано {created} фото на видеокарте RTX 4070 Ti в `{RESULTS_DIR}`."
            else:
                client = StableDiffusionWebUI({"webui": {"url": webui_host or "http://127.0.0.1:7860"}})
                if not client.is_running():
                    return f"❌ Automatic1111 недоступен по адресу {client.base_url}. Выберите 'Локальная видеокарта' или запустите Automatic1111."
                b_gen = BatchGenerator({"webui": {"url": webui_host or "http://127.0.0.1:7860"}})
                files = b_gen.generate_batch(lines)
                return f"✅ **Серия завершена!** Сохранено {len(files)} фото в папку `{RESULTS_DIR}`."

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
    # Поиск свободного порта
    port = STUDIO_PORT
    import socket
    while port < 7900:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        in_use = sock.connect_ex(('127.0.0.1', port)) == 0
        sock.close()
        if not in_use:
            break
        port += 1

    print(f"\n========================================================")
    print(f"🎨 NeuronGen Studio v2.2 запущена!")
    print(f"🌐 Откройте в браузере: http://127.0.0.1:{port}")
    print(f"========================================================\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        show_error=True,
        show_api=False
    )


if __name__ == "__main__":
    main()
