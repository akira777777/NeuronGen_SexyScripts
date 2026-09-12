#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuronGen CLI Generator v2.3 (OnlyFans Optimized Edition)
=========================================================
Универсальный генератор изображений для OnlyFans:
  - Автономная локальная генерация на PyTorch (RTX 4070 Ti)
  - Поддержка Automatic1111 WebUI API и ComfyUI API
  - Мгновенный тестовый Демо-режим
  - Дружелюбное интерактивное меню на русском языке для простого пользователя
  - Диагностика оборудования и моделей
  - ONLYFANS ОПТИМИЗАЦИЯ: 9:16 вертикальный формат, batch-генерация, AI-enhanced prompts

Примеры использования в командной строке:
  python generate.py
  python generate.py --prompt "prompts/best_quality_prompt.txt" --engine demo
  python generate.py --prompt "prompts/onlyfans_bedroom_selfie.txt" --engine diffusers
  python generate.py list-models
  python generate.py test-connection
"""

from __future__ import annotations

import argparse
import warnings
import io
import os
import random
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

# Настройка UTF-8 для корректного вывода в Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Базовые пути
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
PROMPTS_DIR = BASE_DIR / "prompts"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Инициализация Grok клиента для AI-улучшения промптов
_GROK_CLIENT = None

def get_grok_client() -> GrokClient:
    """Ленивая инициализация Grok клиента."""
    global _GROK_CLIENT
    if _GROK_CLIENT is None:
        from grok_client import GrokClient
        _GROK_CLIENT = GrokClient()
    return _GROK_CLIENT

# Импорт модулей проекта
from config import get_config
from prompt_processor import PromptProcessor, dedupe_prompt_tokens
from webui_api import StableDiffusionWebUI
from batch_generator import BatchGenerator


def list_models():
    """Отображение доступных моделей чекпоинтов, ControlNet и апскейлеров на русском языке."""
    print("\n" + "=" * 65)
    print(" 📦 ДОСТУПНЫЕ НЕЙРОСЕТЕВЫЕ МОДЕЛИ NEURONGEN")
    print("=" * 65)

    categories = [
        ("Чекпоинты (Модели генерации)", CHECKPOINTS_DIR, ["*.safetensors", "*.ckpt"]),
        ("ControlNet (Позы и управление)", MODELS_DIR / "controlnet", ["*.pth", "*.safetensors"]),
        ("ESRGAN (Апскейлеры увеличения четкости)", MODELS_DIR / "ESRGAN", ["*.onnx", "*.pth"]),
    ]

    for cat_name, cat_dir, patterns in categories:
        rel_path = cat_dir.relative_to(BASE_DIR) if cat_dir.is_relative_to(BASE_DIR) else cat_dir
        print(f"\n[{cat_name}] (Папка: {rel_path})")
        found = []
        if cat_dir.exists():
            for pat in patterns:
                found.extend(cat_dir.glob(pat))

        if not found:
            print("   (Файлы не найдены)")
            continue

        for f in found:
            size_bytes = f.stat().st_size
            if size_bytes < 1024 * 1024:
                status = "[⚠️ ПОВРЕЖДЕН / СЛИШКОМ МАЛ (<1 МБ)]"
            else:
                size_mb = size_bytes / (1024 * 1024)
                status = f"[🟢 ГОТОВ] ({size_mb:.0f} МБ)"
            print(f"   {status} {f.name}")

    print("\n" + "=" * 65 + "\n")


def test_connections(webui_url: str = "http://127.0.0.1:7860", comfy_url: str = "http://127.0.0.1:8188"):
    """Проверка подключения к внешним серверам Automatic1111 и ComfyUI."""
    print("\n[INFO] Проверка доступности внешних серверов генерации...")
    client = StableDiffusionWebUI({"webui": {"url": webui_url}, "comfyui": {"url": comfy_url}})

    # Проверка Automatic1111
    a1111_ok = client.is_running(webui_url)
    print(f"   Automatic1111 WebUI ({webui_url}): {'[🟢 ОНЛАЙН]' if a1111_ok else '[⚪ НЕ ЗАПУЩЕН]'}")

    # Проверка ComfyUI
    comfy_ok = client.is_comfyui_running(comfy_url)
    print(f"   ComfyUI             ({comfy_url}): {'[🟢 ОНЛАЙН]' if comfy_ok else '[⚪ НЕ ЗАПУЩЕН]'}")

    if not a1111_ok and not comfy_ok:
        print("\n[ПОДСКАЗКА] Внешние серверы не запущены, но они и НЕ требуются!")
        print("   - Для локальной генерации на вашей RTX 4070 Ti запустите 'python main.py'")
        print("     или используйте параметр '--engine diffusers'.")
        print("   - Для мгновенного теста используйте '--engine demo'.")
    print()


def upscale_image_esrgan(
    image_path: Path,
    output_dir: Optional[Path] = None,
    model_name: str = "4x-UltraSharp.pth"
) -> Optional[Path]:
    """Upscale image 4x using RealESRGAN if model exists, with high-quality Lanczos fallback."""
    out_dir = output_dir or image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    upscaled_path = out_dir / f"{image_path.stem}_upscaled_1{image_path.suffix}"

    try:
        esrgan_dir = MODELS_DIR / "ESRGAN"
        model_file = esrgan_dir / model_name

        try:
            import torchvision.transforms.functional as functional_tensor
            sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
        except Exception:
            pass

        if model_file.exists():
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                import torch
                import numpy as np

                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                upsampler = RealESRGANer(scale=4, model_path=str(model_file), model=model, device=device)
                img = Image.open(image_path).convert("RGB")
                img_np = np.array(img)[:, :, ::-1]
                output, _ = upsampler.enhance(img_np, outscale=4)
                out_img = Image.fromarray(output[:, :, ::-1])
                out_img.save(upscaled_path, format="PNG", compress_level=1)
                print(f"   [OK] Апскейл сохранён (RealESRGAN): {upscaled_path.name}")
                return upscaled_path
            except Exception as ex:
                print(f"   [WARN] RealESRGAN inference: {ex}. Переключение на Lanczos...")

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        upscaled_img = img.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
        upscaled_img.save(upscaled_path, format="PNG", compress_level=1)
        print(f"   [OK] Апскейл сохранён (4x Lanczos): {upscaled_path.name}")
        return upscaled_path
    except Exception as e:
        print(f"   [ERROR] Ошибка апскейла: {e}")
        return None


_ACTIVE_DIFFUSERS_PIPE = None
_ACTIVE_CKPT_PATH = None
_ACTIVE_DEVICE = None


def encode_prompt_chunks_sd15(pipe, prompt: str, negative_prompt: str, device: str = "cuda") -> Tuple[Any, Any]:
    """
    Разбивает промпт и негативный промпт любой длины на 77-токенные чанки для SD 1.5,
    устраняя ограничение CLIP (77-token truncation) и сохраняя все детали фотореализма.
    """
    import torch
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    def tokenize_to_chunks(text: str) -> List[List[int]]:
        if not text or not text.strip():
            return [[tokenizer.bos_token_id] + [tokenizer.eos_token_id] * 76]

        orig_max = tokenizer.model_max_length
        tokenizer.model_max_length = 999999
        tokens = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
        tokenizer.model_max_length = orig_max
        if not tokens:
            return [[tokenizer.bos_token_id] + [tokenizer.eos_token_id] * 76]

        chunk_size = 75
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i : i + chunk_size]
            full_chunk = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
            if len(full_chunk) < 77:
                full_chunk += [tokenizer.eos_token_id] * (77 - len(full_chunk))
            chunks.append(full_chunk)
        return chunks

    p_chunks = tokenize_to_chunks(prompt)
    n_chunks = tokenize_to_chunks(negative_prompt or "")

    max_chunks = max(len(p_chunks), len(n_chunks))
    empty_chunk = [tokenizer.bos_token_id] + [tokenizer.eos_token_id] * 76

    while len(p_chunks) < max_chunks:
        p_chunks.append(list(empty_chunk))
    while len(n_chunks) < max_chunks:
        n_chunks.append(list(empty_chunk))

    p_tensors = torch.tensor(p_chunks, dtype=torch.long, device=device)
    n_tensors = torch.tensor(n_chunks, dtype=torch.long, device=device)

    with torch.no_grad():
        p_embeds = [text_encoder(chunk.unsqueeze(0))[0] for chunk in p_tensors]
        prompt_embeds = torch.cat(p_embeds, dim=1)

        n_embeds = [text_encoder(chunk.unsqueeze(0))[0] for chunk in n_tensors]
        negative_prompt_embeds = torch.cat(n_embeds, dim=1)

    return prompt_embeds, negative_prompt_embeds


def apply_hires_fix_sd15(
    pipe,
    base_image: Image.Image,
    prompt_embeds: Any,
    negative_prompt_embeds: Any,
    upscale_factor: float = 1.5,
    denoising_strength: float = 0.38,
    steps: int = 20,
    cfg_scale: float = 6.0,
    generator: Optional[Any] = None,
) -> Image.Image:
    """
    Применяет High-Resolution Fix (Hires Fix): увеличивает изображение методом Lanczos
    и выполняет img2img проход с низким денойзом (0.35-0.40) для глубокой прорисовки микропор кожи.
    """
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    orig_w, orig_h = base_image.size
    target_w = int((orig_w * upscale_factor) // 8) * 8
    target_h = int((orig_h * upscale_factor) // 8) * 8

    print(f"   [HIRES FIX] Апскейл {orig_w}x{orig_h} -> {target_w}x{target_h} (Lanczos + Denoise {denoising_strength})...")
    upscaled = base_image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    img2img = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )

    with torch.inference_mode():
        refined = img2img(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=upscaled,
            strength=denoising_strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]

    return refined


def generate_image_diffusers(
    prompt: str,
    output_path: str,
    steps: int = 25,
    cfg_scale: float = 6.5,
    width: int = 512,
    height: int = 768,
    seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    hires_fix: bool = False,
    hires_scale: float = 1.5,
    hires_denoising: float = 0.38,
) -> List[str]:
    """Генерация фото локально через PyTorch Diffusers с поддержкой длинных промптов и Hires Fix."""
    global _ACTIVE_DIFFUSERS_PIPE, _ACTIVE_CKPT_PATH, _ACTIVE_DEVICE
    print(f"\n[INFO] Генерация через PyTorch Diffusers на видеокарте...")
    print(f"   Промпт: \"{prompt[:70]}...\"")
    print(f"   Разрешение: {width}x{height} | Шаги: {steps} | CFG: {cfg_scale}")

    # Оптимизированный негативный промпт под реализм, если не задан
    if not negative_prompt:
        negative_prompt = (
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), "
            "(worst quality, low quality, normal quality:1.3), "
            "(plastic skin, airbrushed, wax figure:1.25), "
            "(extra fingers, deformed hands, fused fingers, mutated limbs, cross-eyed, malformed eyes:1.4), "
            "(monochrome, grayscale, bad anatomy, bad proportions, unnatural body, distorted features:1.2), "
            "text, close up, cropped, out of frame, duplicate, morbid, mutilated, blurry, dehydrated, cloned face, watermark"
        )

    # Проверка наличия подходящего чекпоинта
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if not ckpt or not ckpt.exists() or ckpt.stat().st_size < 100 * 1024 * 1024:
        valid = [p for p in CHECKPOINTS_DIR.glob("*.safetensors") if p.stat().st_size > 100 * 1024 * 1024]
        if valid:
            ckpt = valid[0]
        else:
            print("[ОШИБКА] В папке models/checkpoints/ не найдена модель .safetensors (>100 МБ)!")
            return []

    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        free_vram_mb = 99999.0

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   [ВИДЕОКАРТА] {gpu_name} (Свободно памяти: {free_vram_mb:.0f} МБ)")
            except Exception:
                free_vram_mb = 99999.0

        # Использование кэшированного пайплайна при совпадении модели и устройства
        if _ACTIVE_DIFFUSERS_PIPE is not None and _ACTIVE_CKPT_PATH == str(ckpt) and _ACTIVE_DEVICE == device:
            print(f"   ⚡ Использование уже загруженного в память пайплайна ({ckpt.name})")
            pipe = _ACTIVE_DIFFUSERS_PIPE
        else:
            is_sdxl = ckpt.stat().st_size > 4 * 1024 * 1024 * 1024
            print(f"   Архитектура модели: {'SDXL' if is_sdxl else 'SD 1.5'}")
            print(f"   Загрузка {ckpt.name} на {device}...")

            if is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                pipe = StableDiffusionXLPipeline.from_single_file(
                    str(ckpt),
                    torch_dtype=dtype,
                    safety_checker=None,
                    use_safetensors=True
                )
            else:
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_single_file(
                    str(ckpt),
                    torch_dtype=dtype,
                    safety_checker=None,
                    use_safetensors=True
                )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            if device == "cuda":
                if free_vram_mb < 6000:
                    print(f"   [ОПТИМИЗАЦИЯ ПАМЯТИ] Свободно {free_vram_mb:.0f} МБ VRAM. Включение динамического model CPU offload...")
                    pipe.enable_model_cpu_offload()
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*should be kept in float32.*",
                            category=UserWarning,
                            module=r"diffusers(\..*)?",
                        )
                        pipe = pipe.to(device)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*should be kept in float32.*",
                        category=UserWarning,
                        module=r"diffusers(\..*)?",
                    )
                    pipe = pipe.to(device)

            _ACTIVE_DIFFUSERS_PIPE = pipe
            _ACTIVE_CKPT_PATH = str(ckpt)
            _ACTIVE_DEVICE = device

        is_sdxl = ckpt.stat().st_size > 4 * 1024 * 1024 * 1024

        # Для SD 1.5 родное разрешение 512x768 или 512x512
        if not is_sdxl and (width > 768 or height > 768):
            width, height = 512, 768
            print(f"   [ОПТИМИЗАЦИЯ] Базовое разрешение скорректировано до {width}x{height} для нативного качества SD 1.5")

        generator = torch.Generator(device="cpu" if hasattr(pipe, "_model_cpu_offload_hook") else device)
        if seed is not None and seed >= 0:
            generator.manual_seed(seed)
        else:
            actual_seed = int(time.time() * 1000) % (2**31)
            generator.manual_seed(actual_seed)

        print(f"   Отрисовка {steps} шагов (CFG {cfg_scale})...")

        # Dedupe repeated quality/prompt tags (casefold); keep higher |weight|
        prompt = dedupe_prompt_tokens(prompt or "")
        negative_prompt = dedupe_prompt_tokens(negative_prompt or "")

        # SD 1.5: chunked CLIP embeds (no silent 77-token truncate).
        # SDXL: Compel when available; else native pipeline encode.
        p_emb = n_emb = None
        pipe_kwargs = {
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "width": width,
            "height": height,
            "generator": generator,
        }
        if not is_sdxl:
            enc_device = str(next(pipe.text_encoder.parameters()).device)
            p_emb, n_emb = encode_prompt_chunks_sd15(
                pipe, prompt, negative_prompt or "", device=enc_device
            )
            pipe_kwargs["prompt_embeds"] = p_emb
            pipe_kwargs["negative_prompt_embeds"] = n_emb
            print(f"   [CLIP] chunked embeds shape={tuple(p_emb.shape)} on {enc_device}")
        else:
            used_compel = False
            try:
                from compel import Compel, ReturnedEmbeddingsType
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
                cond, pooled = compel(prompt)
                neg_cond, neg_pooled = compel(negative_prompt or "")
                cond, neg_cond = compel.pad_conditioning_tensors_to_same_length([cond, neg_cond])
                pipe_kwargs["prompt_embeds"] = cond
                pipe_kwargs["negative_prompt_embeds"] = neg_cond
                pipe_kwargs["pooled_prompt_embeds"] = pooled
                pipe_kwargs["negative_pooled_prompt_embeds"] = neg_pooled
                used_compel = True
                print("   [CLIP] SDXL Compel encode (long prompts OK)")
            except Exception as ce:
                print(f"   [CLIP] Compel unavailable ({ce}); raw SDXL strings")
            if not used_compel:
                pipe_kwargs["prompt"] = prompt
                pipe_kwargs["negative_prompt"] = negative_prompt

        try:
            with torch.inference_mode():
                result = pipe(**pipe_kwargs)
        except torch.cuda.OutOfMemoryError as oom:
            print(f"   [ВНИМАНИЕ] Недостаточно видеопамяти ({oom}). Включение sequential CPU offload...")
            torch.cuda.empty_cache()
            pipe.enable_sequential_cpu_offload()
            if not is_sdxl:
                enc_device = str(next(pipe.text_encoder.parameters()).device)
                p_emb, n_emb = encode_prompt_chunks_sd15(
                    pipe, prompt, negative_prompt or "", device=enc_device
                )
                pipe_kwargs["prompt_embeds"] = p_emb
                pipe_kwargs["negative_prompt_embeds"] = n_emb
            with torch.inference_mode():
                result = pipe(**pipe_kwargs)

        image = result.images[0]

        # Опциональный Hires Fix для ультра-детализации кожи
        if hires_fix and not is_sdxl:
            if p_emb is None or n_emb is None:
                enc_device = str(next(pipe.text_encoder.parameters()).device)
                p_emb, n_emb = encode_prompt_chunks_sd15(
                    pipe, prompt, negative_prompt or "", device=enc_device
                )
            image = apply_hires_fix_sd15(
                pipe=pipe,
                base_image=image,
                prompt_embeds=p_emb,
                negative_prompt_embeds=n_emb,
                upscale_factor=hires_scale,
                denoising_strength=hires_denoising,
                steps=max(15, steps // 2),
                cfg_scale=cfg_scale,
                generator=generator
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", compress_level=1)
        print(f"   [ГОТОВО] Сохранено: {output_path} (Разрешение: {image.width}x{image.height})")
        return [output_path]

    except Exception as e:
        print(f"[ОШИБКА] Сбой генерации: {e}")
        import traceback
        traceback.print_exc()
        return []


def _load_controlnet_poses(single_path: Optional[str], batch_glob: Optional[str]) -> List[Tuple[str, Image.Image]]:
    """Загрузка одной или нескольких поз для ControlNet."""
    poses: List[Tuple[str, Image.Image]] = []
    if single_path:
        import glob
        files = sorted(glob.glob(single_path)) or ([single_path] if os.path.exists(single_path) else [])
        if files:
            try:
                img = Image.open(files[0]).convert("RGB")
                poses.append((files[0], img))
                print(f"[INFO] Загружена поза ControlNet: {os.path.basename(files[0])}")
            except Exception as e:
                print(f"[WARN] Ошибка загрузки позы {files[0]}: {e}")
    elif batch_glob:
        import glob
        for pf in sorted(glob.glob(batch_glob)):
            try:
                img = Image.open(pf).convert("RGB")
                poses.append((pf, img))
            except Exception as e:
                print(f"[WARN] Ошибка загрузки позы {pf}: {e}")
        if poses:
            print(f"[INFO] Загружено поз ControlNet: {len(poses)}")
    return poses


def execute_pipeline(
    prompt: str,
    engine: str = "demo",
    steps: int = 25,
    cfg: float = 6.5,
    width: int = 512,
    height: int = 768,
    batch_size: int = 1,
    seed: int = -1,
    sampler: str = "DPM++ 2M Karras",
    output: Optional[str] = None,
    webui_host: str = "http://127.0.0.1:7860",
    comfy_host: str = "http://127.0.0.1:8188",
    controlnet_single: Optional[str] = None,
    controlnet_batch: Optional[str] = None,
    upscale: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    hires_fix: bool = False,
    hires_scale: float = 1.5,
    hires_denoising: float = 0.38,
):
    """Выполняет генерацию по переданным параметрам."""
    engine_clean = engine.lower().strip()
    if engine_clean in ["webui_api", "api"]:
        engine_clean = "webui"
    elif engine_clean in ["local", "gpu", "torch"]:
        engine_clean = "diffusers"

    # Чтение промпта из файла, если передан путь
    prompt_str = prompt
    if os.path.exists(prompt_str):
        with open(prompt_str, "r", encoding="utf-8") as f:
            prompt_str = f.read().strip()
    elif (PROMPTS_DIR / prompt_str).exists():
        with open(PROMPTS_DIR / prompt_str, "r", encoding="utf-8") as f:
            prompt_str = f.read().strip()

    # AI-улучшение промпта через Grok (если API ключ настроен)
    grok_client = get_grok_client()
    if grok_client.is_configured():
        print("   [INFO] Улучшаю промпт с помощью Grok AI...")
        enhanced_positive, enhanced_negative, caption = grok_client.enhance_prompt(
            prompt_str,
            style_preset="OnlyFans Photorealistic Selfie",
            model="grok-2-latest"
        )
        print(f"   [OK] Grok enhancement complete. Caption: {caption[:80]}...")
    else:
        enhanced_positive, enhanced_negative, caption = prompt_str, "(worst quality, low quality:1.3)", ""

    # Grok negative must reach Diffusers/WebUI (was built then discarded)
    if grok_client.is_configured() or not negative_prompt:
        negative_prompt = enhanced_negative

    proc = PromptProcessor()
    norm_prompt, _ = proc.process_prompt(enhanced_positive if grok_client.is_configured() else prompt_str)
    _, inline_params = proc.extract_parameters(prompt_str)

    actual_steps = inline_params.get("steps", steps)
    actual_cfg = inline_params.get("cfg_scale", cfg)
    actual_width = inline_params.get("width", width)
    actual_height = inline_params.get("height", height)
    actual_seed = inline_params.get("seed", seed)

    controlnet_poses = _load_controlnet_poses(controlnet_single, controlnet_batch)

    run_id = uuid.uuid4().hex[:12]
    frame_total = max(batch_size, len(controlnet_poses) if controlnet_poses else batch_size)
    run_log = {
        "run_id": run_id,
        "engine": engine_clean,
        "positive_prompt": norm_prompt,
        "negative_prompt": negative_prompt,
        "cfg": actual_cfg,
        "steps": actual_steps,
        "seed": actual_seed,
        "width": actual_width,
        "height": actual_height,
        "batch_size": batch_size,
        "frame_total": frame_total,
        "hires_fix": hires_fix,
        "timestamp": datetime.now().isoformat(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_log_path = RESULTS_DIR / f"run_{run_id}.json"
    with open(run_log_path, "w", encoding="utf-8") as rf:
        json.dump(run_log, rf, indent=2, ensure_ascii=False)
    print(f"   [RUN] id={run_id} log={run_log_path.name}")
    print(f"   [RUN] CFG={actual_cfg} seed={actual_seed} engine={engine_clean}")
    print(f"   [RUN] neg[:80]={str(negative_prompt)[:80]!r}")

    print("\n" + "=" * 65)
    print("  🚀 ПАЙПЛАЙН ГЕНЕРАЦИИ NEURONGEN")
    print("=" * 65)
    print(f"Режим: {engine_clean.upper()}")
    print(f"Промпт: \"{norm_prompt[:70]}...\"")
    print(f"Параметры: {actual_width}x{actual_height} | Шаги: {actual_steps} | CFG: {actual_cfg} | Seed: {actual_seed}")
    if hires_fix:
        print(f"Hires Fix: активен (Scale: {hires_scale}x, Denoising: {hires_denoising})")
    if controlnet_poses:
        print(f"ControlNet: активен ({len(controlnet_poses)} поз)")
    print("=" * 65)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output or str(RESULTS_DIR / f"gen_{timestamp}.png")
    b_gen = BatchGenerator()
    saved = []

    # Режим ДЕМО
    if engine_clean == "demo":
        from demo_renderer import create_demo_artwork
        if controlnet_poses:
            for i, (p_path, p_img) in enumerate(controlnet_poses, 1):
                p_name = os.path.basename(p_path) if p_path else f"pose_{i}"
                img = create_demo_artwork(
                    prompt=f"{norm_prompt} [ControlNet: {p_name}]",
                    width=actual_width,
                    height=actual_height,
                    steps=actual_steps,
                    cfg=actual_cfg,
                    seed=actual_seed + i if actual_seed >= 0 else -1,
                    index=i
                )
                out_file = output_path.replace(".png", f"_{i}.png") if len(controlnet_poses) > 1 else output_path
                b_gen._save_image_fast(img, Path(out_file))
                saved.append(out_file)
                print(f"   [ГОТОВО] Сохранено (ControlNet {p_name}): {out_file}")
        else:
            for i in range(1, batch_size + 1):
                img = create_demo_artwork(
                    prompt=norm_prompt,
                    width=actual_width,
                    height=actual_height,
                    steps=actual_steps,
                    cfg=actual_cfg,
                    seed=actual_seed + i if actual_seed >= 0 else -1,
                    index=i
                )
                out_file = output_path.replace(".png", f"_{i}.png") if batch_size > 1 else output_path
                b_gen._save_image_fast(img, Path(out_file))
                saved.append(out_file)
                print(f"   [ГОТОВО] Сохранено: {out_file}")

    # Режим ЛОКАЛЬНОГО DIFFUSERS (видеокарта)
    elif engine_clean == "diffusers":
        for i in range(1, batch_size + 1):
            out_file = output_path.replace(".png", f"_{i}.png") if batch_size > 1 else output_path
            current_seed = actual_seed + i - 1 if actual_seed >= 0 else None
            gen_res = generate_image_diffusers(
                prompt=norm_prompt,
                output_path=out_file,
                steps=actual_steps,
                cfg_scale=actual_cfg,
                width=actual_width,
                height=actual_height,
                seed=current_seed,
                negative_prompt=negative_prompt,
                hires_fix=hires_fix,
                hires_scale=hires_scale,
                hires_denoising=hires_denoising
            )
            if gen_res:
                saved.extend(gen_res)

    # Режим AUTOMATIC1111 / COMFYUI
    else:
        client = StableDiffusionWebUI({"webui": {"url": webui_host}, "comfyui": {"url": comfy_host}})
        if not client.is_running(webui_host):
            print(f"[ОШИБКА] Automatic1111 недоступен по адресу {webui_host}")
            print("   Запустите Automatic1111 с флагом --api или используйте '--engine diffusers'.")
            return

        imgs, info, errs = client.generate_images(
            positive_prompt=norm_prompt,
            negative_prompt=negative_prompt,
            steps=actual_steps,
            cfg_scale=actual_cfg,
            width=actual_width,
            height=actual_height,
            batch_size=batch_size,
            seed=actual_seed,
            sampler_name=sampler,
            controlnet_poses=controlnet_poses if controlnet_poses else None
        )
        if errs:
            print(f"[ОШИБКА] Ошибка генерации: {', '.join(errs)}")
            return

        for i, img in enumerate(imgs, 1):
            out_file = output_path.replace(".png", f"_{i}.png") if len(imgs) > 1 else output_path
            b_gen._save_image_fast(img, Path(out_file))
            saved.append(out_file)
            print(f"   [ГОТОВО] Сохранено: {out_file}")

    # Сохранение метаданных и апскейл
    if saved:
        # Проверяем фактический размер первого сгенерированного файла
        actual_img_w, actual_img_h = actual_width, actual_height
        try:
            with Image.open(saved[0]) as first_img:
                actual_img_w, actual_img_h = first_img.size
        except Exception:
            pass

        meta_data = {
            "run_id": run_id,
            "engine": engine_clean,
            "timestamp": datetime.now().isoformat(),
            "prompt": norm_prompt,
            "negative_prompt": negative_prompt,
            "frames": [{"index": i, "total": len(saved), "path": p} for i, p in enumerate(saved, 1)],
            "parameters": {
                "steps": actual_steps,
                "cfg": actual_cfg,
                "width": actual_img_w,
                "height": actual_img_h,
                "seed": actual_seed,
                "batch_size": batch_size,
                "hires_fix": hires_fix
            },
            "output_files": saved
        }
        if controlnet_poses:
            meta_data["controlnet"] = {
                "mode": "single" if len(controlnet_poses) == 1 else "batch",
                "poses": [os.path.basename(p[0]) for p in controlnet_poses if p[0]]
            }

        if upscale and upscale.lower() in ["esrgan", "true"]:
            print("\n[INFO] Выполняется апскейл 4x (ESRGAN)...")
            upscaled_files = []
            for s_file in saved:
                upscaled_path = upscale_image_esrgan(Path(s_file))
                if upscaled_path:
                    upscaled_files.append(str(upscaled_path))
            if upscaled_files:
                meta_data["upscaled_files"] = upscaled_files

        meta_file = output_path.replace(".png", "_metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        print(f"\n✨ [УСПЕХ] Создано изображений: {len(saved)}.\n📂 Файлы сохранены в: {RESULTS_DIR}")


def interactive_russian_menu():
    """Интерактивное консольное меню для простого пользователя при запуске без параметров."""
    while True:
        print("\n" + "=" * 65)
        print("  🎨 NeuronGen Studio v2.2 — Меню управления")
        print("=" * 65)
        print("  [1] 🚀 Запустить Веб-интерфейс в браузере (Рекомендуется)")
        print("  [2] ⚡ Быстрая генерация по готовому сюжету (Селфи, Пляж, Будуар...)")
        print("  [3] ✍️ Ввести свой текст для генерации фото")
        print("  [4] 🔍 Проверить статус видеокарты, моделей и серверов")
        print("  [5] 📂 Открыть папку с готовыми картинками (results)")
        print("  [0] Выход")
        print("=" * 65)

        try:
            choice = input("Выберите действие [0-5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход из программы.")
            break

        if choice == "1":
            print("\n[INFO] Запуск веб-интерфейса NeuronGen Studio...")
            import subprocess
            subprocess.Popen([sys.executable, str(BASE_DIR / "main.py")])
            time.sleep(2)
            try:
                import webbrowser
                webbrowser.open("http://127.0.0.1:7861")
            except Exception:
                pass
            print("Веб-интерфейс запущен по адресу: http://127.0.0.1:7861")
            break

        elif choice == "2":
            presets = [
                ("onlyfans_bedroom_selfie.txt", "📱 Селфи в спальне (Шёлковая рубашка, утренний свет)"),
                ("onlyfans_mirror_selfie.txt", "🪞 Селфи у зеркала в ванной (Спортивный топ, мрамор)"),
                ("onlyfans_boudoir_studio.txt", "🔥 Чувственный будуар (Кружевное боди, киносвет)"),
                ("juggernaut_poolside_bikini.txt", "🏖️ Пляж и бассейн (Бикини, тропический океан)"),
                ("juggernaut_gym_fitness.txt", "💪 Фитнес в спортзале (Леггинсы, спортивная фигура)"),
                ("best_quality_prompt.txt", "💎 Шедевр качества (Универсальный ультра-фотореализм)"),
            ]
            print("\nВыберите готовый сюжет:")
            for i, (fn, desc) in enumerate(presets, 1):
                print(f"  [{i}] {desc}")
            try:
                p_idx = int(input("\nНомер сюжета [1-6]: ").strip()) - 1
                if 0 <= p_idx < len(presets):
                    chosen_file = presets[p_idx][0]
                    ckpts = [f for f in CHECKPOINTS_DIR.glob("*.safetensors") if f.stat().st_size > 100 * 1024 * 1024]
                    engine_choice = "diffusers" if ckpts else "demo"
                    print(f"\nЗапуск генерации сюжета: {presets[p_idx][1]}...")
                    execute_pipeline(prompt=chosen_file, engine=engine_choice, width=512, height=768)
            except Exception as e:
                print(f"[ОШИБКА] Некорректный выбор: {e}")

        elif choice == "3":
            try:
                user_p = input("\nОпишите желаемый кадр (на английском): ").strip()
                if user_p:
                    ckpts = [f for f in CHECKPOINTS_DIR.glob("*.safetensors") if f.stat().st_size > 100 * 1024 * 1024]
                    engine_choice = "diffusers" if ckpts else "demo"
                    execute_pipeline(prompt=user_p, engine=engine_choice, width=512, height=768)
            except Exception as e:
                print(f"[ОШИБКА] {e}")

        elif choice == "4":
            list_models()
            test_connections()

        elif choice == "5":
            if hasattr(os, "startfile"):
                os.startfile(str(RESULTS_DIR.resolve()))
                print(f"Папка {RESULTS_DIR} открыта в проводнике Windows.")
            else:
                print(f"Путь к папке: {RESULTS_DIR.resolve()}")

        elif choice == "0":
            print("Всего доброго!")
            break
        else:
            print("Неверный выбор. Пожалуйста, введите цифру от 0 до 5.")


def main():
    # Если запуск без параметров — открываем интерактивное меню
    if len(sys.argv) == 1:
        interactive_russian_menu()
        return

    parser = argparse.ArgumentParser(description="NeuronGen CLI Generator v2.2")

    # Основные параметры
    parser.add_argument("--prompt", type=str, default=None,
                        help="Текст описания (промпт) или путь к файлу (например, prompts/best_quality_prompt.txt)")
    parser.add_argument("--negative", "--negative_prompt", dest="negative_prompt", type=str, default=None,
                        help="Негативный промпт (исключение дефектов, пластика, CGI, лишних пальцев)")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения картинки (по умолчанию: results/gen_<timestamp>.png)")
    parser.add_argument("--engine", "--method", dest="engine", type=str,
                        choices=["webui", "webui_api", "comfyui", "diffusers", "local", "demo"],
                        default="demo",
                        help="Режим генерации: diffusers (локально на GPU), webui, comfyui, demo (по умолчанию: demo)")
    
    # Параметры генерации
    # Juggernaut XL v9 оптимизация для фотореализма
    parser.add_argument("--steps", type=int, default=35, help="Количество шагов (Juggernaut XL: 35 для ультра-детализации кожи)")
    parser.add_argument("--cfg", type=float, default=8.0, help="Сила соответствия тексту CFG (Juggernaut XL: 8.0 для баланса реализма и креативности)")
    # ONLYFANS ОПТИМИЗАЦИЯ: 9:16 вертикальный формат для мобильных устройств
    parser.add_argument("--width", type=int, default=1080, help="Ширина изображения (ONLYFANS оптимизация: 1080px по умолчанию)")
    parser.add_argument("--height", type=int, default=1920, help="Высота изображения (ONLYFANS оптимизация: 1920px для формата 9:16)")
    # Batch-генерация для OnlyFans: сразу 5-10 фото за один запрос
    parser.add_argument("--batch_size", type=int, default=5, help="Количество картинок в одном запросе (ONLYFANS оптимизация: 5-10 по умолчанию)")
    parser.add_argument("--seed", type=int, default=-1, help="Зерно случайности (-1 для случайного)")
    # DPM++ SDE Karras — лучший для фотореализма и минимума артефактов
    parser.add_argument("--sampler", type=str, default="DPM++ SDE Karras", help="Название сэмплера (рекомендуется: DPM++ SDE Karras)")
    parser.add_argument("--hires_fix", "--hires", dest="hires_fix", action="store_true", default=False,
                        help="Режим Hires Fix: двухэтапная прорисовка микротекстуры кожи без искажений")
    parser.add_argument("--hires_scale", type=float, default=1.5,
                        help="Коэффициент увеличения Hires Fix (по умолчанию: 1.5)")
    parser.add_argument("--hires_denoising", type=float, default=0.38,
                        help="Сила денойза Hires Fix (по умолчанию: 0.38)")
    # ESRGAN апскейл для HD OnlyFans постов
    parser.add_argument("--upscale", type=str, choices=["esrgan_2x", "esrgan_4x", "none"], default="esrgan_2x",
                        help="Апскейл для OnlyFans HD (рекомендуется: esrgan_2x для соцсетей)")

    # Сетевые адреса
    parser.add_argument("--webui_host", type=str, default="http://127.0.0.1:7860", help="Адрес Automatic1111")
    parser.add_argument("--comfy_host", type=str, default="http://127.0.0.1:8188", help="Адрес ComfyUI")

    # ControlNet параметры
    parser.add_argument("--controlnet_single", type=str, default=None,
                        help="Путь к файлу позы для ControlNet")
    parser.add_argument("--controlnet_batch", type=str, default=None,
                        help="Маска файлов поз для пакетной генерации (например, poses/*.png)")

    # Подкоманды
    subparsers = parser.add_subparsers(dest="command", help="Команды управления")
    subparsers.add_parser("list-models", help="Список доступных моделей и их статус")
    subparsers.add_parser("test-connection", help="Проверка доступности серверов")
    subparsers.add_parser("download-checkpoint", help="Скачать чекпоинт Juggernaut XL v9")
    subparsers.add_parser("download-esrgan", help="Скачать ESRGAN модель для апскейла")

    args = parser.parse_args()

    if args.command == "list-models":
        list_models()
        return

    if args.command == "test-connection":
        test_connections(args.webui_host, args.comfy_host)
        return

    if not args.prompt:
        interactive_russian_menu()
        return

    execute_pipeline(
        prompt=args.prompt,
        engine=args.engine,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
        seed=args.seed,
        sampler=args.sampler,
        output=args.output,
        webui_host=args.webui_host,
        comfy_host=args.comfy_host,
        controlnet_single=args.controlnet_single,
        controlnet_batch=args.controlnet_batch,
        upscale=args.upscale,
        negative_prompt=args.negative_prompt,
        hires_fix=args.hires_fix,
        hires_scale=args.hires_scale,
        hires_denoising=args.hires_denoising
    )


if __name__ == "__main__":
    main()