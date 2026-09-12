from prompt_processor import dedupe_prompt_tokens
import warnings
"""
High-Performance Model & Pipeline Manager for NeuronGen (v2.3).
Features:
- Singleton in-memory pipeline caching (0.00s subsequent reload latency)
- Dynamic VRAM sensing: Full CUDA resident mode (>4.5GB VRAM) or Smart CPU Offload
- Compel integration for unlimited prompt length (>77 tokens) and accurate attention weights
- 2-Pass High-Res Fix (Native base generation + Latent/Bicubic upscale + Img2Img detail refinement)
- PyTorch 2.6 Tensor Core (TF32) + channels_last memory format + VAE tiling & slicing
- PNG tEXt metadata embedding (A1111/Civitai format)
"""

import os
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from PIL import Image, PngImagePlugin
import torch

BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
ESRGAN_DIR = CHECKPOINTS_DIR.parent / "esrgan"


def get_free_vram_mb() -> float:
    """Returns available GPU VRAM in megabytes."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.mem_get_info()[0] / (1024 * 1024)
    except Exception:
        return 4000.0


class ModelManager:
    """Singleton pipeline manager providing persistent caching and acceleration."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Active pipeline state
        self.current_checkpoint: Optional[str] = None
        self.txt2img_pipe = None
        self._img2img_pipe = None
        self.compel = None
        self.is_sdxl: bool = False
        self.mode: str = "Uninitialized"

        # Apply global PyTorch CUDA performance flags
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def find_default_checkpoint(self) -> Optional[Path]:
        """Find the best available checkpoint file (Juggernaut XL v9 priority)."""
        if not CHECKPOINTS_DIR.exists():
            return None
        
        # Priority: Juggernaut XL v9 > any XL model > large checkpoint
        candidates = []
        for f in sorted(CHECKPOINTS_DIR.glob("*.safetensors")):
            name_lower = f.name.lower()
            if "juggernaut" in name_lower and ("xl" in name_lower or "xl v9" in name_lower):
                candidates.append(f)
            elif "xl" in name_lower and f.stat().st_size > 5 * 1024 * 1024:
                candidates.append(f)

        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)
        
        valid = [f for f in sorted(CHECKPOINTS_DIR.glob("*.safetensors")) 
                 if f.stat().st_size > 100 * 1024 * 1024]
        return valid[0] if valid else None

    def load_model(self, checkpoint_path: Optional[Union[str, Path]] = None, force_reload: bool = False) -> bool:
        """
        Loads and caches checkpoint in VRAM. If already loaded, returns instantly in 0.0s.
        """
        ckpt = Path(checkpoint_path) if checkpoint_path else self.find_default_checkpoint()
        if not ckpt or not ckpt.exists() or ckpt.stat().st_size < 100 * 1024 * 1024:
            print(f"[ModelManager] [ERROR] No valid checkpoint found at: {ckpt}")
            return False

        ckpt_str = str(ckpt.resolve())
        if not force_reload and self.current_checkpoint == ckpt_str and self.txt2img_pipe is not None:
            # Already warm and ready in VRAM
            return True

        print(f"\n[ModelManager] [INFO] Loading checkpoint: {ckpt.name} ({ckpt.stat().st_size / (1024**3):.2f} GB)...")
        t0 = time.perf_counter()

        # Clean prior state
        if self.txt2img_pipe is not None:
            del self.txt2img_pipe
            del self._img2img_pipe
            del self.compel
            self.txt2img_pipe = None
            self._img2img_pipe = None
            self.compel = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.is_sdxl = ckpt.stat().st_size > 4 * 1024 * 1024 * 1024
        free_vram = get_free_vram_mb()
        print(f"[ModelManager] Architecture: {'SDXL' if self.is_sdxl else 'SD 1.5'} | Free VRAM: {free_vram:.0f} MB")

        try:
            from diffusers import DPMSolverMultistepScheduler

            if self.is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                pipe = StableDiffusionXLPipeline.from_single_file(
                    ckpt_str,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
                
                if free_vram > 7000:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*should be kept in float32.*",
                            category=UserWarning,
                            module=r"diffusers(\..*)?",
                        )
                        pipe = pipe.to(self.device)
                    self.mode = "Full GPU Resident (SDXL)"
                else:
                    pipe.enable_model_cpu_offload()
                    self.mode = "Smart Model CPU Offload (SDXL)"

            else:
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_single_file(
                    ckpt_str,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

                if free_vram > 4500:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*should be kept in float32.*",
                            category=UserWarning,
                            module=r"diffusers(\..*)?",
                        )
                        pipe = pipe.to(self.device)
                    if hasattr(pipe, "unet"):
                        pipe.unet.to(memory_format=torch.channels_last)
                    self.mode = "Full GPU Resident (Channels-Last TF32)"
                else:
                    pipe.enable_model_cpu_offload()
                    self.mode = "Smart Model CPU Offload (VRAM Guard)"

            # Enable VAE memory optimizations
            if hasattr(pipe, "vae"):
                if hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()

            # Initialize Compel for long prompts & weights
            try:
                from compel import Compel
                if self.is_sdxl:
                    from compel import ReturnedEmbeddingsType
                    self.compel = Compel(
                        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True],
                        truncate_long_prompts=False
                    )
                else:
                    self.compel = Compel(
                        tokenizer=pipe.tokenizer,
                        text_encoder=pipe.text_encoder,
                        truncate_long_prompts=False
                    )
            except Exception as ce:
                print(f"[ModelManager] [WARN] Compel init note: {ce}")
                self.compel = None

            self.txt2img_pipe = pipe
            self._img2img_pipe = None
            self.current_checkpoint = ckpt_str

            load_time = time.perf_counter() - t0
            print(f"[ModelManager] [OK] Ready in {load_time:.2f}s | Mode: {self.mode}\n")
            return True

        except Exception as e:
            print(f"[ModelManager] [ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def upscale_esrgan(self, image: Image.Image, scale_factor: float = 2.0) -> Image.Image:
        """ESRGAN x4 upscale for ONLYFANS quality (Juggernaut XL v9 output optimization)."""
        if not torch.cuda.is_available():
            print("[ESRGAN] [WARN] Running on CPU — using Lanczos upscale")
            return image.resize(
                (int(image.width * scale_factor), int(image.height * scale_factor)),
                Image.Resampling.LANCZOS
            )

        try:
            from realesrgan import RealESRGANer
            
            esrgan_model_path = ESRGAN_DIR / "RealESRGAN_x4plus_anime_6B.pth"
            
            if not esrgan_model_path.exists():
                print("[ESRGAN] [WARN] Model not found — falling back to Lanczos upscale")
                return image.resize(
                    (int(image.width * scale_factor), int(image.height * scale_factor)),
                    Image.Resampling.LANCZOS
                )

            upsampler = RealESRGANer(
                scale=scale_factor,
                model_path=str(esrgan_model_path),
                tile_size=4096,  # Large tiles for skin detail preservation
                tile_pad=10,    # Overlap to prevent artifacts
                pre_pad=0,       # No padding needed for ONLYFANS portraits
                half=True,       # FP16 inference (faster on RTX 4070 Ti)
            )

            upscaled = upsampler.enhance(
                image,
                outscale=scale_factor,
                denoise_strength=0.5,   # Moderate denoising for natural skin texture
                alpha_upsampler="bicubic",  # Bicubic for smooth scaling
                crop_border=1,           # Crop 1px border to remove artifacts
            )

            return Image.fromarray(upscaled)

        except Exception as e:
            print(f"[ESRGAN] [ERROR] Upscale failed: {e} — using Lanczos fallback")
            return image.resize(
                (int(image.width * scale_factor), int(image.height * scale_factor)),
                Image.Resampling.LANCZOS
            )

    def get_img2img_pipe(self):
        """Lazily initialize Img2Img pipeline sharing components with txt2img."""
        if self._img2img_pipe is not None:
            return self._img2img_pipe
        if self.txt2img_pipe is None:
            return None

        pipe = self.txt2img_pipe
        if self.is_sdxl:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            self._img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
        else:
            from diffusers import StableDiffusionImg2ImgPipeline
            self._img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
        return self._img2img_pipe

    def encode_prompts(
        self,
        prompt: str,
        negative_prompt: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode prompts using Compel for unlimited token lengths and balanced weights.
        Returns (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_embeds).
        """
        prompt = dedupe_prompt_tokens(prompt or "")
        negative_prompt = dedupe_prompt_tokens(negative_prompt or "")
        if self.compel is None:
            return None, None, None, None

        try:
            if self.is_sdxl:
                conditioning, pooled = self.compel(prompt)
                neg_conditioning, neg_pooled = self.compel(negative_prompt)
                [conditioning, neg_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                    [conditioning, neg_conditioning]
                )
                return conditioning, neg_conditioning, pooled, neg_pooled
            else:
                conditioning = self.compel.build_conditioning_tensor(prompt)
                neg_conditioning = self.compel.build_conditioning_tensor(negative_prompt)
                [conditioning, neg_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                    [conditioning, neg_conditioning]
                )
                return conditioning, neg_conditioning, None, None
        except Exception as e:
            print(f"[ModelManager] [WARN] Compel encoding fallback to raw string: {e}")
            return None, None, None, None

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 25,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 768,
        seed: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        hires_fix: bool = False,
        hires_scale: float = 1.5,
        hires_denoise: float = 0.38,
        hires_steps: int = 12
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """
        High-performance generation with caching, Compel embeddings, and optional 2-pass High-Res Fix.
        """
        if not self.load_model(checkpoint_path):
            return [], {"error": "Failed to load checkpoint"}

        neg_prompt = negative_prompt or (
            "(plastic skin, airbrushed:1.3), 3d render, cartoon, anime, illustration, "
            "bad anatomy, bad hands, missing fingers, extra fingers, deformed eyes, "
            "blurry, low quality, worst quality, monochrome, distorted body"
        )

        actual_seed = seed if seed is not None and seed >= 0 else int(time.time() * 1000) % (2**31)
        generator = torch.Generator().manual_seed(actual_seed)

        # Prepare dimensions
        if self.is_sdxl:
            target_w = max(512, (width // 64) * 64)
            target_h = max(512, (height // 64) * 64)
            base_w, base_h = target_w, target_h
        else:
            if hires_fix:
                # Pass 1 at native resolution, Pass 2 at target resolution
                base_w = 512
                base_h = 768
                target_w = int(base_w * hires_scale)
                target_h = int(base_h * hires_scale)
                target_w = (target_w // 8) * 8
                target_h = (target_h // 8) * 8
            else:
                if width > 768 or height > 768:
                    base_w, base_h = 512, 768
                else:
                    base_w = (width // 8) * 8
                    base_h = (height // 8) * 8
                target_w, target_h = base_w, base_h

        # Encode prompts via Compel
        cond, neg_cond, pooled, neg_pooled = self.encode_prompts(prompt, neg_prompt)

        t_start = time.perf_counter()
        print(f"[ModelManager] Generating (Seed: {actual_seed}, Steps: {steps}, CFG: {cfg_scale})...")

        # Pass 1: Base generation
        pipe_kwargs = {
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "width": base_w,
            "height": base_h,
            "generator": generator
        }
        if cond is not None and neg_cond is not None:
            pipe_kwargs["prompt_embeds"] = cond
            pipe_kwargs["negative_prompt_embeds"] = neg_cond
            if self.is_sdxl and pooled is not None:
                pipe_kwargs["pooled_prompt_embeds"] = pooled
                pipe_kwargs["negative_pooled_prompt_embeds"] = neg_pooled
        else:
            pipe_kwargs["prompt"] = prompt
            pipe_kwargs["negative_prompt"] = neg_prompt

        with torch.inference_mode():
            base_res = self.txt2img_pipe(**pipe_kwargs)
        base_img = base_res.images[0]

        final_img = base_img

        # Pass 2: High-Res Fix (if enabled)
        if hires_fix and not self.is_sdxl:
            print(f"[ModelManager] High-Res Fix: Upscaling {base_w}x{base_h} -> {target_w}x{target_h} (Denoise: {hires_denoise})...")
            upscaled = base_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img2img = self.get_img2img_pipe()

            img2img_kwargs = {
                "image": upscaled,
                "num_inference_steps": max(8, hires_steps),
                "strength": hires_denoise,
                "guidance_scale": cfg_scale,
                "generator": generator
            }
            if cond is not None and neg_cond is not None:
                img2img_kwargs["prompt_embeds"] = cond
                img2img_kwargs["negative_prompt_embeds"] = neg_cond
            else:
                img2img_kwargs["prompt"] = prompt
                img2img_kwargs["negative_prompt"] = neg_prompt

            with torch.inference_mode():
                hires_res = img2img(**img2img_kwargs)
            final_img = hires_res.images[0]

        total_sec = time.perf_counter() - t_start
        print(f"[ModelManager] Finished in {total_sec:.2f}s ({final_img.size[0]}x{final_img.size[1]})")

        metadata = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "seed": actual_seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": final_img.size[0],
            "height": final_img.size[1],
            "hires_fix": hires_fix,
            "hires_scale": hires_scale if hires_fix else 1.0,
            "hires_denoise": hires_denoise if hires_fix else 0.0,
            "model": Path(self.current_checkpoint).name if self.current_checkpoint else "unknown",
            "elapsed_sec": round(total_sec, 2),
            "mode": self.mode
        }

        return [final_img], metadata

    @staticmethod
    def save_image_fast_with_metadata(image: Image.Image, output_path: Union[str, Path], metadata: Dict[str, Any]) -> None:
        """
        Saves PNG with embedded A1111/Civitai-compatible metadata and JSON sidecar.
        Fast compression (level 1) minimizes disk write overhead.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        png_info = PngImagePlugin.PngInfo()
        
        # ONLYFANS-optimized metadata string (Juggernaut XL v9 standard)
        model_name = Path(self.current_checkpoint).name if self.current_checkpoint else "unknown"
        meta_str = (
            f"Prompt: {metadata.get('prompt', '')}\n"
            f"Negative prompt: {metadata.get('negative_prompt', '')}\n"
            f"Steps: {metadata.get('steps')}, CFG scale: {metadata.get('cfg_scale')}, "
            f"Seed: {metadata.get('seed')}, Size: {image.size[0]}x{image.size[1]}, Model: Juggernaut XL v9 ({model_name})\n"
        )
        
        png_info.add_text("parameters", meta_str)

        image.save(out_path, format="PNG", pnginfo=png_info, compress_level=1)

        meta_json_path = out_path.parent / f"{out_path.stem}_metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_model_manager() -> ModelManager:
    return ModelManager()
