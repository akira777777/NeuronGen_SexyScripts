# NeuronGen SexyScripts v2.3 — Pre-Flight Verification Checklist

## ✅ Completed & Verified (Dec 15, 2024)

### Core Infrastructure
- [x] **Project Structure:** All modules present and accessible
  - `generate.py` — Main CLI with Grok integration ✓
  - `webui_api.py` — WebUI/ComfyUI API clients ✓
  - `batch_generator.py` — Batch generation orchestrator ✓
  - `config.py` — Centralized JSON config management ✓
  - `prompt_processor.py` — Optimized prompt parsing with LRU cache ✓
  - `grok_client.py` — AI prompt enhancement via xAI Grok ✓
  - `model_manager.py` — Singleton pipeline manager (v2.3) ✓

### Prompt Library (`prompts/`)
- [x] **Best Quality:** `best_quality_prompt.txt` — Universal ultra-realism preset
- [x] **OnlyFans Bedroom Selfie:** `onlyfans_bedroom_selfie.txt` — Silk shirt/lace bralette aesthetic
- [x] **Boudoir Studio:** `onlyfans_boudoir_studio.txt` — Luxury studio portrait with sheer lace bodysuit
- [x] **Mirror Selfie:** `onlyfans_mirror_selfie.txt` — Bathroom mirror, toned midriff emphasis
- [x] **Sexy Variant 1:** `sexy_variant.txt` — Lingerie bedroom with dramatic rim lighting
- [x] **Sexy Variant 2:** `sexy_variant2.txt` — Beach sunset micro bikini scenario

### Test Suite (`tests/`)
- [x] **Photorealistic Generator:** `test_photorealistic.py`
  - 3 character archetypes (Slavic blonde, Latina brunette, Nordic redhead)
  - 3 photographic scenarios (bedroom morning, mirror selfie, bikini poolside)
  - Uses optimized parameters: steps=35, cfg_scale=8.0

### Configuration (`configs/`)
- [x] **generation_config.json:** Optimized for Juggernaut XL v9
  - Steps: 35 (increased from 25 for finer detail)
  - CFG Scale: 8.0 (stronger prompt adherence)
  - Resolution: 896×1152 (HD portrait format)
  - Upscale: Enabled by default

### Documentation
- [x] **README.md:** Extended with new features and CLI examples
- [x] **SESSION_20241215_ENHANCEMENTS.md:** Complete changelog of this session's work

---

## ⚠️ Pending Items (Require User Action)

### 1. Model Downloads
| Model | Size | Command | Status |
|-------|------|---------|--------|
| Juggernaut XL v9 | ~2.1 GB | `python generate.py download-checkpoint` | ⬜ Not downloaded |
| ESRGAN 4x-UltraSharp | ~50 MB | `python generate.py download-esrgan` | ⬜ Not downloaded |

**Note:** Downloads require either:
- Environment variable `HF_TOKEN` (recommended)
- Or Grok API key as fallback (`GROK_API_KEY`)

### 2. External Services
| Service | Purpose | Status |
|---------|---------|--------|
| WebUI Backend | Optional A1111 integration | ⬜ Not running |
| ComfyUI Backend | Optional workflow flexibility | ⬜ Not running |

**To run WebUI:**
```bash
cd /path/to/Auto1111
python main.py --api --nowebui
```

### 3. API Keys (Optional but Recommended)
- **Grok/xAI API Key:** For AI prompt enhancement and caption generation
- **HuggingFace Token:** For authenticated model downloads (faster, no rate limits)

---

## 🧪 Testing Protocol

### Test 1: Syntax Verification
```bash
python3 -m py_compile generate.py webui_api.py batch_generator.py config.py prompt_processor.py grok_client.py
# Expected: No output = success
```

### Test 2: Model Listing
```bash
python generate.py list-models
# Expected: Shows checkpoints, ControlNets, ESRGAN models with status
```

### Test 3: Connection Check
```bash
python generate.py test-connection
# Expected: Shows WebUI/ComfyUI online/offline status
```

### Test 4: Demo Generation (No GPU Required)
```bash
python tests/test_photorealistic.py --engine demo
# Expected: Generates 3 demo images using create_demo_artwork()
```

### Test 5: Full Pipeline with Grok API
```bash
set GROK_API_KEY=your_xai_api_key_here
python generate.py --prompt "prompts/onlyfans_bedroom_selfie.txt" --engine diffusers --steps 20
# Expected: 
#   - Prints "Улучшаю промпт с помощью Grok AI..."
#   - Generates image with enhanced prompt
#   - Saves to results/gen_YYYYMMDD_HHMMSS.png
```

### Test 6: Batch ControlNet Generation
```bash
python generate.py --prompt "prompts/best_quality_prompt.txt" --controlnet_batch poses/*.png
# Expected: 
#   - Loads all pose images from directory
#   - Generates one image per pose file
#   - Each output named with corresponding pose filename
```

---

## 📊 Performance Benchmarks (Expected on RTX 4070 Ti)

| Operation | Estimated Time | Notes |
|-----------|----------------|-------|
| Demo generation (3 images) | <1 second each | CPU-only, no GPU load |
| Single image (512×768, steps=25) | ~15-20 seconds | WebUI API mode |
| Single image (896×1152, steps=35) | ~45-60 seconds | Diffusers local GPU mode |
| Batch ControlNet (10 poses) | ~10-15 minutes total | Sequential processing |
| ESRGAN upscale 4x | ~30-45 seconds per image | Requires model loaded in VRAM |

---

## 🐛 Known Limitations & Workarounds

### 1. HuggingFace Authentication
**Issue:** HF now requires auth even for public models  
**Workaround:** Use `HF_TOKEN` env var or Grok API key as fallback (already implemented)

### 2. WebUI ControlNet Injection
**Issue:** A1111's `alwayson_scripts` parameter may not work on all instances  
**Workaround:** Use ComfyUI for guaranteed ControlNet support, or test with latest A1111 nightly build

### 3. ESRGAN Dependencies
**Issue:** Requires `realesrgan`, `basicsr` packages  
**Workaround:** Falls back to 4x Lanczos resize if inference fails (already implemented)

---

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (optional but recommended)
set HF_TOKEN=your_huggingface_token
set GROK_API_KEY=your_grok_api_key

# 3. Download required models
python generate.py download-checkpoint
python generate.py download-esrgan

# 4. Run test generation with AI prompt enhancement
python tests/test_photorealistic.py --engine diffusers --steps 20

# 5. Generate photorealistic OnlyFans-style image
python generate.py --prompt "prompts/onlyfans_bedroom_selfie.txt" --engine diffusers --steps 35 --cfg 8.0

# 6. Batch generation with ControlNet poses
python generate.py --prompt "prompts/best_quality_prompt.txt" --controlnet_batch poses/*.png

# 7. High-res generation with Hires Fix and upscale
python generate.py --prompt "..." --width 896 --height 1152 --hires_fix --upscale esrgan
```

---

## 📞 Support & Troubleshooting

### Issue: Grok enhancement not activating
**Check:** `GROK_API_KEY` environment variable is set and valid  
**Verify:** Run with verbose output to see API response

### Issue: Model download fails with 401
**Solution:** Obtain HF_TOKEN from https://huggingface.co/settings/tokens (read/write scope)

### Issue: "CUDA not available" error
**Check:** NVIDIA drivers are installed and up-to-date  
**Verify:** Run `nvidia-smi` in terminal to confirm GPU detection

---

**Last Updated:** December 15, 2024  
**Version:** v2.3 Enhanced Edition  
**Status:** Ready for production use (pending model downloads)