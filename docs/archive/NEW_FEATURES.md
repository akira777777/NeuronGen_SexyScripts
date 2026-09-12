# New Features in v2.1

## ControlNet OpenPose Integration

Generate pose-guided images using ControlNet's OpenPose model via WebUI API's `alwayson_scripts` parameter.

**Usage:**
```bash
python generate.py --prompt "..." --controlnet openpose
```

- Automatically loads `models/controlnet/control_v11e_sd15_openpose.pth` if missing
- ControlNet data should be provided separately (not yet implemented in CLI)

## ESRGAN Upscale Pipeline

Automatically upscale generated images 4x using the 4x-UltraSharp model.

**Usage:**
```bash
python generate.py --prompt "..." --upscale esrgan
```

- Saves both original and upscaled versions:
  - `results/image_001.png` (original)
  - `results/image_001_upscaled_1.png` (4x upscaled)
- Requires ESRGAN model at `models/ESRGAN/4x-UltraSharp.onnx`

## CLI Management Commands

### list-models
Display all available models and their paths:
```bash
python generate.py list-models
```

Output format:
```
[Checkpoints]
   [OK] Juggernaut_XL_v9.safetensors
      Path: models\checkpoints\Juggernaut_XL_v9.safetensors

[ControlNets]
   [MISSING] control_v11e_sd15_openpose.pth
```

### download-checkpoint <name>
Download a checkpoint model directly from HuggingFace.

**Supported checkpoints:**
- `juggernaut_xl_v9` → Juggernaut_XL_v9.safetensors
- `realisticvision_v60b3` → RealisticVisionV60B3_v40VAE.safetensors

**Usage:**
```bash
python generate.py download-checkpoint juggernaut_xl_v9
```

### test-connection
Validate WebUI API connectivity:
```bash
python generate.py test-connection
```

## Combined Example

Generate pose-guided, upscaled images with batch mode:
```bash
python generate.py \
    --prompt "prompts/best_quality_prompt.txt" \
    --steps 30 --cfg 8.5 \
    --batch_size 4 \
    --controlnet openpose \
    --upscale esrgan \
    --output results/batch_001/
```

## Notes

- All Unicode symbols in console output have been replaced with ASCII-safe alternatives for Windows compatibility
- ControlNet integration uses WebUI's `alwayson_scripts` parameter structure
- Upscaling currently uses diffusers pipeline as a base; future versions may use real-esrgan directly