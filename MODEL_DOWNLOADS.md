# 📦 REQUIRED MODEL DOWNLOADS (ONLYFANS Photorealism)

## 1. Juggernaut XL v9 — КРИТИЧЕСКОЕ ⚡ (~6 GB)
**Используется:** Основная генерация фотореализма  
**Где скачать:** [Civitai](https://civitai.com/models/105374/juggernaut-xl-v90-xl) или HuggingFace

```bash
# Расположение:
models/checkpoints/juggernaut_xl_v9.safetensors
```

### Параметры Juggernaut XL v9 (оптимизировано):
- **Resolution:** 1024×1536 (SDXL native)
- **Steps:** 40 (ultra skin detail)
- **CFG Scale:** 7.5 (photorealism standard)
- **Sampler:** DPM++ SDE Karras

---

## 2. ESRGAN x4plus — ДЛЯ АПСКЕЙЛА (~50 MB)
**Используется:** Апскейл изображений до ONLYFANS качества  
**Где скачать:** [Civitai](https://civitai.com/models/136879/realesrgan-x4plus-anime-6b)

```bash
# Расположение:
models/esrgan/RealESRGAN_x4plus_anime_6B.pth
```

### Настройки ESRGAN (ONLYFANS оптимизация):
- **Scale:** 2.0x or 4.0x
- **Denoise Strength:** 0.5 (natural skin texture)
- **Tile Size:** 4096 (skin detail preservation)

---

## 🚀 QUICK START

1. Загрузите Juggernaut XL v9 в `models/checkpoints/`
2. Запустите Automatic1111 WebUI или используйте локальный режим
3. Вставьте API ключ Grok в переменную окружения:
   ```bash
   set XAI_API_KEY=your_xai_api_key_here
   ```

---

## 📊 EXPECTED PERFORMANCE (RTX 4070 Ti)

| Режим | Время на фото | VRAM Usage |
|-------|---------------|------------|
| Демо-режим | <1с | CPU only |
| Локальная генерация | ~8–12с | ~6.5 GB |
| WebUI API (Juggernaut XL) | ~10–15с | ~7 GB |

---

## 🎯 ONLYFANS OPTIMIZATION CHECKLIST

- [ ] Juggernaut XL v9 загружен и выбран в WebUI
- [ ] ESRGAN модель доступна для апскейла
- [ ] Hires Fix включен по умолчанию (config.py)
- [ ] Grok API ключ настроен для AI enhancement
- [ ] CFG Scale = 7.5, Steps = 40 (photorealism standard)
