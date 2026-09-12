# 🎨 NeuronGen Studio v2.3 (Photorealistic OnlyFans Edition)

Высокопроизводительный комплекс для генерации **фотореалистичных изображений** в стиле OnlyFans через **Automatic1111 WebUI**, **ComfyUI**, **PyTorch Diffusers** и встроенный **Demo Studio**.

---

## 📦 Структура проекта

```
NeuronGen_SexyScripts/
├── main.py                  # Веб-интерфейс NeuronGen Studio (http://localhost:7861)
├── generate.py              # Универсальный высокоскоростной CLI-генератор v2.3
├── batch_generator.py       # Пакетный генератор с параллельным пулом и быстрым I/O
├── webui_api.py             # Клиент с HTTP Keep-Alive пулом для A1111 и ComfyUI
├── prompt_processor.py      # LRU-кэшированный парсер промптов (>120k промптов/сек)
├── config.py                # Менеджер настроек (JSON)
├── requirements.txt         # Зависимости Python
├── configs/
│   └── generation_config.json # Конфигурация генерации и адреса серверов
├── prompts/                 # Готовые качественные шаблоны промптов для OnlyFans стиля
├── models/                  # Чекпоинты, ControlNet и апскейлеры
├── results/                 # Результаты генераций и сопутствующие метаданные
└── tests/                   # Набор автоматических тестов и бенчмарков
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd C:/Users/novra/Desktop/NeuronGen_SexyScripts
pip install -r requirements.txt
```

### 2. Запуск веб-студии (Рекомендуется)

```bash
python main.py
```
Откройте в браузере: **http://127.0.0.1:7861**

В интерфейсе доступны:
- **⚡ Выбор движка генерации**:
  - `Demo & Test Mode`: мгновенная генерация стилизованных превью без тяжёлых моделей и внешних серверов.
  - `Automatic1111 WebUI API`: отправка запросов на сервер A1111 (http://127.0.0.1:7860).
  - `Local Diffusers Pipeline`: прямая локальная генерация на PyTorch при наличии скачанного чекпоинта.
- **📁 Шаблоны промптов**: мгновенная вставка промптов из `prompts/`.
- **📊 Пакетный режим**: генерация по списку промптов.
- **🔍 Диагностика**: мониторинг статуса видеокарты, серверов и моделей.

---

## ⚡ Использование через консоль (CLI)

### Генерация через Demo Mode (быстрая проверка)
```bash
python generate.py --prompt "prompts/best_quality_prompt.txt" --engine demo
```

### Генерация через Automatic1111 WebUI API
```bash
python generate.py --prompt "prompts/sexy_variant.txt" --engine webui --steps 35 --cfg 8.0
```

### Проверка подключения к серверам
```bash
python generate.py test-connection
```

### Список доступных моделей
```bash
python generate.py list-models
```

---

## 📸 Параметры для максимальной реалистичности и сексуальности

### Рекомендуемые настройки (Juggernaut XL v9)

| Параметр | Значение | Описание |
|----------|----------|----------|
| **Steps** | 35-40 | Больше шагов = больше деталей кожи, текстуры, реалистичность |
| **CFG Scale** | 7.5-8.5 | Оптимальный баланс между следованием промпту и естественностью |
| **Resolution** | 896x1152 или 768x1024 | Вертикальный формат для OnlyFans, SDXL нативная поддержка |
| **Sampler** | DPM++ 2M Karras | Быстрый и стабильный сэмплер для фотореализма |

### Ключевые элементы промптов для реалистичности:

1. **Текстура кожи**: `(natural detailed skin texture with visible pores and subtle imperfections, goosebumps on arms:1.1)`
2. **Освещение**: `(soft natural lighting creating dramatic shadows across curves, photorealistic catchlights in eyes:1.05)`
3. **Камера/линза**: `shot on 85mm portrait lens f/1.4, creamy bokeh background`
4. **Стиль фото**: `(raw photo, candid amateur photography, 8k uhd, dslr, high resolution:1.2)`

### Negative Prompt (встроен в config.py):
```
(plastic skin, airbrushed, wax figure, CGI, 3D render, cartoon, anime, illustration:1.4), 
(extra fingers, deformed hands, fused fingers, mutated limbs, cross-eyed, malformed eyes:1.4), 
(worst quality, low quality, normal quality:1.3), 
(monochrome, grayscale, bad anatomy, bad proportions, unnatural body, distorted features:1.2), 
(overly symmetrical face, doll-like features, artificial lighting, studio perfection:1.1)
```

---

## 🏎️ Оптимизации производительности v2.3

- **LRU-кэширование и Pre-compiled Regex**: парсинг промптов оптимизирован до **124,000+ операций в секунду**.
- **HTTP Connection Pooling**: постоянная сессия с Keep-Alive и адаптерами пула (`requests.Session` + `HTTPAdapter`) устраняет задержки повторных TCP-хэндшейков.
- **Sub-Millisecond Health Check**: быстрый сокет-тест портов возвращает статус недоступного сервера за доли миллисекунды без зависания на таймаутах.
- **Fast Image I/O**: многопоточное сохранение PNG с оптимальным уровнем сжатия без потери качества.
- **Zero-Crash Design**: отсутствие падений при отсутствии серверов или моделей — интерфейс всегда подсказывает оптимальный шаг.

---

## 🧪 Запуск тестов и бенчмарков

```bash
python -m unittest tests/test_pipeline.py -v
```

### Тестирование OnlyFans стиля:
```bash
# Быстрый демо-тест
python tests/test_photorealistic.py --engine demo --archetype latina_brunette --scenario bikini_poolside

# Реальная генерация через WebUI API
python tests/test_photorealistic.py --engine webui_api --archetype nordic_redhead --scenario mirror_selfie
```

---

## 📁 Промпты в prompts/директории

| Файл | Описание |
|------|----------|
| `best_quality_prompt.txt` | Универсальный высококачественный промпт с акцентом на реалистичность |
| `onlyfans_bedroom_selfie.txt` | Спальня, утреннее освещение, шелковая одежда |
| `onlyfans_boudoir_studio.txt` | Студийная бодиуар фотография в роскошном интерьере |
| `onlyfans_mirror_selfie.txt` | Ванная комната, зеркальное selfie с акцентом на фигуру |
| `sexy_variant.txt` | Обнажённая лиingerie в спальне, драматическое освещение |
| `sexy_variant2.txt` | Пляжный закат, микро бикини, естественный ветер |

---

## 🔧 Настройка конфигурации (configs/generation_config.json)

```json
{
  "generation": {
    "steps": 35,           // Больше для реалистичности
    "cfg_scale": 8.0,      // Оптимально для Juggernaut XL v9
    "width": 896,          // SDXL нативная ширина
    "height": 1152,        // Вертикальный формат OnlyFans
    "sampler_name": "DPM++ 2M Karras"
  }
}
```

---

## 📊 Метаданные и отслеживание

Каждое изображение генерируется с JSON метаданными:
- `results/gen_TIMESTAMP_metadata.json` содержит:
  - Использованный движок (webui_api, comfyui, diffusers)
  - Исходный промпт и параметры
  - Timestamp и seed
  - Список сохранённых файлов

---

## 🌐 Интеграция с ControlNet OpenPose

```bash
# Single pose
python generate.py --prompt "prompts/sexy_variant.txt" --engine webui --controlnet_single poses/pose1.png

# Batch poses (один результат на позу)
python generate.py --prompt "prompts/sexy_variant.txt" --engine webui --controlnet_batch poses/*.png
```

---

## 📞 Поддержка и сообщество

- **Discord**: https://discord.gg/j8mjujwcG — майнор-дженерации, API ключи Grok, обсуждения