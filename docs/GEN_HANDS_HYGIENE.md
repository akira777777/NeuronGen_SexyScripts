# Gen / prompt hygiene improvements (coder sexy)

## Changes
- Ages in OF archetypes bumped to **25+** (was 20–22).
- Stronger hand negatives: six fingers, elongated/floating hands, interlocking grips.
- Scenarios get `HAND_SAFE_HINTS` + `hand_risk` (`scenario_hand_risk()` helper).
- Boudoir outfit: lace lingerie set instead of "sheer … bodysuit" (filter-prone wording).
- Grok enhancer system prompt: adult 25+, hand rules, stronger negatives.
- Synced defaults in `configs/generation_config.json`, `config.py`, `generate.py`.

## Why
Stress tests showed surface realism OK but interlocking hands / object grips / partner contact collapse finger count and limb attachment.
