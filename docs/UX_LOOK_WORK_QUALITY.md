# Look / Work / Quality (NeuronGen Studio)

## How it should look (UI)
1. **Mode switch** at top of Studio tab: `Speed` | `Realism / Quality` (default for content work: Realism).
2. Realism mode auto-sets CFG **6.0**, steps **28**, phone portrait **512×768**, keeps hand+anti-plastic negative locked (read-only unless advanced unlock).
3. Next to gallery: **Run card** from `results/run_<id>.json` — show actual CFG/seed/engine + truncated pos/neg (debugger log), not the disk preset text alone.
4. Optional badge on preset: `hand_risk: low|medium|high` from `onlyfans_presets.scenario_hand_risk`.
5. Score strip after batch (manual or later semi-auto): Skin / Eyes / Hands / Light 1–5 per `REALISM_CHECKLIST.md`.

## How it should work
1. Pick archetype + scenario → build OF prompt (25+ ages, hand-safe camera hints).
2. Generate with logged `run_id`; N/M frames in metadata.
3. Score against checklist; mean(skin, hands, light) ≥ 4.0 before any video path.
4. Stability layer (Improver/debugger): model cache, offload, no float32 cache — UI stays thin over a solid pipe.

## Quality = realism, not “more polish”
- Ban beauty tokens in Realism profile (`perfect skin`, `airbrushed`, …).
- Prefer pores, natural asymmetry, window/phone light.
- Hands: avoid interlock/object-grip poses; strong negatives stay on.
- CFG > 8 usually hurts photoreal — Realism profile caps intent at 6.0.

## Owners
- Realism/quality presets + checklist: **coder sexy**
- Pipe stability / dtype: Improver*
- Run JSON log / crashes: debugger
- Gradio chrome: thin wrapper once pipe is stable
