# Realism checklist (NeuronGen)

Gate before video: **mean ≥ 4.0** on skin / hands / light across 6 refs (selfie + 85mm).
Eyes scored too, but gate uses the three above.

## Batch matrix
| # | Archetype | Scenario | Lens | Seed |
|---|-----------|----------|------|------|
| 1 | slavic_blonde | bedroom_morning | selfie | 1001 |
| 2 | slavic_blonde | bedroom_morning | 85mm | 1001 |
| 3 | latina_brunette | bikini_poolside | selfie | 1002 |
| 4 | latina_brunette | bikini_poolside | 85mm | 1002 |
| 5 | nordic_redhead | boudoir_studio | selfie | 1003 |
| 6 | nordic_redhead | boudoir_studio | 85mm | 1003 |

Same seed per pair (selfie vs 85mm). Log actual `pos/neg/CFG/seed/engine` per run (debugger JSON).

## Scoring 1–5
| Axis | 1 | 3 | 5 |
|------|---|---|---|
| **Skin** | plastic / wax / pores missing | mixed: some pores, still beauty-smooth | pores, peach fuzz, natural uneven tone |
| **Eyes** | dead / cross / wrong catchlights | ok shape, flat catchlights | natural asymmetry, believable catchlights |
| **Hands** | extra/fused/floating digits | mostly 5 fingers, soft joints | clear 5 fingers, believable contact |
| **Light** | studio beauty / overclean | mixed window+beauty | window/phone/golden hour, soft shadows |

Fail fast: any hand score ≤2 on a frame with hands in frame → mark scenario `hand_risk` and re-roll with hand-safe pose.

## CFG / prompt defaults for this batch
- **CFG 5.5–7.0** (default **6.0** for realism batch; avoid >8 for photoreal)
- Steps: 25–30 (SD1.5 Realistic Vision) / 30–35 (SDXL Juggernaut)
- Negatives: keep project hand + anti-plastic block (do not strip)
- Ban in positive: `perfect skin`, `airbrushed`, `flawless glass skin`, `beauty retouch`
- Prefer: pores, subtle imperfections, natural asymmetry, window/phone light

## Pass criteria
- Mean(skin, hands, light) ≥ 4.0 over all 6 frames
- No frame with hands ≤2 if hands visible
- Then unlock video experiments (AnimateDiff / API)

## Owner
- Checklist + anti-plastic/hand presets: **coder sexy**
- Actual prompt log JSON: **debugger**
- dtype / CFG wiring: Improver(s) — sync CFG default to **6.0** for realism preset if separate from speed preset

## Score fields for `run_<id>.json` / UI card
Optional block written after manual (or later semi-auto) review:

```json
"scores": {
  "skin": 1,
  "eyes": 1,
  "hands": 1,
  "light": 1,
  "notes": "",
  "scored_by": "coder sexy"
}
```

`mean_gate = (skin + hands + light) / 3` — eyes tracked but not in gate.
