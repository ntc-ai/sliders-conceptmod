# trip-hop — separate the two sliders, then combine them

Same lyrics and seed. Slider clips use the **neutral** caption.
Transformer amplitude is calibrated unit_scale=1.81963 (one trained concept).
LM amplitude is 2 (same as the working gender LM slider).

| file | sec | rms | setup |
|------|----:|----:|-------|
| `01_SEPARATE_base.wav` | 20.02 | 0.0975 | no sliders |
| `02_SEPARATE_TF_triphop.wav` | 20.02 | 0.0933 | transformer only +1.81963 |
| `03_SEPARATE_TF_glossypop.wav` | 20.02 | 0.1046 | transformer only -1.81963 |
| `04_SEPARATE_LM_triphop.wav` | 20.02 | 0.1431 | LM only +2 |
| `05_SEPARATE_LM_glossypop.wav` | 20.02 | 0.1307 | LM only -2 |
| `06_COMBINED_both_triphop.wav` | 20.02 | 0.1425 | LM + transformer trip-hop |
| `07_SPLIT_TFtriphop_LMpop.wav` | 20.02 | 0.1238 | TF trip-hop mix, LM glossy plan |
| `08_SPLIT_TFpop_LMtriphop.wav` | 20.02 | 0.1441 | LM trip-hop plan, TF glossy mix |
| `09_REF_prompt_triphop.wav` | 20.02 | 0.1193 | no slider, trip-hop prompt |
| `10_REF_prompt_glossypop.wav` | 20.02 | 0.1385 | no slider, glossy-pop prompt |

Play **01 → 02 → 04 → 06**. That is: dry, TF mix only, LM plan only, both.
Then **07 / 08** to hear the axes pulled apart.
Then **09** as the prompt-only trip-hop ceiling.

- duration: 20.0s  seed: 7
- tf weights: `/ml2/music/sliders-conceptmod/models/triphop-slider/triphop_alpha8.0_rank8_full_last.safetensors`
- lm weights: `/ml2/music/sliders-conceptmod/models/triphop-lm-slider/triphop-lm_last.safetensors`
