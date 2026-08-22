# breath slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_Clean_minus2.wav` | 20.02 | 0.1088 | more Clean (slider -2); LoRA ×-2 |
| 2 | `02_slider_Clean_minus1.wav` | 20.02 | 0.0765 | more Clean (slider -1); LoRA ×-1 |
| 3 | `03_slider_neutral_base_zero.wav` | 20.02 | 0.1229 | neutral base (slider off) |
| 4 | `04_slider_Breathy_plus1.wav` | 20.02 | 0.0996 | more Breathy (slider +1); LoRA ×1 |
| 5 | `05_slider_Breathy_plus2.wav` | 20.02 | 0.0989 | more Breathy (slider +2); LoRA ×2 |
| 6 | `06_REF_prompt_Breathy_no_slider.wav` | 20.02 | 0.1332 | no slider; prompt is the Breathy caption |
| 7 | `07_REF_prompt_Clean_no_slider.wav` | 20.02 | 0.1090 | no slider; prompt is the Clean caption |

- weights: `models/breath-lm-v3/breath-lm-v3_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm
- unit_scale: 1  (user ±1 is one trained concept; LoRA multiplier = user_scale × 1)

If it works, `05_slider_Breathy_plus2.wav` should lean toward the Breathy REF, and `01_slider_Clean_minus2.wav` toward the Clean REF.
