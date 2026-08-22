# energy-lm-v3 slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_Quiet_minus2.wav` | 20.02 | 0.0864 | more Quiet (slider -2); LoRA ×-2 |
| 2 | `02_slider_Quiet_minus1.wav` | 20.02 | 0.1059 | more Quiet (slider -1); LoRA ×-1 |
| 3 | `03_slider_neutral_base_zero.wav` | 20.02 | 0.1119 | neutral base (slider off) |
| 4 | `04_slider_Loud_plus1.wav` | 20.02 | 0.1244 | more Loud (slider +1); LoRA ×1 |
| 5 | `05_slider_Loud_plus2.wav` | 20.02 | 0.1500 | more Loud (slider +2); LoRA ×2 |
| 6 | `06_REF_prompt_Loud_no_slider.wav` | 20.02 | 0.1358 | no slider; prompt is the Loud caption |
| 7 | `07_REF_prompt_Quiet_no_slider.wav` | 20.02 | 0.0770 | no slider; prompt is the Quiet caption |

- weights: `models/energy-lm-v3/energy-lm-v3_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm
- unit_scale: 1  (user ±1 is one trained concept; LoRA multiplier = user_scale × 1)

If it works, `05_slider_Loud_plus2.wav` should lean toward the Loud REF, and `01_slider_Quiet_minus2.wav` toward the Quiet REF.
