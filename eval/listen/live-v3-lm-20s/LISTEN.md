# live slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_Studio_minus2.wav` | 20.02 | 0.1252 | more Studio (slider -2); LoRA ×-2 |
| 2 | `02_slider_Studio_minus1.wav` | 20.02 | 0.1154 | more Studio (slider -1); LoRA ×-1 |
| 3 | `03_slider_neutral_base_zero.wav` | 20.02 | 0.1081 | neutral base (slider off) |
| 4 | `04_slider_Live_plus1.wav` | 20.02 | 0.1329 | more Live (slider +1); LoRA ×1 |
| 5 | `05_slider_Live_plus2.wav` | 20.02 | 0.0893 | more Live (slider +2); LoRA ×2 |
| 6 | `06_REF_prompt_Live_no_slider.wav` | 20.02 | 0.1341 | no slider; prompt is the Live caption |
| 7 | `07_REF_prompt_Studio_no_slider.wav` | 20.02 | 0.1349 | no slider; prompt is the Studio caption |

- weights: `models/live-lm-v3/live-lm-v3_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm
- unit_scale: 1  (user ±1 is one trained concept; LoRA multiplier = user_scale × 1)

If it works, `05_slider_Live_plus2.wav` should lean toward the Live REF, and `01_slider_Studio_minus2.wav` toward the Studio REF.
