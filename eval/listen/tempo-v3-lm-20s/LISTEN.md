# tempo-lm-v3 slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_Slow_minus2.wav` | 20.02 | 0.0982 | more Slow (slider -2); LoRA ×-2 |
| 2 | `02_slider_Slow_minus1.wav` | 20.02 | 0.0849 | more Slow (slider -1); LoRA ×-1 |
| 3 | `03_slider_neutral_base_zero.wav` | 20.02 | 0.0544 | neutral base (slider off) |
| 4 | `04_slider_Fast_plus1.wav` | 20.02 | 0.1272 | more Fast (slider +1); LoRA ×1 |
| 5 | `05_slider_Fast_plus2.wav` | 19.81 | 0.1010 | more Fast (slider +2); LoRA ×2 |
| 6 | `06_REF_prompt_Fast_no_slider.wav` | 20.02 | 0.1334 | no slider; prompt is the Fast caption |
| 7 | `07_REF_prompt_Slow_no_slider.wav` | 20.02 | 0.0818 | no slider; prompt is the Slow caption |

- weights: `models/tempo-lm-v3/tempo-lm-v3_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm
- unit_scale: 1  (user ±1 is one trained concept; LoRA multiplier = user_scale × 1)

If it works, `05_slider_Fast_plus2.wav` should lean toward the Fast REF, and `01_slider_Slow_minus2.wav` toward the Slow REF.
