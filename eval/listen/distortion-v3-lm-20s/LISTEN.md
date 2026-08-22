# distortion-lm-v3 slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_Clean_minus2.wav` | 20.02 | 0.1120 | more Clean (slider -2); LoRA ×-2 |
| 2 | `02_slider_Clean_minus1.wav` | 20.02 | 0.1097 | more Clean (slider -1); LoRA ×-1 |
| 3 | `03_slider_neutral_base_zero.wav` | 20.02 | 0.0672 | neutral base (slider off) |
| 4 | `04_slider_Heavy_plus1.wav` | 20.02 | 0.1174 | more Heavy (slider +1); LoRA ×1 |
| 5 | `05_slider_Heavy_plus2.wav` | 20.02 | 0.0752 | more Heavy (slider +2); LoRA ×2 |
| 6 | `06_REF_prompt_Heavy_no_slider.wav` | 20.02 | 0.1301 | no slider; prompt is the Heavy caption |
| 7 | `07_REF_prompt_Clean_no_slider.wav` | 20.02 | 0.0792 | no slider; prompt is the Clean caption |

- weights: `models/distortion-lm-v3/distortion-lm-v3_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm
- unit_scale: 1  (user ±1 is one trained concept; LoRA multiplier = user_scale × 1)

If it works, `05_slider_Heavy_plus2.wav` should lean toward the Heavy REF, and `01_slider_Clean_minus2.wav` toward the Clean REF.
