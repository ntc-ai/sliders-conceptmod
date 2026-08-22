# triphop-lm-v3 slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_Pop_minus2.wav` | 20.02 | 0.0998 | more Pop (slider -2); LoRA ×-2 |
| 2 | `02_slider_Pop_minus1.wav` | 20.02 | 0.1424 | more Pop (slider -1); LoRA ×-1 |
| 3 | `03_slider_neutral_base_zero.wav` | 20.02 | 0.0975 | neutral base (slider off) |
| 4 | `04_slider_Trip-hop_plus1.wav` | 20.02 | 0.1446 | more Trip-hop (slider +1); LoRA ×1 |
| 5 | `05_slider_Trip-hop_plus2.wav` | 20.02 | 0.1104 | more Trip-hop (slider +2); LoRA ×2 |
| 6 | `06_REF_prompt_Trip-hop_no_slider.wav` | 20.02 | 0.1113 | no slider; prompt is the Trip-hop caption |
| 7 | `07_REF_prompt_Pop_no_slider.wav` | 20.02 | 0.1361 | no slider; prompt is the Pop caption |

- weights: `models/triphop-lm-v3/triphop-lm-v3_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm
- unit_scale: 1  (user ±1 is one trained concept; LoRA multiplier = user_scale × 1)

If it works, `05_slider_Trip-hop_plus2.wav` should lean toward the Trip-hop REF, and `01_slider_Pop_minus2.wav` toward the Pop REF.
