# gender transformer v1 — it DOES move (mix, not singer)

Same lyrics, same seed, **same captions the LoRA was trained on**.
Play −4 then +4. Loudness/body change. The singer stays the same person.

The earlier `gender/` and `gender-stack/` TF clips used a different prompt
file than training, so ±2 sounded frozen.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_male_minus4.wav` | 8.00 | 0.0259 | more male (slider -4) |
| 2 | `02_slider_male_minus2.wav` | 8.00 | 0.0335 | more male (slider -2) |
| 3 | `03_slider_neutral_base_zero.wav` | 8.00 | 0.0432 | neutral base (slider off) |
| 4 | `04_slider_female_plus2.wav` | 8.00 | 0.0574 | more female (slider +2) |
| 5 | `05_slider_female_plus4.wav` | 8.00 | 0.0817 | more female (slider +4) |
| 6 | `06_REF_prompt_female_no_slider.wav` | 8.00 | 0.0639 | no slider; prompt is the female caption |
| 7 | `07_REF_prompt_male_no_slider.wav` | 8.00 | 0.0225 | no slider; prompt is the male caption |

- weights: `/ml2/music/sliders-conceptmod/models/gender-slider/gender_alpha8.0_rank8_full_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 8.0s  seed: 7  rank: 8  alpha: 8.0  kind: transformer

If it works, `05_slider_female_plus4.wav` should lean toward the female REF, and `01_slider_male_minus4.wav` toward the male REF.
