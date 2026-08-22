# gender-tf-v2 slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_male_minus4.wav` | 8.00 | 0.0386 | more male (slider -4) |
| 2 | `02_slider_male_minus2.wav` | 8.00 | 0.0430 | more male (slider -2) |
| 3 | `03_slider_neutral_base_zero.wav` | 8.00 | 0.0432 | neutral base (slider off) |
| 4 | `04_slider_female_plus2.wav` | 8.00 | 0.0447 | more female (slider +2) |
| 5 | `05_slider_female_plus4.wav` | 8.00 | 0.3974 | more female (slider +4) |
| 6 | `06_REF_prompt_female_no_slider.wav` | 8.00 | 0.0639 | no slider; prompt is the female caption |
| 7 | `07_REF_prompt_male_no_slider.wav` | 8.00 | 0.0225 | no slider; prompt is the male caption |

- weights: `/ml2/music/sliders-conceptmod/models/gender-tf-v2/gender-tf-v2_alpha16.0_rank16_full_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 8.0s  seed: 7  rank: 16  alpha: 16.0  kind: transformer

If it works, `05_slider_female_plus4.wav` should lean toward the female REF, and `01_slider_male_minus4.wav` toward the male REF.
