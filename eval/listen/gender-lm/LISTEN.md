# gender-lm slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | what it should do |
|-----:|------|-------------------|
| 1 | `01_slider_male_minus2.wav` | more male (slider -2) |
| 2 | `02_slider_neutral_base_zero.wav` | neutral base (slider off) |
| 3 | `03_slider_female_plus2.wav` | more female (slider +2) |
| 4 | `04_REF_prompt_female_no_slider.wav` | no slider; prompt is the female caption |
| 5 | `05_REF_prompt_male_no_slider.wav` | no slider; prompt is the male caption |

- weights: `/ml2/music/sliders-conceptmod/models/gender-lm-slider/gender-lm_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- duration: 8.0s  seed: 7  rank: 8  alpha: 8.0

If it works, `01_slider_female_plus2.wav` should lean toward the female REF, and the minus extreme toward the male REF.
