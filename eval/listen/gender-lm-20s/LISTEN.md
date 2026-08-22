# gender-lm slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_male_minus2.wav` | 20.02 | 0.1505 | more male (slider -2) |
| 2 | `02_slider_neutral_base_zero.wav` | 20.02 | 0.0818 | neutral base (slider off) |
| 3 | `03_slider_female_plus2.wav` | 20.02 | 0.1294 | more female (slider +2) |
| 4 | `04_REF_prompt_female_no_slider.wav` | 20.02 | 0.1123 | no slider; prompt is the female caption |
| 5 | `05_REF_prompt_male_no_slider.wav` | 20.02 | 0.1220 | no slider; prompt is the male caption |

- weights: `/ml2/music/sliders-conceptmod/models/gender-lm-slider/gender-lm_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm

If it works, `03_slider_female_plus2.wav` should lean toward the female REF, and `01_slider_male_minus2.wav` toward the male REF.
