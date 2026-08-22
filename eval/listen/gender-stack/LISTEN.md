# gender maximize — LM vs transformer vs both

Same lyrics and seed. Slider clips use the **neutral** caption.
Compare 01/03 (LM ±2) to 04/05 (LM ±3) to 06/07 (both ±2) to 08/09 (transformer only).

| file | seconds | rms | setup |
|------|--------:|----:|-------|
| `01_LMonly_male_minus2.wav` | 8.00 | 0.1218 | lm=-2 tf=0 |
| `02_LMonly_neutral_zero.wav` | 8.00 | 0.0904 | lm=0 tf=0 |
| `03_LMonly_female_plus2.wav` | 8.00 | 0.0992 | lm=2 tf=0 |
| `04_LMhot_male_minus3.wav` | 8.00 | 0.0900 | lm=-3 tf=0 |
| `05_LMhot_female_plus3.wav` | 8.00 | 0.0990 | lm=3 tf=0 |
| `06_BOTH_male_minus2.wav` | 8.00 | 0.1161 | lm=-2 tf=-2 |
| `07_BOTH_female_plus2.wav` | 8.00 | 0.1158 | lm=2 tf=2 |
| `08_TFonly_male_minus2.wav` | 8.00 | 0.0790 | lm=0 tf=-2 |
| `09_TFonly_female_plus2.wav` | 8.00 | 0.1052 | lm=0 tf=2 |
| `10_REF_prompt_female_no_slider.wav` | 8.00 | 0.0661 | lm=0 tf=0 |
| `11_REF_prompt_male_no_slider.wav` | 8.00 | 0.0930 | lm=0 tf=0 |

- lm weights: `/ml2/music/sliders-conceptmod/models/gender-lm-slider/gender-lm_last.safetensors`
- transformer weights: `/ml2/music/sliders-conceptmod/models/gender-slider/gender_alpha8.0_rank8_full_last.safetensors`
- duration: 8.0s  seed: 7
