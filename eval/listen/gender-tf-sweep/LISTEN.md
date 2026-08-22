# gender TF raw-scale sweep — pick your amplitude

Same lyrics and seed. These are **raw** LoRA multipliers (not calibrated units).
RMS rises smoothly from −8 → +8. Use this to decide how hard to push.

Rough guide from this set:
- ±1: subtle
- ±2: clearly moving
- ±4: about as loud as the female REF prompt
- ±6–8: overcooked / slammed

Calibrated ±1 (one trained concept ≈ raw ×2.55) is in `../gender-tf-unit/`.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_male_minus8.wav` | 8.00 | 0.0172 | more male (slider -8) |
| 2 | `02_slider_male_minus6.wav` | 8.00 | 0.0204 | more male (slider -6) |
| 3 | `03_slider_male_minus4.wav` | 8.00 | 0.0259 | more male (slider -4) |
| 4 | `04_slider_male_minus2.wav` | 8.00 | 0.0335 | more male (slider -2) |
| 5 | `05_slider_male_minus1.wav` | 8.00 | 0.0382 | more male (slider -1) |
| 6 | `06_slider_neutral_base_zero.wav` | 8.00 | 0.0432 | neutral base (slider off) |
| 7 | `07_slider_female_plus1.wav` | 8.00 | 0.0494 | more female (slider +1) |
| 8 | `08_slider_female_plus2.wav` | 8.00 | 0.0574 | more female (slider +2) |
| 9 | `09_slider_female_plus4.wav` | 8.00 | 0.0817 | more female (slider +4) |
| 10 | `10_slider_female_plus6.wav` | 8.00 | 0.1234 | more female (slider +6) |
| 11 | `11_slider_female_plus8.wav` | 8.00 | 0.1815 | more female (slider +8) |
| 12 | `12_REF_prompt_female_no_slider.wav` | 8.00 | 0.0639 | no slider; prompt is the female caption |
| 13 | `13_REF_prompt_male_no_slider.wav` | 8.00 | 0.0225 | no slider; prompt is the male caption |

- weights: `/ml2/music/sliders-conceptmod/models/gender-slider/gender_alpha8.0_rank8_full_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 8.0s  seed: 7  rank: 8  alpha: 8.0  kind: transformer

If it works, `11_slider_female_plus8.wav` should lean toward the female REF, and `01_slider_male_minus8.wav` toward the male REF.
