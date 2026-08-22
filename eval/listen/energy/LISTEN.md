# energy slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | what it should do |
|-----:|------|-------------------|
| 1 | `01_slider_quiet_minus2.wav` | more quiet (slider -2) |
| 2 | `02_slider_neutral_base_zero.wav` | neutral base (slider off) |
| 3 | `03_slider_loud_plus2.wav` | more loud (slider +2) |
| 4 | `04_REF_prompt_loud_no_slider.wav` | no slider; prompt is the loud caption |
| 5 | `05_REF_prompt_quiet_no_slider.wav` | no slider; prompt is the quiet caption |

- weights: `/ml2/music/sliders-conceptmod/models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- duration: 8.0s  seed: 7  rank: 8  alpha: 8.0

If it works, `01_slider_loud_plus2.wav` should lean toward the loud REF, and the minus extreme toward the quiet REF.
