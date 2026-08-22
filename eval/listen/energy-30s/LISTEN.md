# energy slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_quiet_minus2.wav` | 30.02 | 0.0995 | more quiet (slider -2) |
| 2 | `02_slider_neutral_base_zero.wav` | 30.02 | 0.1237 | neutral base (slider off) |
| 3 | `03_slider_loud_plus2.wav` | 30.02 | 0.1594 | more loud (slider +2) |
| 4 | `04_REF_prompt_loud_no_slider.wav` | 30.02 | 0.1346 | no slider; prompt is the loud caption |
| 5 | `05_REF_prompt_quiet_no_slider.wav` | 30.02 | 0.0833 | no slider; prompt is the quiet caption |

- weights: `/ml2/music/sliders-conceptmod/models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 30.0s  seed: 7  rank: 8  alpha: 8.0  kind: transformer

If it works, `03_slider_loud_plus2.wav` should lean toward the loud REF, and `01_slider_quiet_minus2.wav` toward the quiet REF.
