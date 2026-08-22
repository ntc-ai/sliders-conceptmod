# distortion slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_acoustic_minus2.wav` | 20.02 | 0.0503 | more acoustic (slider -2) |
| 2 | `02_slider_neutral_base_zero.wav` | 20.02 | 0.0672 | neutral base (slider off) |
| 3 | `03_slider_distorted_plus2.wav` | 20.02 | 0.0934 | more distorted (slider +2) |
| 4 | `04_REF_prompt_distorted_no_slider.wav` | 20.02 | 0.1301 | no slider; prompt is the distorted caption |
| 5 | `05_REF_prompt_acoustic_no_slider.wav` | 20.02 | 0.0792 | no slider; prompt is the acoustic caption |

- weights: `/ml2/music/sliders-conceptmod/models/distortion-slider/distortion_alpha8.0_rank8_full_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: transformer

If it works, `03_slider_distorted_plus2.wav` should lean toward the distorted REF, and `01_slider_acoustic_minus2.wav` toward the acoustic REF.
