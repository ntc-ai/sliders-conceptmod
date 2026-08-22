# rhyme (held-out runtime caption) slider — play in order

Same lyrics and same seed. Slider clips use the **neutral** caption;
only the LoRA scale changes. REF clips change the **prompt** with the slider off.

| play | file | seconds | rms | what it should do |
|-----:|------|--------:|----:|-------------------|
| 1 | `01_slider_Prose_minus2.wav` | 20.02 | 0.1372 | more Prose (slider -2); LoRA ×-2 |
| 2 | `02_slider_Rhyme_plus2.wav` | 20.02 | 0.1224 | more Rhyme (slider +2); LoRA ×2 |
| 3 | `03_REF_prompt_Rhyme_no_slider.wav` | 20.02 | 0.1320 | no slider; prompt is the Rhyme caption |
| 4 | `04_REF_prompt_Prose_no_slider.wav` | 20.02 | 0.1326 | no slider; prompt is the Prose caption |

- weights: `models/rhyme-lm-v4/rhyme-lm-v4_last.safetensors`
- lyrics: `[verse] / I can feel it in the air tonight / [chorus] / Louder now or fade away`
- requested: 20.0s  seed: 7  rank: 8  alpha: 8.0  kind: lm
- unit_scale: 1  (user ±1 is one trained concept; LoRA multiplier = user_scale × 1)

If it works, `02_slider_Rhyme_plus2.wav` should lean toward the Rhyme REF, and `01_slider_Prose_minus2.wav` toward the Prose REF.
