# Pop <-> Trip-hop stack — LM vs transformer vs both

Same lyrics and seed. Slider clips use the **neutral** caption.
Compare 01/03 (LM ±1.5) to 04/05 (LM ±2.5) to 06/07 (both) to 08/09 (transformer only).

| file | seconds | rms | setup |
|------|--------:|----:|-------|
| `01_LMonly_Pop_minus1.5.wav` | 20.02 | 0.1290 | lm=-1.5 tf=0 |
| `02_LMonly_neutral_zero.wav` | 20.02 | 0.0975 | lm=0 tf=0 |
| `03_LMonly_Trip-hop_plus1.5.wav` | 20.02 | 0.0762 | lm=1.5 tf=0 |
| `04_LMhot_Pop_minus2.5.wav` | 20.02 | 0.1021 | lm=-2.5 tf=0 |
| `05_LMhot_Trip-hop_plus2.5.wav` | 20.02 | 0.1258 | lm=2.5 tf=0 |
| `06_BOTH_Pop_minus1.5.wav` | 20.02 | 0.1340 | lm=-1.5 tf=-2.72945 |
| `07_BOTH_Trip-hop_plus1.5.wav` | 20.02 | 0.0687 | lm=1.5 tf=2.72945 |
| `08_TFonly_Pop_minus1.5.wav` | 20.02 | 0.1097 | lm=0 tf=-2.72945 |
| `09_TFonly_Trip-hop_plus1.5.wav` | 20.02 | 0.0898 | lm=0 tf=2.72945 |
| `10_REF_prompt_Trip-hop_no_slider.wav` | 20.02 | 0.1113 | lm=0 tf=0 |
| `11_REF_prompt_Pop_no_slider.wav` | 20.02 | 0.1361 | lm=0 tf=0 |

- lm weights: `models/triphop-lm-v3/triphop-lm-v3_last.safetensors`
- transformer weights: `models/triphop-slider/triphop_alpha8.0_rank8_full_last.safetensors`
- duration: 20.0s  seed: 7
