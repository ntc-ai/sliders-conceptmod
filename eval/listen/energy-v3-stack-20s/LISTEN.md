# Quiet <-> Loud stack — LM vs transformer vs both

Same lyrics and seed. Slider clips use the **neutral** caption.
Compare 01/03 (LM ±2) to 04/05 (LM ±3) to 06/07 (both) to 08/09 (transformer only).

| file | seconds | rms | setup |
|------|--------:|----:|-------|
| `01_LMonly_Quiet_minus2.wav` | 20.02 | 0.0864 | reused |
| `02_LMonly_neutral_zero.wav` | 20.02 | 0.1119 | reused |
| `03_LMonly_Loud_plus2.wav` | 20.02 | 0.1500 | reused |
| `04_LMhot_Quiet_minus3.wav` | 20.02 | 0.0331 | lm=-3 tf=0 |
| `05_LMhot_Loud_plus3.wav` | 20.02 | 0.1349 | lm=3 tf=0 |
| `06_BOTH_Quiet_minus2.wav` | 20.02 | 0.0630 | lm=-2 tf=-4.02088 |
| `07_BOTH_Loud_plus2.wav` | 20.02 | 0.2785 | lm=2 tf=4.02088 |
| `08_TFonly_Quiet_minus2.wav` | 20.02 | 0.0772 | lm=0 tf=-4.02088 |
| `09_TFonly_Loud_plus2.wav` | 20.02 | 0.2477 | lm=0 tf=4.02088 |
| `10_REF_prompt_Loud_no_slider.wav` | 20.02 | 0.1358 | lm=0 tf=0 |
| `11_REF_prompt_Quiet_no_slider.wav` | 20.02 | 0.0770 | lm=0 tf=0 |

- lm weights: `models/energy-lm-v3/energy-lm-v3_last.safetensors`
- transformer weights: `models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors`
- duration: 20.0s  seed: 7
