# energy LoRA range sweep

Same caption, lyrics and seed. Only the shipped slider scale changes.

| file | scale | seconds | rms | applied |
|------|------:|--------:|----:|---------|
| `01_scale_Quiet_minus1.wav` | -1 | 8.00 | 0.1046 | energyx-0.50, energy-lm-v4x-1.00 |
| `02_scale_Quiet_minus0.75.wav` | -0.75 | 8.00 | 0.1041 | energyx-0.38, energy-lm-v4x-0.75 |
| `03_scale_Quiet_minus0.5.wav` | -0.5 | 8.00 | 0.0785 | energyx-0.25, energy-lm-v4x-0.50 |
| `04_scale_Quiet_minus0.25.wav` | -0.25 | 8.00 | 0.1489 | energyx-0.12, energy-lm-v4x-0.25 |
| `05_scale_zero.wav` | +0 | 8.00 | 0.0722 | none |
| `06_scale_Loud_plus0.25.wav` | +0.25 | 8.00 | 0.0717 | energyx+0.12, energy-lm-v4x+0.25 |
| `07_scale_Loud_plus0.5.wav` | +0.5 | 8.00 | 0.0956 | energyx+0.25, energy-lm-v4x+0.50 |
| `08_scale_Loud_plus0.75.wav` | +0.75 | 8.00 | 0.1080 | energyx+0.38, energy-lm-v4x+0.75 |
| `09_scale_Loud_plus1.wav` | +1 | 8.00 | 0.1322 | energyx+0.50, energy-lm-v4x+1.00 |

Mix: `energy_sweep_-1_to_1.wav`  hold=1.5s sweep=5.0s
