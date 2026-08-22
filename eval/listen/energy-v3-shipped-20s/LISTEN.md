# energy — exactly what the studio ships

Resolved through `app/sliders.json` (Quiet <-> Loud), so the multipliers
below include each component's ratio, gain and sidecar `unit_scale`.
Same neutral caption, lyrics and seed throughout.

| file | seconds | rms | applied |
|------|--------:|----:|---------|
| `01_shipped_Quiet_minus2.wav` | 20.02 | 0.0715 | energyx-2.00, energy-lm-v3x-2.00 |
| `02_shipped_zero.wav` | 20.02 | 0.1119 | none |
| `03_shipped_Loud_plus2.wav` | 20.02 | 0.1957 | energyx+2.00, energy-lm-v3x+2.00 |

- seed: 7  duration: 20.0s  prompt row: 0
