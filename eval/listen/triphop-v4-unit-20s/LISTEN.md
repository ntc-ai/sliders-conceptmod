# triphop — exactly what the studio ships

Resolved through `app/sliders.json` (Pop <-> Trip-hop), so the multipliers
below include each component's ratio, gain and sidecar `unit_scale`.
Same neutral caption, lyrics and seed throughout.

| file | seconds | rms | applied |
|------|--------:|----:|---------|
| `01_shipped_Pop_minus2.wav` | 20.02 | 0.1474 | triphop-tf-v4x-2.00, triphop-lm-v3x-2.00 |
| `02_shipped_zero.wav` | 20.02 | 0.0975 | none |
| `03_shipped_Trip-hop_plus2.wav` | 20.02 | 0.0718 | triphop-tf-v4x+2.00, triphop-lm-v3x+2.00 |

- seed: 7  duration: 20.0s  prompt row: 0
