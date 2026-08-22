# Same-seed ending A/B

Source: library song `5ec87fed`. Seed 7, requested 90s — identical for every row,
so the only difference is the applied LoRA. tail/overall well under 0.1 means the cut
faded out on its own; a hot tail at the cap means the composer never sampled
`<|audio_end|>` and was guillotined.

| file | scale | seconds | tail/overall | ended naturally |
|------|------:|--------:|-------------:|-----------------|
| `01_base.wav` | +0 | 76.73 | 0.020 | yes |
| `02_gender-v4+2.wav` | +2 | 81.50 | 0.025 | yes |
| `03_rapslow-v4+2.wav` | +2 | 82.81 | 0.019 | yes |
| `04_triphop-v4+2.wav` | +2 | 85.34 | 0.276 | yes |
| `05_energy-v4+2.wav` | +2 | 72.61 | 0.024 | yes |
| `06_tempo-v4+2.wav` | +2 | 65.20 | 0.428 | yes |
| `07_distortion-v4+2.wav` | +2 | 82.66 | 0.658 | yes |
| `08_breath-v4+2.wav` | +2 | 62.19 | 0.023 | yes |
| `09_live-v5+2.wav` | +2 | 67.76 | 0.018 | yes |
| `10_rhyme-v5+2.wav` | +2 | 90.11 | 1.644 | **no — cap** |
