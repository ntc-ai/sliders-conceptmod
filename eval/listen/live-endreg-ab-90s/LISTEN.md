# Same-seed ending A/B

Source: library song `5ec87fed`. Seed 7, requested 90s — identical for every row,
so the only difference is the applied LoRA. tail/overall well under 0.1 means the cut
faded out on its own; a hot tail at the cap means the composer never sampled
`<|audio_end|>` and was guillotined.

| file | scale | seconds | tail/overall | ended naturally |
|------|------:|--------:|-------------:|-----------------|
| `01_base.wav` | +0 | 76.73 | 0.020 | yes |
| `02_live-v3+2.wav` | +2 | 90.11 | 1.045 | **no — cap** |
| `03_live-v4+2.wav` | +2 | 84.99 | 0.018 | yes |
