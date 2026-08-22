# Same-seed ending A/B

Source: library song `5ec87fed`. Seed 7, requested 60s — identical for every row,
so the only difference is the applied LoRA. tail/overall well under 0.1 means the cut
faded out on its own; a hot tail at the cap means the composer never sampled
`<|audio_end|>` and was guillotined.

| file | scale | seconds | tail/overall | ended naturally |
|------|------:|--------:|-------------:|-----------------|
| `01_base.wav` | +0 | 60.07 | 0.147 | **no — cap** |
| `02_live-v3+2.wav` | +2 | 60.07 | 0.936 | **no — cap** |
| `03_live-v4+2.wav` | +2 | 60.07 | 0.436 | **no — cap** |
