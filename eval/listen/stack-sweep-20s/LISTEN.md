# Stack sweep — how much slider can you pile on at once?

Every clip is resolved through `app/sliders.json` exactly as a studio job is
(`scripts/render_stack_sweep.py`), on the **unit-normalized** catalog, with the
combined budget deliberately off — this folder is what the budget was measured
from. Same neutral caption, lyrics and seed within each pass.

`total` is `sum(|multiplier|)` over the resolved LoRA components — the quantity
`combined_budget` caps. Note it is not the fader sum: triphop, distortion and
tempo each contribute 2 per unit of fader (TF + LM halves), energy 1.5 (its TF
half rides at ratio 0.5), the LM-only axes 1. All ten sliders at ±2 is 27.0,
the maximum reachable.

`rms` and `crest` use the `probe_axis.py` / `probe.json` convention (mono
downmix; crest = 20log10(peak/rms)). `x base` and `d crest` are against that
pass's neutral render.

## Pass A — triphop-v3-single neutral caption, seed 7

| file | total | rms | x base | crest dB | d crest | stack |
|------|------:|----:|-------:|---------:|--------:|-------|
| `00_total00.0.wav` | 0.00 | 0.09288 | 1.000 | 20.15 | 0.00 | (neutral) |
| `01_total02.0.wav` | 2.00 | 0.13827 | 1.489 | 17.19 | -2.96 | energy:+1, space:-0.5 |
| `02_total04.0.wav` | 4.00 | 0.11833 | 1.274 | 18.54 | -1.61 | energy:+1, distortion:-1, space:-0.5 |
| `03_total06.0.wav` | 6.00 | 0.18344 | 1.975 | 14.73 | -5.42 | energy:+1, distortion:-1, tempo:+1, gender:+0.5 |
| `04_total08.0.wav` | 8.00 | 0.12114 | 1.304 | 17.88 | -2.27 | energy:+1, distortion:-1, tempo:+1, triphop:+1, gender:+0.5 |
| `05_total10.0.wav` | 10.00 | 0.08476 | 0.913 | 20.22 | +0.07 | energy:+1.5, distortion:-1, tempo:+1, triphop:+1, breath:+1, gender:-0.75 |
| `06_total12.0.wav` | 12.00 | 0.12864 | 1.385 | 16.83 | -3.32 | energy:+2, distortion:-1.5, tempo:+1, triphop:+1, breath:+1, live:-1 |
| `07_total16.0.wav` | 16.00 | 0.06570 | 0.707 | 19.83 | -0.32 | energy:+2, distortion:-2, tempo:+1.5, triphop:+1.5, breath:+1, live:-1, gender:+1 |
| `08_total20.0.wav` | 20.00 | 0.06757 | 0.727 | 19.44 | -0.71 | energy:+2, distortion:-2, tempo:+2, triphop:+2, breath:+1, live:-1, gender:+1, rapslow:-1, space:+1 |
| `09_total27.0.wav` | 27.00 | 0.09141 | 0.984 | 19.74 | -0.41 | **all ten at +2** |
| `10_total27.0.wav` | 27.00 | 0.07762 | 0.836 | 17.16 | -2.99 | **all ten at -2** |

## Pass B — energy-v3 neutral caption, seed 23 (knee stability check)

Its baseline is a hot, compressed render (rms 0.16485, crest 15.45 dB), so the
ratios read low while the absolute levels stay ordinary and crest *rises*.

| file | total | rms | x base | crest dB | d crest | stack |
|------|------:|----:|-------:|---------:|--------:|-------|
| `00_total00.0_capB.wav` | 0.00 | 0.16485 | 1.000 | 15.45 | 0.00 | (neutral) |
| `01_total08.0_capB.wav` | 8.00 | 0.10130 | 0.614 | 18.37 | +2.92 | energy:+1, distortion:-1, tempo:+1, triphop:+1, gender:+0.5 |
| `02_total20.0_capB.wav` | 20.00 | 0.07873 | 0.478 | 18.16 | +2.71 | 9 sliders, mixed signs, most at ±2 |
| `03_total27.0_capB.wav` | 27.00 | 0.11990 | 0.727 | 16.34 | +0.89 | **all ten at +2** |
| `04_total27.0_capB.wav` | 27.00 | 0.09808 | 0.595 | 19.64 | +4.19 | **all ten at -2** |

## Reading

There is no knee. Over all 16 renders absolute rms stays in 0.0657-0.1834 and
crest in 14.7-20.2 dB — nothing collapses, and the floor sits 22x above the
rms-0.003 near-silence the un-normalized triphop used to produce. Degradation
does not track `total`: the largest rms excursion (1.975x) is at total **6**,
because that stack leans on energy and tempo, which move loudness by design.
The ±2 extremes at total 27 land within 2% and 17% of baseline.

`combined_budget` is therefore set to **28.0** in `app/sliders.json` — just
above the 27.0 maximum reachable, so today it never fires. It is a tested
guard that engages on its own as sliders are added; re-run this sweep and
re-derive it when the catalog grows.

Machine-readable results: `sweep.json` (pass A), `sweep-capB.json` (pass B).
