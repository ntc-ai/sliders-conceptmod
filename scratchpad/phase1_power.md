# Phase-1 statistical power: between-approach vs between-seed spread

2026-08-21, computed from renders already on disk (`scratchpad/bs_clips.csv`,
i.e. the blind-spot audit's scan of `eval/listen/curve-gp-4s` and friends).
Reproduce: the deltas are ln(feature@+1 / feature@zero), same seed, same folder.

## The headline numbers

**Unpaired, phase 1 is unfalsifiable. Paired on frozen seeds, it works.**

ln-rms delta at +1, trip-hop pair (the phase-1 testbed):

| approach | n seeds | median | seed sd |
|---|--:|--:|--:|
| E-gp05 (gain_penalty 0.5) | 3 | -1.422 | 0.977 |
| E-gp2 (gain_penalty 2.0)  | 3 | -1.248 | 0.837 |
| R-final-r8-s500 (baseline)| 3 | -1.423 | 1.021 |
| D-pole-uni (pole target)  | 5 | -3.401 | 0.837 |
| D-pop-uni (pole target)   | 5 | -0.195 | 0.345 |

Between-approach sd of medians (1.16) vs median within-approach seed sd (0.84):
ratio 1.39 — and that ratio only exceeds 1 because the pool contains
catastrophes (silence, song replacement). Among the surviving recipe variants
(E-gp05 vs E-gp2) the medians differ by 0.17 against a seed sd of ~0.9:
**unpaired, a real recipe effect is one-fifth of the noise.**

Paired per-seed differences on the SAME seeds (7, 23, 77):

| comparison | per-seed diffs (ln rms) | mean | paired sd | vs unpaired sd |
|---|---|--:|--:|--:|
| E-gp2 - R-final | +0.455, +0.109, +0.174 | +0.25 | 0.18 | 1.02 (5.5x) |
| E-gp05 - R-final | +0.075, -0.003, +0.000 | +0.02 | 0.04 | 1.02 (23x) |
| E-gp2 - E-gp05 | +0.380, +0.112, +0.174 | +0.22 | 0.14 | 0.98 (7x) |
| F-dust-gp05 - F-dust-nopen | -0.009, +0.013, -0.001 | 0.00 | 0.011 | 0.06 (5.5x) |

Same pattern on centroid (paired sd 0.016-0.045 vs unpaired 0.25-0.27).

So: pairing buys a 5-20x noise reduction, recipe effects of ~0.25 ln become
sign-consistent 3/3 at three seeds, and the dust pair's "gp05 and nopen are
identical" claim is confirmed at 0.001 +- 0.011 — i.e. the paired instrument
can also certify a TIE, which the unpaired one never could. With paired sd
~0.18, 3 seeds give sem ~0.10: effects >= ~0.3 ln are clearly resolvable,
~0.15-0.25 needs 5 seeds or the sign test. Anything smaller is inside the
retrain-noise floor and should be called a tie (the floor replicates in the
phase-1 sweep measure that floor directly).

## Consequences baked into the pipeline

- every seed pinned in the spec and recorded in the sidecar (train/init seed,
  cond seed, x0 anchors, eval probe seed, render seeds);
- ranking = paired per-seed differences vs the baseline variant, with
  sign-consistency, never a comparison of independent medians;
- loss/penalty variants at the same --seed share IDENTICAL LoRA init and
  (t, eps) data stream (init draws depend only on rank/targets; the data
  stream is a dedicated generator) — for those comparisons even init noise
  cancels, and the floor replicates bound the worst case;
- holdout seeds (11, 42) are named in the spec and never used for ranking:
  the winner must reproduce its paired advantage there or it was seed-fitting.

## Testbed choice (phase-1 pair)

Trip-hop, because it has headroom in BOTH directions: the live-composed
teacher renders ~-2% rms (target reachable), the trained baseline -76%
(room to improve, effect sizes ~1.3 ln >> paired noise 0.18), silence nearby
(room to get worse), and a known recipe effect (+0.25 ln, E-gp2) exists as a
positive control for the instrument. Dust is at the opposite ceiling: healthy
(-7.5%) with certified-identical recipe variants — nothing to rank, but the
right regression guard for the eventual winner. G-vintage fails G0 (p=0.75),
G-grit is inaudible, G-energy is a loudness ringer: none of them can rank
approaches. If trip-hop turns out to saturate (all formulations equally
collapsed), the honest fallback is a NEW mid-difficulty pair built per the F2
rule (one production attribute at fixed genre/BPM/arrangement) with a stronger
attribute gap than dust — certified through the onboard stage before any
training GPU is spent.

## Post-sweep correction (same day, after the controlled cells ran)

The paired-design POWER argument above stands, but two specific effect claims
made from the archive are now RETRACTED by controlled cells the pipeline ran:

- **gain_penalty 2.0 (+0.25 ln "known effect"): does not reproduce.** With
  everything frozen (seed 7, same init, same data stream), the controlled cell
  reads +0.04 +- 0.08 (2/3 seeds) at x0_per_row 2 and +0.06 +- 0.09 at
  x0_per_row 8 — inside the retrain floor. It is therefore NOT usable as a
  positive control, and the archive's E-gp2-vs-R-final delta was instrument
  drift, not the knob.
- **x0_per_row 8-vs-2 ("+0.8 ln, halves the collapse"): does not reproduce.**
  Controlled: -0.00 +- 0.05. The +0.8 came from comparing pipeline ladders
  (rendered from `_last` checkpoints) against the legacy curve-gp ladders
  (rendered from `_best` checkpoints, different render script settings). An
  exact-config rebuild (X-x0x8, seed 7, anchors 8) still lands +0.85 above the
  legacy R-final ladder per-seed — so checkpoint selection (_best vs _last)
  and/or the legacy render path is worth that much on this pair, and NO
  cross-instrument comparison of ladder deltas is valid. Within-pipeline
  comparisons only.

The measured retrain floor (|paired mean| of same-config reruns at new seeds):
rms_ln ~0.08, intended proj ~0.14. Both retracted "effects" sit at or under it
once measured on one instrument. The sweep's honest phase-1 conclusion: on the
trip-hop pair, NO loss formulation tried (mse, cos, gain-penalty, gain-match,
t-weighted gain-match, orthogonal shape/gain split) moves rendered collapse
out of the veto band; mse trades more collapse for more brightness; the gain
family is inert at train time. The mechanism that matters is not reachable by
these per-step losses — consistent with MUSIC3.md's "the caption pair, not
the recipe, decides".
