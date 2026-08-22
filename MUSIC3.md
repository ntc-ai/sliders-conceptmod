# MiniMax Music 3 concept sliders

Trained weights and 20-second listening examples are published to the Hub:
**[ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)**
(`weights/` mirrors `models/`, `samples/` mirrors `eval/listen/`). They are kept out
of git because they run to ~680 MB.

Use the `minimax-music3` conda env. **Never** `pip install -r requirements.txt` (it pins ancient torch/diffusers).

GPU 0 only:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/ml2/music/sliders-conceptmod
export HF_HUB_OFFLINE=1
export HF_HOME=/ml2/music/.cache/huggingface
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
```

## What works

| slider | LoRA host | 8s | 20s | last-step cos / residual |
|---|---|---|---|---|
| **energy** (quiet ↔ loud) | flow transformer `MiniMaxMusic3Attention` | `eval/listen/energy/` | `eval/listen/energy-20s/` and `energy-30s/` | cos ~0.43 |
| **distortion** (acoustic ↔ metal) | transformer | `eval/listen/distortion/` | `eval/listen/distortion-20s/` | cos ~0.43 |
| **tempo** (slow ↔ fast) | transformer | — | `eval/listen/tempo-20s/` | cos ~0.47 |
| **space** (dry ↔ wet) | transformer | — | `eval/listen/space-20s/` | cos ~0.42 |
| **gender** (male ↔ female) | **language model** `Qwen3Attention` | `eval/listen/gender-lm/` | `eval/listen/gender-lm-20s/` | residual ~5%, cos ~1.0 |

v3 adds language-model halves for energy, tempo and distortion, so those axes
now change the *arrangement* (how hard the band plays, the composed BPM, which
instruments are used) as well as the mix. See "Which host does an axis belong
to?" below.

A transformer LoRA **cannot** change singer gender. That is decided in the AR language model. `eval/listen/gender/` is the failed transformer attempt.

Working recipe: rank 8, alpha 8, 500 steps, lr `1e-4`, train duration 4s (AR cache). Generation duration is independent of train duration.

## v3 recipe (August 2026)

Four trainer bugs were found and fixed; v3 sliders (`models/*-v3/`) use:

1. **On-manifold x_t** — the model's flow time runs 0 = noise → 1 = clean
   (`denoise.py`), but the old trainer fed pure `randn` at every t. v3 anchors
   `x_t = (1−t)·ε + t·x0` to clean latents generated per condition
   (`--xt_mode anchor`, cached as `cache/<name>/x0_*.pt`; `noise` = legacy A/B).
2. **Bidirectional loss** (`--bidirectional`, default on) — the −1 direction is
   trained explicitly against `vel_neu − g·(vel_pos − vel_neg)`; per-step
   `cos_pos` / `cos_neg` / `collapse` are logged.
3. **Symmetric LM targets** (`train_lm_slider_music3.py --symmetric`, default
   on) — targets are antisymmetrized around neutral: `tgt(±1) = neu ± (pos−neg)/2`.
   This fixed the collapse metric from **+0.53** (gender-lm v2: both directions
   pointed the same way) to **−0.97** (gender-lm-v3: near-opposite).
   `--common_beta` blends the shared component back in if ±1 sounds weak.
4. **Module-identity dedupe in `lora.py`** — `--targets full` used to wrap
   every attention/FF Linear **twice** (438 modules with two name prefixes,
   2× gain). Now 222 unique modules. Old double-wrapped checkpoints
   (gender-tf-v2) still load with their own sidecars; never recalibrate them.

**Recipe A/B (triphop TF, what actually matters):**

| x_t | rows | steps | cos | cos_axis | verdict |
|---|---|---|---|---|---|
| noise (v2) | single | 500 | 0.46 | — (unit 1.82) | old baseline, mild but real |
| **anchor** | **single** | 250 | 0.45 | 0.41 | **matches v2 at half the steps** |
| noise | multi (4×2 seeds) | 250 | 0.16 | 0.08 | axis lost |
| anchor | multi (4×2 seeds) | 500 | 0.25 | 0.05 | axis lost |

So: the x0 anchor speeds transformer-slider convergence, but **multi-row /
attributes averaging destroys transformer style sliders** — a rank-8 attention
LoRA cannot find one shared "trip-hop direction" across diverse conditions.
Use multi-row + attributes for **LM** sliders (they thrive: gender-lm-v3
collapse −0.97, rapslow-lm-v3 −0.996, triphop-lm-v3 −0.99) and single-row for
**transformer** sliders. Style/delivery axes (gender, rap↔slow) belong in the
LM entirely — the rapslow transformer attempt calibrated at cos_axis 0.08
(`axis_tracking_low`), same failure mode as gender-tf.

**Shipped triphop TF is now `triphop-tf-v4`** (August 2026): a clean retrain of
the anchor/single cell from `prompts-triphop-v3-single.yaml` after the v2
baseline's prompt was found to contain artist names (weights quarantined in
`models/retired-triphop-slider-artist-names/`, banned by AGENTS.md). Two
scaling facts learned shipping it:

- The attn-target checkpoint has ~1.9x lower gain per multiplier than the old
  v2 `full` one, so calibration lands at unit_scale 3.6 — above the
  `min(unit_scale, 2.0)` cap in `app/sliders.py`, which silently truncates
  normalization for exactly the checkpoints that need it most. Worse, the
  calibration target itself ("one trained concept" from 3-timestep velocity
  deltas on a 4s clip) over-drives this checkpoint: at face-value settings the
  +2 render is near-silent (rms 0.003) even though its applied delta is
  *smaller* than the v2 baseline's. Velocity deltas do not predict 50-step
  20s renders; the shipped-path render (`scripts/render_shipped_slider.py`)
  is the only strength signal to trust.
- Fix that shipped: the v4 checkpoint's `.alpha` tensors and sidecar are
  de-rated 8.0 -> 1.0 (an empirical 8x cut sized by render RMS against the old
  shipped baseline), recalibrated to unit_scale 31.1, ratio 1.0 in
  sliders.json. Note the safetensors filename still says `alpha8.0`.
  (Superseded — the `min(unit,2)` cap this originally leaned on is gone; see
  "Every shipped LoRA is unit-normalized" below.)

## Every shipped LoRA is unit-normalized (August 2026)

A checkpoint must **drop in at strength 1**. The old runtime formula carried
normalization at inference time — `scale x ratio x gain x min(unit_scale, 2)` —
and that `min(..., 2)` was a silent lie: it truncated exactly the checkpoints
whose calibration said they needed the most correction (triphop-tf-v4 at
unit 31.1), so the file on disk and the sound in the studio disagreed.

The fix is to bake the runtime factor into the file. For each shipped
component, `B = gain x min(unit_scale, 2.0)` — the exact number the old code
would have applied. `B` was 1.0 for 12 of the 14 shipped components; the two
that needed it got a **new** file beside the original (originals untouched):

| axis | new file | B | `.alpha` | sidecar `alpha` | sidecar `unit_scale` |
|---|---|--:|---|---|---|
| triphop TF | `triphop-tf-v4/triphop-tf-v4_unit_last.safetensors` | 2.0 | 1.0 -> 2.0 | 1.0 -> 2.0 | 31.109 -> 1.0 |
| energy TF | `energy-slider-v2/energy_unit_last.safetensors` | 2.0 | 8.0 -> 16.0 | 8.0 -> 16.0 | 2.010 -> 1.0 |

Both halves have to move together: `app/lora_runtime.refresh_scales()` derives
each module's scale from the checkpoint's `.alpha` **tensors**, while
`conceptmod/textsliders/lora.py` derives it from the **sidecar** `alpha` at
construction and never re-reads the buffers. Bake one and not the other and the
studio and the eval scripts drift apart silently.

`min(unit, 2.0)` is now deleted from `app/sliders.py`; the formula is plain
`scale x ratio x gain x unit_scale`, and a resolved component whose `unit_scale`
is not 1.0 (±1e-3) logs a loud warning. `ratio` stays a recipe knob — energy's
transformer half still rides at 0.5.

Applied strength is unchanged by the bake, by construction (multiplier /= B,
alpha *= B) and by measurement: re-rendering triphop through
`scripts/render_shipped_slider.py` off the normalized file reproduces
`eval/listen/triphop-v4-shipped-20s/` to five decimals (0.1474 / 0.0975 /
0.0718 at -2 / 0 / +2).

### Combined budget: measured, and a no-op at today's catalog

With normalization in the files, the only runtime knob left is a cap on how
much slider a user can stack at once. `app/sliders.json` takes an optional
top-level `combined_budget`: when `sum(|multiplier|)` exceeds it, every
multiplier is scaled by the same `budget / total`, preserving the balance the
user dialled in; below it nothing changes at all.

Its value was **measured**, not guessed, with `scripts/render_stack_sweep.py`
(20s, fixed neutral caption + lyrics, rms/crest via the `probe_axis.py`
convention). Escalating mixed TF+LM stacks with alternating signs, plus the
worst case of all ten sliders at ±2 (which totals **27.0**, the maximum
reachable — triphop/distortion/tempo contribute 2 each and energy 1.5):

| total | rms | x base | crest dB | d crest | stack |
|--:|--:|--:|--:|--:|---|
| 0.0 | 0.09288 | 1.000 | 20.15 | 0.00 | (neutral baseline) |
| 2.0 | 0.13827 | 1.489 | 17.19 | -2.96 | energy+1, space-0.5 |
| 4.0 | 0.11833 | 1.274 | 18.54 | -1.61 | + distortion-1 |
| 6.0 | 0.18344 | 1.975 | 14.73 | -5.42 | energy+1, distortion-1, tempo+1, gender+0.5 |
| 8.0 | 0.12114 | 1.304 | 17.88 | -2.27 | + triphop+1 |
| 10.0 | 0.08476 | 0.913 | 20.22 | +0.07 | energy+1.5, distortion-1, tempo+1, triphop+1, breath+1, gender-0.75 |
| 12.0 | 0.12864 | 1.385 | 16.83 | -3.32 | energy+2, distortion-1.5, tempo+1, triphop+1, breath+1, live-1 |
| 16.0 | 0.06570 | 0.707 | 19.83 | -0.32 | + gender+1, tempo/triphop to 1.5, distortion-2 |
| 20.0 | 0.06757 | 0.727 | 19.44 | -0.71 | 9 sliders, mixed signs, most at ±2 |
| 27.0 | 0.09141 | 0.984 | 19.74 | -0.41 | **all ten at +2** |
| 27.0 | 0.07762 | 0.836 | 17.16 | -2.99 | **all ten at -2** |

Second caption + seed (energy-v3 row 0, seed 23; baseline is a hot 0.16485 /
15.45 dB) at four points: total 8 -> 0.10130 (0.61x, crest +2.92), total 20 ->
0.07873 (0.48x, +2.71), all +2 -> 0.11990 (0.73x, +0.89), all -2 -> 0.09808
(0.60x, +4.19).

**There is no knee.** Across all 16 renders absolute rms stays in
0.0657-0.1834 (a 2.8x spread, and 22x above the rms-0.003 near-silence that the
un-normalized triphop produced) and crest stays in 14.7-20.2 dB — never
collapsed. Degradation does not track the total: the *worst* rms excursion
(1.975x) is at total **6**, driven by energy being a loudness axis doing its
job, and the ±2 extremes at total 27 land within 2% and 17% of baseline. The
apparent 0.48x at caption B is an artifact of that caption's unusually loud
baseline, and its crest goes *up* — less squashed, not degraded.

So `combined_budget` ships at **28.0**: above the 27.0 maximum reachable today,
i.e. a deliberate **no-op guard**. Nothing a user can do to the current ten
sliders is budgeted. It exists so the mechanism is in place and tested, and it
starts biting automatically as sliders are added — re-run the sweep and
re-derive it when the catalog grows.

Also (fixed, see below): `calibrate_scale.py` `_load_cache_entries` KeyErrored on the `x0_*.pt`
anchor files in a shared cache dir (it expects only condition entries) —
work around with a scratch dir holding just the condition files.

## Read this before the `cos` sections below (August 21 2026)

A day spent raising the probe metric from 0.52 to 0.81 produced no audible
improvement, and the measurement campaign that followed retracted most of what
the next four sections assert. They are kept because their *measurements* are
sound and still useful; their *conclusions* are superseded. Specifically:

**`cos` does not predict renders.** Checkpoints scoring 0.82, 0.78 and 0.77 on
the probe render, respectively, digital silence, a usable effect, and a track
with 96% of its level gone. The probe evaluates the adapter at x_t drawn from
*unperturbed* trajectories — precisely the states an open-loop adapter handles
well — while at inference it drives its own trajectory for ~50 steps. Rank runs
with the probe to catch broken ones; accept them only on a rendered ladder
(`scripts/score_render_curve.py`).

**Single-seed 4s feature deltas are not evidence.** With captions and conditions
fixed and only the denoise seed changed, the caption swap's own signature swings
from +110.6% centroid to −51.6% across five seeds. Every percentage in the
sections below, and in the git history of August 20–21, is one draw.

Longer clips do not fix it. The caption swap stays unstable at every valid
duration measured: REF_pos against neutral reads −8% rms at seed 7 and −64% at
seed 11 at 8s. **Report medians over ≥3 seeds, quote the spread, and never treat
a single-seed caption swap as ground truth.**

**Composed/condmix cells above 8s are invalid — do not quote them.** Past ~200 AR
frames the pipeline denoises in *overlapping* windows and re-encodes the
condition per window; the diagnostic scripts aligned cached conditions from
position 0, so every 12s cell was silently corrupted (the tell: the u=0 identity
fails at 12s and is exact to four digits at 8s). Both scripts now refuse >8s.
An earlier revision of this file cited 12s teacher numbers from those cells;
they are withdrawn.

**The caption pair, not the recipe, decides whether a slider works.** Identical
recipe, 3-seed grouped ladders, at slider +1.0:

| pair | level | brightness |
|---|---|---|
| trip-hop ↔ glossy EDM (`R-final`) | **−76.0%** | +46.0% |
| dusty ↔ glossy (`F-dust`) | **−7.5%** | +46.2% |

Same brightness, 10x less level damage, and a symmetric minus side (+6.4% level
/ −13.0% centroid, against the trip-hop pair's +77% level / −8% centroid). Two
sessions of recipe search — loss, clipping, rank, anchors, `--traj_frac`,
`--target_mode`, `--gain_penalty` — moved the level channel by single digits.
The pair moved it by 68 points. Poles that differ in BPM and arrangement put the
edit in content the *AR plan* owns, and the transformer LoRA cannot reach it; it
distils instead into a compounding level term. **Choose pairs that differ in one
production attribute at fixed genre/BPM/arrangement.**

**The mergeable-LoRA constraint is not what limits these sliders.** Composing the
teacher on the conditional CFG branch and merging it into both branches agree to
0.2 rms points at equal post-CFG net strength (nets 1.0 and 1.7 both checked).

**What does limit them, reproducibly:** the teacher's rendered level recovers as
strength rises (median +7% at net 1.7 over five seeds, always at or above
neutral) while every trained LoRA falls monotonically to −86…−99%. Closed-loop
x_t sampling (`--traj_frac`, three settings) moves that by ≤3 points and is not
the mechanism. Aiming each side at its own pole instead of the pos−neg axis is
worse, and is ill-posed bidirectionally — the trainer now refuses that
combination.

**Three retracted claims, so they are not rediscovered:** "the pole-edge teacher
at net 1.0 reproduces the caption swap" is an algebraic identity, not evidence
about targets; "cond-only and both-branch composition differ at net 1.7" came
from comparing a stereo-array `.std()` print against a mono-downmix file score;
and the prompt-pair geometry screen (`cos(edge+, -edge-)`, `even/odd`) returns
≈ −0.73 for every pair tried because the probe's x_t are anchored to latents
generated from the *neutral* caption. Measured at un-anchored x_t the same pairs
spread over 0.26 instead of 0.04 and change rank order. **Do not gate prompt
pairs on the anchored number.**

## Two more candidate metrics, tested and rejected (August 21 2026)

Both were built as cheap collapse early-warnings; both were verified against
the checkpoints with known render verdicts before being trusted, and both
failed. Recorded here so neither is rebuilt.

**`gain_frac` (probe-time fraction of the LoRA delta along `vel_neu`) does not
separate anything.** Hypothesis: pure gain is the one delta component coherent
across all ~50 solve steps, so a rising probe-time gain fraction should predict
a volume-knob or silent render. Measured on the fixed eval probe across 13
checkpoints spanning every verdict: R-final family (renders −76 % rms) 0.160–
0.163, G-space 0.147, **D-pole-uni (renders SILENCE) 0.126**, E-gp05 (−76 %)
0.111, **F-dust-nopen (the accepted slider) 0.107**, G-grit (inaudible) 0.072,
D-pop-uni (song replacement) 0.066, E-gp2 (−71 %) 0.058, G-vintage (no axis)
0.036. No threshold separates good from bad in either direction — the accepted
slider sits between silence and collapse — and the number mostly re-measures
the `--gain_penalty` knob itself (E-gp2, trained with the penalty, scores
lowest while still collapsing −71 %). The mechanism is the same as the probe
cos failure: one-step measurements at unperturbed states cannot see a
compounding property. The code was deleted, not surfaced.

**`trajectory_cos` (end-of-solve cosine against a live-composed teacher) ranks
silence above working sliders — even within one caption pair.** The metric
solves 50 deterministic steps for neutral / teacher / LoRA from shared noise
and scores `cos(x_lora−x_neu, x_tea−x_neu)`. Cross-pair it was already known
to invert (F-dust-nopen 0.31–0.35 vs D-pole-uni 0.53–0.68). The rescue
hypothesis — that within a FIXED pair, where every variant chases the same
teacher, it becomes a valid training-progress signal — was tested on six
trip-hop-pair variants and fails the same way: at matched multiplier
**D-pole-uni (silence) scores highest** (0.70 at m=1 vs E-gp2 0.566, E-gp05
0.555, R-final-s500 0.519), because the teacher's endpoint displacement on
this pair is itself gain-dominated and total collapse is the fastest way to
move far along it. `cos_at_matched` (gain-corrected) demotes silence to last
but then ranks **D-pop-uni — a complete song replacement — in a tie for
first** (0.5658 vs E-gp2's 0.5659). Among the three mild same-family variants
it does order correctly (E-gp2 > E-gp05 ≈ R-final, matching the paired render
ordering), but a signal that inverts on exactly the catastrophic modes it was
built to catch cannot rank, gate, or early-abort anything. Deleted; the
measurements live in this section. On the dust pair it also resolved a 0.03
"difference" between two variants whose rendered ladders are certified
identical (paired diff 0.001 ± 0.011 ln) — it measures fit-to-teacher, and
fit-to-teacher is not render quality.

**What replaced them: the paired comparison.** With every seed frozen
(train/init seed, cond seed, x0 anchors, eval seed, render seeds), recipe
effects that are invisible unpaired become certifiable: on the trip-hop pair
the between-seed sd of the +1 rms delta is ~1.0 ln, yet the paired per-seed
difference E-gp2 − R-final is +0.46/+0.11/+0.17 — sign-consistent 3/3, mean
+0.25, paired sd 0.18 (a 5–20x noise reduction). The dust pair's gp05-vs-nopen
"identical ladders" claim is confirmed at 0.001 ± 0.011 ln the same way. This
is the design basis of `scripts/slider_pipeline.py`.

## Why transformer `cos` stops near 0.5, and the comparable metric (August 2026)

Transformer sliders all plateau at last-step `cos` 0.43–0.55 while LM halves reach
0.90–0.97. That gap is **not** the transformer slider being weak — it is the metric.
Measured on triphop-tf-v3's own conditions and x0 anchors:

**The target is not reachable by any conditioning.** `cos` scores the LoRA delta
against `vel_pos − vel_neg`. Literally swapping the neutral caption for the
positive-pole caption — the ground-truth edit — scores only:

| t | 0.10 | 0.20 | 0.30 | 0.50 | 0.70 | 0.90 |
|---|---|---|---|---|---|---|
| `cos(vel_pos − vel_neu, axis)` | 0.255 | 0.266 | 0.352 | 0.394 | 0.479 | 0.554 |
| trained LoRA `cos` | 0.549 | 0.553 | 0.537 | 0.500 | 0.493 | 0.480 |

`‖vel_pos − vel_neu‖` is 1.0–2.1x `‖vel_pos − vel_neg‖`: most of a real caption move
is *off* the pole-to-pole axis, and that shared component cancels in `pos − neg`
but not when you can only move one way from neutral. The trained LoRA already beats
the real caption swap. **Do not read `cos` as slider strength, and do not chase 1.0
on it** — 1.0 means synthesizing a velocity field no prompt produces.

**The axis is mostly not a direction.** Mean pairwise cos between targets across
(t, ε) is 0.32; the top singular direction holds 37% of the energy. Same t, two
noise draws: 0.31–0.44. Same seed, t=0.5 vs t=0.2: 0.50; vs t=0.05: 0.16. So the
best possible *constant* delta scores ≈0.57 — the shipped 0.50 is below even that,
i.e. there is headroom before anything exotic is needed.

Other measured facts: `‖vel_pos − vel_neg‖` is only 2–5% of `‖vel_neu‖`, falling
from 0.37 at t=0.05 to 0.023 at t=0.97; bf16 teacher passes cost real signal
(`cos(dir_bf16, dir_fp32) = 0.965` at t=0.5, worse where the delta is smaller — an
fp32 teacher copy costs 9 GB and 2x teacher time, so it was not taken); and cos
peaks exactly at the trained multiplier (m=1 → 0.500, m=2 → 0.474, m=4 → 0.402),
so saturation is not the cap. Condition-space deltas are a measured dead end:
`c_neu + 2·(c_pos − c_neg)` reaches mean cos 0.49 — the same as the LoRA — and a
frame-mean or rank-1 condition direction only 0.23–0.28.

### The fixed eval probe — use this to compare runs

The per-step `cos` in the train jsonl is one random (t, ε) draw out of a field whose
target norm spans ~200x across t. It is far too noisy to rank runs by. Every run now
also logs an `eval` block against a **fixed probe**: 9 timesteps x 2 pinned noise
draws, anchored to the row's x0 bank, seeded by `--eval_seed` (default 1234) and
independent of the training RNG.

```bash
# during training (default: every 50 steps; --eval_every 0 disables)
--eval_every 50 --eval_seed 1234

# rank finished runs
python scripts/compare_slider_runs.py models/*/[a-z]*_train.jsonl --by_t

# score checkpoints trained before the probe existed (v3/v4 have no eval blocks)
python scripts/eval_slider_probe.py models/triphop-tf-v3/*_attn_last.safetensors \
  --prompts_file conceptmod/textsliders/data/prompts-triphop-v3-single.yaml \
  --cache_dir cache/triphop-v3 --by_t
```

`cos` is axis tracking, `mag` is `‖delta(+1)‖ / ‖guidance·(vel_pos − vel_neg)‖`
(whether the axis is actually *reached*), `collapse` should sit near −1. The probe
reproduces the recipe A/B from one shot, with none of the per-step noise:

| checkpoint | cos | cos_neg | collapse | mag |
|---|--:|--:|--:|--:|
| triphop-tf-v3 (anchor, 500 steps) | 0.482 | 0.476 | −0.853 | 0.547 |
| triphop-ab-anchor-single (250) | 0.436 | 0.436 | −0.886 | 0.533 |
| triphop-ab-noise (250) | 0.094 | 0.033 | +0.254 | 0.133 |

The probe axis is signed by the row's `action`, so `erase` prompt files report a
positive cos for a working slider — the old per-step metric reported those negated.

### Trainer fixes that came out of this

- **`--loss {mse,nmse,cos}`, default `nmse`.** Plain MSE tracks `‖axis‖²`, which
  spans ~200x across t, so a handful of low-t steps own the run: in the shipped
  triphop log the **top 5% of steps carry 73% of total MSE mass** (max 32.4 vs
  median 0.13). `nmse` divides by per-step target energy; `cos` optimizes the
  reported metric directly plus a `--mag_weight` magnitude term.
- **`--grad_clip {norm,value}`, default `norm`.** `clip_grad_value_` clamps each
  element to ±1, so those 250x outlier steps degenerated into sign-like updates.
- **Deterministic data stream.** `x_t`/`t` now come from a dedicated generator
  seeded by `--seed`. LoRA init consumes global RNG in proportion to rank and
  module count, so two runs differing only in `--rank` or `--targets` used to see
  different (t, ε) sequences and could not be compared step for step.
- `calibrate_scale.py` no longer KeyErrors on `x0_*.pt` anchors sharing a cache dir
  (the scratch-dir workaround noted above is no longer needed).

To reproduce the v3/v4 recipe exactly, pass `--loss mse --grad_clip value`.

### Loss A/B, measured (triphop single row, 250 steps, identical data stream)

| run | cos | mag | proj_abs |
|---|--:|--:|--:|
| **`--loss nmse --grad_clip norm`** | **0.5839** | 0.500 | 0.2724 |
| `--loss cos --grad_clip norm` | 0.5123 | 0.636 | 0.2838 |
| `--loss mse --grad_clip norm` | 0.4718 | 0.541 | 0.2515 |
| `--loss mse --grad_clip value` (v3/v4 recipe) | 0.4593 | 0.548 | 0.2469 |
| triphop-tf-v3, for scale (500 steps) | 0.5083 | 0.562 | 0.3033 |

The clip fix alone is worth +0.013 cos; the loss carries +0.112. `nmse` at 250
steps beats the shipped 500-step v3 checkpoint on cos.

**Rank by `cos`, not `proj_abs`.** The two disagree here (cos likes `nmse`,
proj_abs likes `cos`-loss), and the tiebreak is that **`mag` is recoverable and
`cos` is not**: `unit_scale` calibration scales a weak slider back up, and cos
holds to 0.474 out to multiplier 2, but no gain fixes a delta pointing the wrong
way. Use `proj_abs` as the guard that a high-cos run is not inaudibly weak.
(This is also why the naive "MSE is the theoretically right weighting because
`delta_x_final = integral of delta_v dt` is uniform-in-t and absolute" argument
does not decide it — absolute magnitude is normalized away downstream.)

`--loss cos` is available but not recommended as a default: it is definitionally
matched to the reported metric, and cos discards magnitude, so it needs the
`--mag_weight` patch to stay audible at all.

### What limits cos: capacity x loss together, not expressivity

Fresh rank-8/rank-64 LoRAs trained against a cached teacher set, 300 steps,
reporting train cos and held-out cos on 8 unseen instances:

| cell | train cos | held cos | mag |
|---|--:|--:|--:|
| **one fixed (t=0.5, eps)** | | | |
| rank8 attn, mse | 0.834 | 0.517 | 0.60 |
| rank8 attn, cos | 0.850 | 0.489 | 0.28 |
| rank64 full, cos | **0.991** | 0.503 | 0.55 |
| **32 random (t, eps)** | | | |
| rank8 attn, mse (v3/v4 recipe) | 0.443 | 0.386 | 0.53 |
| rank8 attn, **nmse** | 0.600 | **0.552** | 0.56 |
| rank8 attn, cos | 0.371 | 0.308 | 0.24 |
| rank8 full, cos | 0.427 | 0.378 | 0.36 |
| rank64 full, cos | 0.573 | 0.518 | 0.40 |

A rank-64 `full` adapter fits a **single** (t, eps) instance to cos 0.991, so the
parameterization can represent the target essentially exactly at a point. Across
32 instances train ≈ held everywhere (0.60 -> 0.55, 0.57 -> 0.52), so the residual
gap is **underfitting the field, not overfitting the sample**.

**Capacity and loss multiply — do not read either row alone.** The rank-64 cell
above uses the handicapped `cos` loss, and on that basis capacity looks useless
(0.518, below rank-8 nmse at 0.552). Pairing capacity with the good loss on the
shipped path says the opposite:

| 250-step shipped-path run | cos | mag | proj_abs |
|---|--:|--:|--:|
| rank8 attn, mse + clip_value (v3/v4) | 0.4593 | 0.548 | 0.2469 |
| rank8 attn, nmse + clip_norm | 0.5839 | 0.500 | 0.2724 |
| **rank64 full, nmse + clip_norm** | **0.7712** | **0.731** | **0.5458** |

`--rank 64 --targets full --loss nmse` reaches **0.77** — far past the ~0.57
best-constant-direction bound, and double the axis displacement of the rank-8
cell. So a static weight delta *can* track the instance-specific component; it
just needs both the capacity and a loss that does not spend it on the low-t
steps. Its per-timestep profile is also much flatter and peaks mid-range
(t=0.1: 0.671, t=0.5: 0.807, t=0.9: 0.732).

Cost: 222 modules at rank 64 is a **420 MB** checkpoint versus ~19 MB at rank 8
attn — the reason to check whether rank 16/32 captures most of the gain before
shipping this.

`--loss cos` collapses magnitude (mag 0.16-0.40 vs 0.53 for mse), the Goodhart
failure it was predicted to have: cos is scale-free, and `--mag_weight` only
partly compensates.

### `--targets full` emitted double-delimiter keys (fixed)

`TARGET_REPLACE_FULL` includes the root `MiniMaxMusic3Transformer1DModel`, whose
`named_modules()` name is `""`. `lora_name = prefix + "." + name + "." + child`
then produced `lora_unet..transformer_blocks…` -> **`lora_unet--…`**. Because the
v3 module-identity dedupe lets the root claim every module first, *all* 222 keys
came out doubled — disagreeing with the single-delimiter names `--targets attn`
gives the very same attention modules, and with the ComfyUI converter, which
matches `lora_unet-`. A converted `full` checkpoint would not error; it would
log `lora key not loaded` and apply nothing, exactly the silent-partial-load
failure noted for merged-qkv text encoders.

Nothing shipped is affected: every catalog checkpoint has 144 single-delimiter
modules, and `gender-tf-v2` (438 = 222 doubled + 216 single) is the already
quarantined double-wrap. Note this also means the sliders with `_full_` in their
filenames (energy, distortion, tempo, space) contain **attention only** — so the
old "full beats attn" A/B was comparing two attention-only checkpoints.

The fix skips empty path segments, and had to land in **both**
`conceptmod/textsliders/lora.py` and `app/lora_runtime.py`: each builds the key
independently, and if they disagree the studio silently fails to find a trained
module. `scripts/normalize_lora_keys.py` rewrites pre-fix checkpoints in place.

### The teacher axis is fragile in bf16 — keep the compute path fixed

`vel_pos - vel_neg` is a 3–13% difference of large bf16 numbers, so it is
ill-conditioned. Measured on the triphop row:

- The model is fully deterministic: the same call twice is **bitwise identical**.
- Computing pos/neg **batched together** instead of as two batch-1 forwards
  rotates the axis by cos 0.949 (t=0.3) / 0.888 (t=0.7).
- Merely *constructing* a `LoRANetwork` at multiplier 0 — whose forward adds an
  exact-zero tensor, but shifts downstream kernel selection — rotates it by cos
  0.886 mean, down to 0.73 at t=0.9. `vel_neu` itself is unaffected (0.9998):
  it is only the small difference that is sensitive.

Consequences: **bf16 alone caps `cos` against the true axis at roughly 0.9**
(the earlier `cos(dir_bf16, dir_fp32) = 0.965` understated it by holding the
batch layout fixed), and any tool that scores a checkpoint must reproduce the
trainer's exact path or it will read ~0.03 low. `scripts/eval_slider_probe.py`
therefore attaches the LoRA wrapper *before* building the probe; with that it
reproduces the in-training numbers to four decimals. Do not "optimize" the
teacher forwards by batching the three conditions together.

## Which host does an axis belong to?

Don't guess — measure first with `scripts/probe_lm_axis_signal.py`, which encodes
each pole's caption through the AR language model and reports how far the plan
moves (`sep = ||pos-neg|| / ||neu||`) and how antisymmetric that move is (`cos`
near 0 is good; high `cos` means both poles shift the same shared way).

| axis | sep | cos | host |
|---|---|---|---|
| rap ↔ slow | 0.32 | 0.04 | LM only (transformer tried: cos_axis 0.08) |
| energy | 0.28 | −0.10 | **both** |
| distortion | 0.23 | 0.03 | **both** |
| tempo | 0.21 | 0.34 | **both** |
| gender | 0.20 | 0.03 | LM only (transformer moves F0 by 1 Hz) |
| trip-hop | 0.20 | 0.28 | **both** |
| space (dry ↔ wet) | 0.17 | **0.70** | transformer only |

Gender works at sep 0.20, so that is roughly the usable floor. Space is the one
axis that is genuinely transformer-only: its poles move the plan mostly in the
*same* direction, i.e. the LM encodes "this caption is about room acoustics"
rather than an opposing dry/wet axis. Reverb is a rendering property.

**Prompt-writing rule this measurement produced:** for LM sliders, *divergent*
poles beat tidy minimal pairs. Rewriting the energy poles as one swapped clause
in an otherwise identical caption dropped sep 0.30 → 0.11 and pushed cos to
0.43, because the shared caption dominates the hidden state. Let genre, BPM and
instrumentation all move with the axis. (Gender is the exception — "male"/"female"
is lexically potent enough to carry a minimal pair.)

Also new: multi-row prompt YAMLs with top-level `plus_label`/`minus_label` and
per-row `attributes:` expansion (pin gender on style axes and vice versa),
`--cond_seeds 7,17` for AR-take diversity, sidecar v3 (self-describing: kind,
prefix, labels, `unit_scale`, `recommended_range` — inference scripts no longer
hardcode rank/alpha/targets), `calibrate_scale.py` reports
`unit_scale_projected` (axis-component-only) and the CFG-effective delta, and
`scripts/probe_axis.py` gates each listen folder per axis (F0 / onset /
centroid, ref-relative where full-mix metrics are unreliable).

## LM sliders and song endings (August 2026)

A MiniMax Music 3 cut ends when the AR language model samples the
`<|audio_end|>` token; `audio_duration` is only a hard frame cap. LM-half
LoRAs perturb exactly those logits, so stacked LM sliders make the model
measurably less likely to reach the end token in time — the render blows
through the cap and gets guillotined mid-phrase (tail RMS ≈ overall RMS
instead of a fade). Confirmed in the library: on 2026-08-16, most ≥60 s
slider-stack renders sat at exactly the requested duration with hot tails,
while no-slider renders from 08-13/14 mostly ended naturally.

The fix is at training time: `train_lm_slider_music3.py` now carries an
**audio-end regularizer** (`--endreg_weight`, default 1.0). Each prompt row
is pre-rolled once with the pristine base model — the same CFG compose loop
inference runs, LM + RVQ depth decoder only, cached under `cache/endreg/` —
and every training step teacher-forces the LoRA'd model over that
composition and penalizes drift of the *end margin*,
`logit(<|audio_end|>) − logsumexp(semantic-code band)`, at every decode
position (both ±1 poles). The margin is the exact log-odds that decides
stop-vs-continue, so the slider can still move the musical plan but not the
ending. Causality means the prompt-last hidden state in the teacher-forced
forward equals the prompt-only one, so the slider loss itself is untouched.
Watch `edrift_p`/`edrift_n` in the train jsonl (mean |margin drift| in
nats); the sidecar records the final window under `endreg`. Same-seed 60/90s
A/B (`scripts/render_end_ab.py`, library song `5ec87fed`): at 90s the
un-regularized live-v3 +2 blows through the cap (tail/overall 1.045) while
the regularized retrain ends naturally (0.018), like the base model. Every
shipped LM half has since been retrained with the regularizer — see the
next section.

## Structured-caption + end-regularized LM halves: v4/v5 (August 2026)

Every LM slider half was retrained in one pass with both August fixes at once:

1. **Structured Caption prompts.** The studio rewriter only ever emits the
   three-section Structured Caption (Global Metadata / Vocal Details /
   Arrangement), so the old flat `"Genre: X. BPM: Y."` poles were
   off-distribution for the caption the LM actually sees at runtime. The
   rewrite followed the `prompts-rhyme-v4.yaml` precedent: lean skeletons,
   divergent poles (genre/BPM/mood/mix/instruments all move with the axis),
   gender pinned via `attributes`, delivery pinned "Sung, not rapped" except
   where delivery is the axis (rapslow) — and gender itself stays a minimal
   pair. New files: `prompts-{gender,rapslow,triphop,energy,tempo,distortion,
   breath,live}-v4.yaml` (rhyme reuses `prompts-rhyme-v4.yaml`).
2. **Audio-end regularizer** (`--endreg_weight 1.0`, frames 250, seed 7).

Probed signal improved (or held) on every axis vs the v3 table above —
rapslow 0.343/−0.08, energy 0.331/0.03, tempo 0.314/0.06 (cos was 0.34),
distortion 0.269/−0.03, breath 0.268/0.24, live 0.245/0.32, gender
0.235/−0.08, triphop 0.224/0.10, rhyme 0.270/−0.05 (sep/cos).

Training: rank 8, alpha 8, lr 5e-4, 800 steps, symmetric, common_beta 0,
seed-7 endreg pre-rolls, GPU 0 with the studio stopped (the bf16 LM alone is
16 GB — it does not fit beside the studio's 26 GB). Rhyme ran
`--no-early_stop` per its documented tail instability; its last-200-step
window was checked and is stable (loss still falling at step 800, collapse
steady −0.95). Final-50-step windows:

| checkpoint | cos± | collapse | p/n perc | edrift± |
|---|---|---|---|---|
| gender-lm-v4 | 0.97 | −0.95 | 0.23 | 0.060 |
| rapslow-lm-v4 | 0.96 | −0.96 | 0.29 | 0.080 |
| triphop-lm-v4 | 0.97 | −0.95 | 0.25 | 0.059 |
| energy-lm-v4 | 0.96 | −0.97 | 0.29 | 0.072 |
| tempo-lm-v4 | 0.95 | −0.95 | 0.30 | 0.071 |
| distortion-lm-v4 | 0.94 | −0.96 | 0.33 | 0.076 |
| breath-lm-v4 | 0.93 | −0.95 | 0.36 | 0.069 |
| live-lm-v5 | 0.90 | −0.91 | 0.43 | 0.068 |
| rhyme-lm-v5 | 0.94 | −0.95 | 0.34 | 0.070 |

perc runs higher than the un-regularized v3 numbers by construction — the
end-margin term competes with the slider target — and live matches its
endreg-only v4 profile almost exactly, so the structured prompts cost
nothing there. `app/sliders.json` now points every LM component at these
nine (live-lm-v4 and rhyme-lm-v4, each carrying only one of the two fixes,
are superseded; v3 halves are retained on disk but unshipped). All sidecars
are unit_scale 1.0 — drop-in at strength 1, no recalibration.

Ending A/B for all nine at +2 (90s, song `5ec87fed`, seed 7):
`eval/listen/v4-endreg-ab-90s/`. Eight of nine end naturally short of the
cap (base 76.7s, sliders 62–85s). rhyme-v5 +2 hit the cap on that one seed
(tail/overall 1.64) — endings are sampled and only one seed was rendered, so
sweep more seeds before drawing a conclusion if long rhyme renders truncate
in practice.

## Dust LM campaign: the ladder gates do not transfer to LM hosts (August 21 2026)

Trained the LM half for the dust pair (`prompts-cand-dust-v1.yaml`, the pair of
the shipped `dust-tf-v1` = overnight `F-dust-nopen`) with the v4/v5 recipe plus
a `--seed` arg added to `train_lm_slider_music3.py` (LoRA init was the only
unseeded RNG). Seven runs, all with healthy training metrics — cos± 0.93–0.98,
collapse −0.82…−0.95, edrift ≤ 0.05:

| run | delta vs recipe | steps | cos± | collapse |
|---|---|---:|---|---|
| dust-lm-v1 | baseline (seed 7) | 719 | 0.98 | −0.95 |
| dust-lm-v1-s101 / -s202 | floor replicates | 800 / 657 | 0.98 | −0.95 |
| dust-lm-v1-planreg03 / -planreg10 | `--planreg_weight 0.3/1.0` | 709 / 800 | 0.98 | −0.95 |
| dust-lm-v1-ts05 | `--target_scale 0.5` | 800 | 0.93 | −0.82 |
| dust-lm-v1-r4 | rank 4 / alpha 4 | 542 | 0.98 | −0.95 |

**Every one was vetoed by the SCORING.md ladder gates, and so was the
control.** Rendered paired ladders (seeds 7/23/77, ±1 ladder + ±2 over-range)
at 4 s and again at 12 s, scored with `slider_pipeline.gates` (axis frozen to
`centroid:+1`, the cond-interp-certified dust direction): G2 level passes at
12 s, but G3 direction (rho −0.64…+0.57), G4 identity (onset corr 0.05–0.13 at
*every* scale, no dose-response toward 0) and G6 null (E_min 0.00–0.41 vs E95
0.48) fail for all seven. The contestants were designed attacks on those
gates: `--planreg_weight` penalizes hidden drift along the teacher-forced
composition (it halves frame-position drift by construction — pdrift 0.0014 →
0.0006 at weight 1.0), `--target_scale` trains a cooler unit, rank 4 cuts
capacity. None moved rendered identity, because the identity channel is
**sampled-token divergence**: any prompt-last displacement flips early AR
tokens and the composition re-rolls; teacher-forced hidden drift is not the
mechanism.

The decisive control: **shipped, ears-approved `energy-lm-v4` fails the
identical 12 s bed the same way** (G4 onset corr 0.045–0.13, G3 rho 0.18, G6
E_min 0.00, even G5 trips). The gates were frozen from the transformer
campaign — G4's "genuinely different song scores 0.172" calibration, the 4 s
bed, and the G6 null are all transformer-render facts. LM halves *by design*
change the arrangement (that is why axes moved into the LM), so same-seed
onset/envelope correlation against the scale-0 clip is ~0 for **any working LM
slider**. The instrument is valid for transformer sliders only; scoring an LM
half with it measures the host, not the slider. Consequences:

- Do not gate LM halves with `slider_pipeline` until an LM-specific contract
  exists. The instruments that *are* validated for LM hosts: the axis probe
  (`probe_lm_axis_signal.py`) as the pre-training screen, training metrics
  (cos/collapse/edrift), ending behavior (`render_end_ab.py`), and ears
  (LISTEN sets / the blind AB session).
- On the validated screen the dust pair is **weak**: sep 0.136 / cos 0.17 —
  below the ~0.20 usable floor (gender works at 0.20; energy-v4 is 0.33). The
  flat minimal-pair captions are the documented failure mode for LM sliders
  ("divergent poles beat tidy minimal pairs"). If the 40 s listen set shows
  nothing, the remedy is a structured-caption v4-style dust pair — which needs
  human sign-off per the caption policy — not more recipe search. Seven
  recipes moved nothing; the pair is the lever, same as the TF campaign found.
- Deliverables kept: `models/dust-lm-v1` (+ variants), listen sets under
  `eval/listen/dust-lm-40s/` and `eval/listen/pipeline/dust-lm-{v1,12s}/`
  (scores.json + REPORT.md per bed), control under
  `eval/listen/pipeline/control-energy-lm-12s/`. Nothing registered into
  `app/sliders.json` — dust ships TF-only until ears pass on an LM half.

Tooling added: `--seed` (LM trainer, sidecar-recorded), `--planreg_weight`,
`--target_scale`, `generate_listen.py --accept_short` (default on: an early
`<|audio_end|>` is a natural song ending, kept at first-draw seed; the old
reject/retry behavior is `--no-accept_short`).

## Train

**For recipe/loss comparison sweeps, do not hand-launch trainings — use the
pipeline**, which pins every seed, refuses config drift, renders paired
ladders, applies the SCORING.md gates, and writes a ranked evidence report:

```bash
$PY scripts/slider_pipeline.py selftest        # instrument controls (run first)
$PY scripts/slider_pipeline.py train  slider_pipeline/specs/phase1-loss-triphop.yaml --gpu 0
$PY scripts/slider_pipeline.py render slider_pipeline/specs/phase1-loss-triphop.yaml --gpu 0
$PY scripts/slider_pipeline.py score  slider_pipeline/specs/phase1-loss-triphop.yaml
$PY scripts/slider_pipeline.py report slider_pipeline/specs/phase1-loss-triphop.yaml
# new caption pair? certify its axis BEFORE training (G0):
$PY scripts/slider_pipeline.py onboard --prompts_file ... --cache_dir ... \
  --out_root eval/listen/pipeline/onboard-<pair> --intended centroid:+1
```

One-off trainings (a shipped slider, an LM half) stay manual:

Transformer slider (energy / distortion / tempo / space):

```bash
$PY conceptmod/textsliders/train_lora_music3.py \
  --name energy --rank 8 --alpha 8 --steps 500 --lr 1e-4 \
  --duration 4 --seed 7 --device 0 \
  --prompts_file conceptmod/textsliders/data/prompts-energy.yaml \
  --cache_dir /ml2/music/sliders-conceptmod/cache/energy-v2 \
  --save_dir /ml2/music/sliders-conceptmod/models/energy-slider-v2
```

Language-model slider (gender):

```bash
$PY conceptmod/textsliders/train_lm_slider_music3.py \
  --name gender-lm-v4 \
  --prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml \
  --save_dir /ml2/music/sliders-conceptmod/models/gender-lm-v4 \
  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --device 0
```

Or run the sequential helper (skips existing last.safetensors and valid wavs):

```bash
./scripts/run_gpu0_sliders.sh all          # train missing + 20s demos + verify
./scripts/run_gpu0_sliders.sh demo-20      # 20s listen sets only
./scripts/run_gpu0_sliders.sh verify
FORCE=1 ./scripts/run_gpu0_sliders.sh demo-20   # regenerate wavs
```

## Demo (labeled wavs)

```bash
$PY conceptmod/textsliders/generate_listen.py \
  --weights models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors \
  --prompts_file conceptmod/textsliders/data/prompts-energy.yaml \
  --name energy --plus_label loud --minus_label quiet \
  --kind transformer --rank 8 --alpha 8 \
  --out_dir eval/listen/energy-20s \
  --scales=-2,0,2 --duration 20 --seed 7 --device 0
```

For gender, pass `--kind lm` and the LM weights.

`generate_listen.py` is resume-safe: existing wavs that pass duration and silence checks are skipped. It rejects short or silent output. Play files in order. Slider clips use the **neutral** caption; `REF` clips change the prompt with the slider off.

## ComfyUI

Do not add a second converter in this repo. Shipped sliders use LoRANetwork
names (`lora_unet-transformer_blocks-N-attn-to_q.lora_down.weight`,
`lora_te-model-layers-N-self_attn-q_proj…`), not PEFT. Convert them with
[`scripts/convert_lora_comfyui.py`](https://github.com/mikkel/conceptmod/blob/main/scripts/convert_lora_comfyui.py)
on **[mikkel/conceptmod](https://github.com/mikkel/conceptmod) `main`**.
Music 3 backends (`music3`, `music3_lm`) landed in
[8f865fe](https://github.com/mikkel/conceptmod/commit/8f865fea59e02d439a479d80466196044ed00076);
`dd0c165` is tests-only and still skips LoRANetwork files as
`no lora_A/lora_B keys`. Detection is from `lora_unet-` / `lora_te-` keys.

Krea verification (the Krea map is still derived from conceptmod's loader,
not a known-good ComfyUI LoRA) and further Qwen / Music 3 CLIP work happen
on conceptmod. Do not duplicate them here.

```bash
# clone mikkel/conceptmod at main (8f865fe or later)
python /path/to/conceptmod/scripts/convert_lora_comfyui.py \
  path/to/energy_unit_last.safetensors
```

That writes `energy_unit_last_comfyui.safetensors` beside the original. Keys look like:

```
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_A.weight
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_B.weight
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.alpha
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_out.lora_{A,B}.weight
```

`to_q` / `to_k` / `to_v` are fused into ComfyUI's single `to_qkv` Linear
(block-diagonal, scale preserved) from `comfy/ldm/minimax_music/dit.py`.
Put the file in `ComfyUI/models/loras/` and load it with **Load LoRA** on the
MiniMax Music 3 MODEL. LoRA strength is the slider scale: `0` is off, `±1` is
the trained unit (already baked into `*_unit_last` / current sidecars with
`unit_scale: 1.0`), `±2` is a typical listen-folder pole.

### Language-model sliders

LM sliders convert to ComfyUI's generic CLIP key form from `comfy/lora.py`
`model_lora_keys_clip`:

```
text_encoders.model.layers.N.self_attn.q_proj.lora_A.weight
```

That matches an unmerged Qwen3 text encoder — the Hugging Face `language_model`
the trainers wrap. Load the same file through **Load LoRA** with the CLIP /
MiniMax Music 3 text encoder connected (MODEL strength `0` if the file is
LM-only).

ComfyUI can also load a Music 3 TE that was saved with merged `qkv_proj`.
Those checkpoints have no module for the separate q/k/v adapters, so those
keys will log `lora key not loaded`; `o_proj` still applies. The convert
script does not invent a merged-qkv file: GQA makes `q` 4096-wide and `k`/`v`
1024-wide, and the official trainer never wrote `qkv_proj`.

```bash
python /path/to/conceptmod/scripts/convert_lora_comfyui.py \
  path/to/gender-lm-v4_last.safetensors
```

Tests for the mapping live next to the script:

```bash
cd /path/to/conceptmod
pytest tests/test_convert_lora_comfyui.py tests/test_comfyui_lora_apply.py
```

## Encoder-first note

`train_encoder_music3.py` is the condition-encoder / notrigger analog. Dummy LoRA converges; a single Conv1d cannot fully remap real AR hiddens. Use the LM trainer for identity (gender) and the transformer trainer for production (loud, distortion, tempo, space).
