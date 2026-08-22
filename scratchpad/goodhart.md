# How the slider loop will cheat the score — red-team findings

2026-08-21. All numbers below are **measured on renders already on disk** — no new
training was needed, because the campaign has already trained almost every
adversarial example a score-hacking loop would find. `scratchpad/score_attacks.py`
reproduces every table from `eval/listen/{curve-gp-4s,pairs-4s,curve-4s}`.
(Note: the two ladder-render jobs running during this analysis were appending
new seed folders — seed303/404/505 landed mid-run — so re-running the script
shifts third decimals; no ranking in this file changes.) Human verdicts used
as ground truth: R-final = reject (−76% rms collapse),
F-dust = accept, G-grit = reject (inaudible), G-energy = loudness pair used as a
wrong-axis ringer, D-pole-uni / R-final-r8-s500 (curve-4s) = reject (hiss / near
silence at high scale).

## 0. Why gain is the universal attractor (read first)

Every attack below reduces to one mechanism, so name it once. The training
target `v_neu + 3(v_pos − v_neg)` contains a shared loudness/brightness
component whenever the poles differ at all in level — and almost every caption
pair does (O3: five of six pairs move rms double digits). A rank-8 attention
LoRA's cheapest *globally consistent* delta is a gain/tilt on activations; the
axis-specific content is instance-dependent (mean pairwise target cos 0.32,
MUSIC3.md) and much harder to fit. At render the delta acts open-loop on both
CFG branches for ~50 Euler steps, so a small per-step gain bias compounds:
observed endpoints are −96..−99% rms on + and +197/+446% on − (O4), while the
teacher itself *recovers* level with strength. Gain is therefore not an exotic
adversary: it is the **default local minimum**, and any score with a magnitude
term walks straight into it. Conversely "do nothing" is the second attractor:
a near-zero delta is perfectly symmetric, monotone-trivial, and side-effect
free. A useful score must be repelled by *both* attractors simultaneously.

## (a)+(b) The score families and their degenerate maximizers

### F1. Effect size along a caption-derived direction
(any "Δfeature at +1", projection on REF_pos−REF_neg in feature space,
"distance moved from base")

**Degenerate solution: level collapse into hiss.** rms enters every feature
(directly, and via centroid/hi4k of the hiss floor once music is gone), and the
caption direction itself has a level component, so the −99% checkpoint projects
enormously. Measured, z-scored log-feature space, pool-whitened, direction from
each folder's own REF pair:

| checkpoint | S1 effect size | reality |
|---|--:|---|
| D-pole-uni @+1.2 | **+6.86** | rms −99.4%, hiss |
| R-final-r8-s500 @+1.5 | +6.82 | rms −98.9% |
| R-final @+1 (3-seed med) | +1.92 | reject, collapse |
| **F-dust @+1 (accepted)** | **+1.34** | the actual good slider |
| G-grit | +0.05 | inaudible |

The score prefers digital near-silence over the accepted slider **5:1**, and
prefers the rejected R-final over the accepted F-dust. **Reachable: it is the
observed default failure** — no adversary needed, the loop's first collapse run
wins immediately and permanently.

Sub-family: *effect size at train states* (probe cos/mag). Already Goodharted
in the wild: O1, cos 0.52→0.81 with zero render change. Mechanism: the
optimizer controls the state distribution (`--traj_frac`, anchor mode, eval
seed), and probe states are unperturbed trajectories — exactly where an
open-loop adapter looks fine. Any metric computed at states the trainer
samples is optimizer-ownable. **Demonstrated reachable.**

### F2. Ratio of intended to unintended change
(e.g. |Δcentroid| / |Δrms|, or any "purity" quotient)

**Degenerate solution A: the do-(almost-)nothing slider.** The ratio has no
effect-size floor; as the delta → 0 the quotient becomes 0/0 seed noise and a
micro-EQ with lucky seeds wins. Measured: inaudible G-grit (cen +5.8%) already
ranks above every WEAK-but-audible slider and within noise of the accepted
pair's runner-up config (naive ratio 2.92 vs F-dust-nopen 3.03).

**Degenerate solution B: high-frequency noise injection.** Adding 14–20 kHz
noise at **2% of signal energy** to the accepted pair's own neutral clip moves
centroid **+244% at rms +1.0%** — ratio ≈ 240 versus the accepted slider's 6.1.
A ratio score is beaten 40:1 by faint hiss that any musician rejects on first
listen. To merely match the caption-swap brightness target (+55%) costs ~0.4%
energy, rms +0.2%. **Reachable:** stationary broadband texture is exactly what
these LoRAs produce easily (the collapse checkpoints' outputs *are* hiss; the
distortion slider is deliberate texture injection). The tell: spectral flatness
+46% and hi4k +5.6%..+45% — see gate G4.

**Degenerate solution C: wrong-axis purity.** "Unintended" must be defined
per-axis or a loudness slider wins a brightness contest: G-energy scores 2.96,
statistically tied with the accepted dust pair on the naive ratio.

### F3. Distance to the caption-swap render (feature or waveform space)

**Degenerate solution: seed-lottery harvesting + do-nothing.** The target point
is noise (O2: centroid +110.6%..−51.6%, rms −15.5%..−69.3% across five seeds;
in F-dust-gp05-seed7 the swap reads rms **+33.9%** where the 5-seed median is
≈ −15%). Measured consequences on the existing pool:

- The per-seed winner is a different run on nearly every seed: G-energy wins
  seeds 11 and 23, G-vintage wins seeds 7 and 77, F-dust-nopen wins 42 and 101.
  A loop scoring on ≤2 seeds selects a coin flip and then optimizes it.
- The **zero clip beats the slider's +1 clip** in many cells (E-gp05 seed7:
  distance 4.19 doing nothing vs 7.53 with the slider; G-energy seed11:
  1.26 vs 1.52). A no-op LoRA is near-optimal whenever the seed idiosyncrasy
  dominates, so the loop converges to weak sliders that surf seed luck.
- With k candidates × few seeds this is a multiple-comparisons machine: recipe
  knobs that only reshuffle noise will show monotone "progress" (exactly the
  O1 sweep's shape).

**Reachable without any adversarial LoRA at all** — the selection loop commits
this attack by itself. Waveform-space distances (spectrogram L2, MSS loss) are
strictly worse: they are dominated by the seed's arrangement, so the argmin is
"whatever render happens to share the seed's drum pattern".

### F4. Classifier / embedding similarity (CLAP-style, or a trained pos/neg probe)

Partially theoretical here (no CLAP/MERT in the env, hub offline), but the
attack surface is well characterized:

- **Nuisance-axis surfing (reachable).** Audio-text embeddings' dominant
  cheap directions are loudness, brightness, and density. The F1/F2 solutions
  (gain, tilt, hiss) move embedding similarity for words like "glossy",
  "bright", "lo-fi", "dusty" without any musical content change. A LoRA that
  adds vinyl-crackle texture maximizes sim("dusty") — texture injection is
  demonstrated in-class for these LoRAs.
- **OOD hallucination (reachable).** Near-silence and hiss are off-manifold
  for the embedding; cosine there is arbitrary and sometimes high. The
  collapse attractor is therefore not reliably punished by an embedding score
  — the one family where silence might *win* rather than lose.
- **Trained-probe leakage (reachable).** A pos/neg classifier trained on
  caption-swap renders learns the level/brightness confound of O3 (five of six
  pairs move rms and centroid together), i.e. it *is* an rms+centroid score
  with extra steps, inheriting F1's failure.
- **Gradient adversarial perturbations (theoretical).** The loop has no
  gradient through render→embedding, and renderer stochasticity breaks
  transfer of tiny perturbations; black-box search over checkpoints finds only
  the coarse texture attacks above — which are already enough.

### F5. Monotonicity / symmetry composites

**Degenerate solution: the gain knob.** Level collapse is the most monotone,
most antisymmetric object in the system. Measured: every collapse checkpoint
scores |Spearman(scale, projection)| = 1.000 (ties with the accepted slider —
the metric cannot separate them), and R-final's ladder (−76% at +1, +199% at
−0.5) is a textbook monotone antisymmetric curve. Meanwhile the **symmetry**
term is maximized by weakness: measured winners are G-vintage (weak) and
G-grit (inaudible) because near-zero deltas mirror perfectly. A composite of
effect size + monotonicity + symmetry describes a volume knob exactly.
**Reachable: trained repeatedly by accident.**

### F6. Match-then-measure (the current `score_render_curve` scheme, for honesty)

"Find the scale where the intended feature matches GT, score the nuisance error
there" is reparameterization-invariant (immune to alpha/unit_scale inflation —
important, since any fixed-multiplier score is gamed by gain calibration) and
has an implicit audibility gate ("never reaches brightness" fails). Its
residual hole: **reach the intended feature by non-musical means.** The hiss
injection hits any centroid target at ~0 rms cost, so it brightness-matches
with level error ≈ 0 and beats every real slider. The existing rms>−90% hiss
guard does not fire (rms +0.2%). Secondary hole: the GT point is seed noise
(O2), fixed only by multi-seed median targets or a human-fixed band.

### Reachability summary

| attack | status |
|---|---|
| level collapse / explosion (F1, F5) | **observed, default failure** |
| probe-metric hacking via state distribution (F1b) | **observed** (O1) |
| do-nothing / micro-EQ (F2, F3, F5-sym) | trivially reachable; loop finds it via seed luck |
| seed-lottery harvesting (F3, and any small-n score) | **observed shape**; no adversary needed |
| HF-noise / texture injection (F2, F4, F6) | reachable (hiss & texture are demonstrated LoRA outputs); not yet observed as a *selected* optimum because nothing has optimized a ratio score yet |
| wrong-axis loudness pair via captions (F2, F4) | **observed** (G-energy ties the accepted pair) |
| embedding OOD / nuisance surfing (F4) | reachable in weak form; untested locally |
| gradient-adversarial embedding perturbation (F4) | theoretical in this loop |

## (c) Minimum score structure that survives the above

Principles first: **gates are hard and separate from the ranking scalar**; the
scalar ranks only gate-passers; every gate below exists because a specific
measured attack defeats the scalar alone. Everything is computed on the
**shipped render path** (50-step, unit-normalized files, both CFG branches) —
never at train states (O1) — as **medians over ≥3 seeds with the spread
reported**, gating on the worst seed where noted.

- **G1 Level integrity.** At every ladder step in the recommended range, both
  signs: median Δrms within an axis-specific band (default −30%..+50%; wider
  for a loudness axis), absolute rms ≥10% of base (the existing hiss floor).
  Kills F1/F5 collapse and minus-side explosion.
- **G2 Audibility floor.** Median intended-feature move at +1 above an
  axis-specific perceptibility threshold (e.g. centroid ≥ +15%, or
  `probe_axis.py`'s per-axis F0/onset gates). Kills every do-nothing winner
  (F2A, F3, F5-sym).
- **G3 Direction & sign consistency.** Endpoint sign agreement ≥ 3/3 seeds on
  the intended feature; + and − endpoints opposite signs. Kills wrong-way
  (rank-corr −1.0, O4) and non-mirror checkpoints.
- **G4 Off-axis must-not-move list.** Per axis, bands on: rms (non-loudness
  axes), onset rate/tempo (non-tempo axes), **spectral flatness and hi4k**
  (the hiss detector — the F2B attack is +46% flatness at 2% injected energy),
  crest factor, and for LM halves the end-token behavior at ≥60s
  (tail/overall RMS, the endreg lesson) and delivered duration. Kills hiss
  injection and the collateral-damage family.
- **G5 Held-out everything.** Score at ≥2 durations (4s ladder + one 20s);
  at multipliers past the fit point (+1.5/+2 — the silence-above-threshold
  failure, O4); render-time neutral caption drawn from a **fixed harness-owned
  bank** including ≥1 caption never seen in training. Kills train-state
  overfit and caption-conditioned cheats.
- **G6 Monotonicity as a gate, never a reward.** |Spearman| ≥ 0.9 on the
  intended feature required; adding it to the objective selects the volume
  knob (F5).
- **G7 Null calibration (the seed-noise defense).** Before ranking, compute
  the score's null distribution: zero-multiplier renders across seeds, and
  REF re-renders (the z0 distances measured above are exactly this). A
  candidate must beat the null's 95th percentile; score differences smaller
  than the null spread are ties. This is the only thing that stops F3's
  lottery and stops "optimizer progress" that is noise (O1's sweep shape).
- **Ranking scalar**, applied only to gate-passers: match-then-measure (level
  error at brightness match, per `score_render_curve.py`) — it is invariant to
  gain reparameterization, which any fixed-multiplier scalar is not — with the
  GT point replaced by a multi-seed median band or a human-fixed target
  profile (O2 forbids single-seed GT).
- **Visibility over impossibility.** Log the full gate vector for every
  candidate, not the scalar alone: a Goodhart walk shows up as gates trending
  to their boundaries generations before the scalar looks wrong. Final
  acceptance of the shortlist stays a human ear; the score's job is triage
  and making cheating legible. The feature extractor and gate code are frozen
  outside the optimizer's search space.

## (d) The caption channel

Letting the optimizer write the three captions opens four distinct doors:

1. **Axis redefinition — grading its own homework.** If the score's direction
   or GT comes from the optimizer's captions, poles drift toward whatever the
   features reward: loudness/brightness vocabulary (measured: the loudness
   pair G-energy already ties the accepted dust pair on a brightness-purity
   score), or big genre/BPM/arrangement divergence — which puts the edit in
   AR-plan-owned content that a transformer LoRA can only render as gain (the
   R-final mechanism). Captions alone, identical recipe, moved the level
   channel **68 points** (R-final −76% vs F-dust −7.5%): the caption *is* the
   most powerful knob the optimizer holds, more than every recipe knob
   combined.
2. **GT inflation/deflation.** With GT = caption-swap render: pick poles whose
   swap is enormous so fixed tolerances become proportionally trivial — or
   poles with pos≈neg so "distance to GT" is achieved by doing nothing (and
   the training axis degenerates into bf16 noise, which the probe still
   happily scores).
3. **Content exploits.** Artist names (already happened once —
   `models/retired-triphop-slider-artist-names/`, banned in AGENTS.md), lyric
   or production tokens that hijack the AR plan, and caption formats
   off-distribution from the runtime Structured Caption so probes and runtime
   disagree (the v4 lesson, invertible by an adversary).
4. **Probe gaming.** `probe_lm_axis_signal` sep can be inflated by irrelevant
   pole divergence; the anchored pair-geometry screen is already known to be
   blind (MUSIC3.md retraction — it returns ≈ −0.73 for every pair).

Closing the channel:

- **The human fixes the measurement spec before any caption search**: axis
  name, intended feature(s), must-not-move list, target bands. Captions are
  then a training knob that cannot redefine success.
- **The scorer never sees the optimizer's captions.** Evaluation renders use
  the harness-owned neutral bank; if REF swaps are used, their caption text is
  frozen by the harness. Nothing caption-derived enters the score direction.
- **Caption lint + probe gate before training**: banned-term list (artists,
  lyrics), sep in a sane band (~0.15–0.4) with low shared-component cos for
  LM axes, single-production-attribute diffs for TF axes (the O3 lesson),
  Structured-Caption format enforced. Caption diffs are tiny text — cheap to
  human-review, so require sign-off on them even in an otherwise automated
  loop.
- **Incumbent test**: a new caption pair must beat the current best under the
  same frozen scorer and gates on fresh seeds, with G7's null calibration —
  otherwise "caption search progress" is the O1 sweep again.

## What was not run, and why

No adversarial LoRA was trained: both GPUs were occupied by the ladder jobs,
and — the stronger reason — the pool already contains the maximizer of every
score family except F4 (collapse checkpoints for F1/F5, G-grit/G-vintage for
F2A/F5-sym, G-energy for F2C, the seed spread for F3), so training one would
demonstrate reachability of things already observed. The one experiment that
would still add information when a GPU frees: train a slider on
optimizer-style "bright, crisp, airy / dark, dull, muffled" poles with a small
hiss-texture reward and confirm the F2B/F6 hiss attractor is found by SGD, not
just by construction. Second candidate: embed the existing pool with CLAP
(needs the hub online once) and check whether the collapse checkpoints score
above the accepted slider on text similarity — the F4 OOD question.
