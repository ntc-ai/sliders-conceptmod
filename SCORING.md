# Slider scoring spec (2026-08-21)

This is the contract for an automated loop that trains sliders and optimizes a
number. It is written to be frozen *before* any search begins. Four independent
reviews fed into it; each found a defect the others missed, and every clause
below traces to a measurement, not a preference.

## Why this is not a single scalar

Every single-scalar family was attacked empirically, using checkpoints this
campaign had already trained — the loop's degenerate solutions are not
hypothetical, they are on disk:

| scalar family | what actually maximizes it | measured |
|---|---|---|
| effect size along a caption direction | digital near-silence | `D-pole-uni` (rms 0.0019 vs 0.104) scores **+6.86** vs the good slider's +1.34 |
| intended / unintended ratio | a do-nothing slider, or hiss | 14-20 kHz noise at 2% energy: centroid **+244%** at rms +1.0%, ratio ~240 vs a real slider's 6.1 |
| distance to the caption-swap render | a no-op LoRA, by seed lottery | the **zero clip beats the +1 clip** in many cells |
| monotonicity / symmetry composite | a volume knob | every collapse checkpoint scores Spearman **1.000**, tied with the good slider |

The mechanism is the same each time: **gain is the cheapest globally-consistent
rank-8 delta, and it compounds over ~50 open-loop Euler steps.** Level collapse
is not an adversary finding an exploit; it is what gradient descent finds by
default. So rejection modes must be *vetoes*, never terms a strong effect can
trade against.

## Gates — any failure rejects, no trade-offs

- **G0 unscoreable.** The concept has no certifiable acoustic direction
  (sign-flip permutation test over pooled axis spans). A prompt problem, not a
  training problem. `G-vintage` fails this at p=0.75.
- **G1 silence floor.** rms >= 0.02x the folder's own scale-0 clip, plus an
  absolute dBFS floor.
- **G2 level containment.** median |d rms| <= 8 dB, worst seed <= 14 dB. Wide
  enough that the perfect teacher's -55% mid-range dip passes; the -76%
  trip-hop collapse and the +197% minus-side explosion do not.
- **G3 direction.** Spearman rho >= 0.8 on the median curve, and endpoint sign
  agreement >= 2/3 seeds on *each* side. Catches both reversed sliders and a
  dead pole.
- **G4 song identity — the gate nothing else catches.** Onset/envelope/chroma
  correlation against the *same seed's* scale-0 clip, >= 0.9 at every rendered
  setting. A genuinely different song scores **0.172** (measured from cross-seed
  renders). `D-pop-uni` at +1.0 scores **0.130** while reading healthy on every
  spectral feature (rms +4.9%, centroid -6.5%, both inside seed noise) — it had
  discarded the song and rendered a new one. A loop on spectral features alone
  *prefers* that clip to a merely-quiet one. Average-spectrum similarity reads
  **0.87 between completely different songs**, so no amount of spectral
  statistics can substitute for this.
- **G5 artefact guardrails.** Spectral flatness / hi4k excursion (the hiss
  attack's tell is flatness +46%), roughness fraction, crest. Per-axis
  must-not-move lists: the nuisance set differs by concept.
- **G6 null calibration.** A candidate must beat the 95th percentile of the
  score distribution obtained from *zero-slider re-renders at different seeds*.
  Without this, seed noise alone manufactures apparent progress — which is
  exactly how one session of this campaign was spent.
- **G7 held out.** Durations beyond the training bed, over-range multipliers,
  and harness-owned neutral captions the optimizer never sees.

## The ranking scalar, among gate-passers

On-axis **audible steps per unit of scale**: per-clip features in ln space
(rms, *magnitude*-spectrum centroid, hi4k, flatness, crest, stereo width,
spectral flux), each delta taken against the folder's own scale-0 clip and
divided by a fixed "clearly audible step" constant (~3 dB rms, ~20% centroid,
...), projected onto the concept direction, median across seeds, attenuated by
off-axis excess, squashed as `E/(E+2)`. Report `min(plus side, minus side)` so a
dead pole cannot be averaged away. 0.5 means two audible steps per unit on the
weaker side.

Three implementation rulings, each of which changed the ranking:

1. **The axis comes from condition-interpolation ladders, not caption swaps.**
   Re-aiming dust at its cond-interp axis turned both dust runs from "backwards"
   into textbook monotone (rho +1.00, endpoint sign 10/10) and dropped energy's
   off-axis residual 2.2 -> 0.8. The sliders were never off-axis; the REF axes
   were mis-aimed. Certify 3-5 cond-interp seeds per concept at onboarding.
2. **Never normalize by the caption swap's magnitude.** Over 10 seeds the dust
   swap's centroid is -50%,-25%,-11%,-9%,-5%,+31%,+42%,+48%,+55%,+92%: median
   +13%, sd 41. Magnitude is unmeasurable; only its *direction* is certifiable.
   Fixed audibility constants supply the scale instead.
3. **Magnitude spectrum, not power**; and one rms convention everywhere. Power
   centroid buries the audible dust crackle (+55% magnitude vs -11% power on the
   same clips), and three different rms conventions are live in this repo
   (per-channel power, mono downmix, sqrt-mean-square with DC) disagreeing up to
   40% on the same files through phase cancellation.

## Policy — the optimizer must not grade its own homework

Caption wording is the strongest knob available: identical recipe, captions
alone swung the level channel **68 points**. Therefore the measurement spec is
frozen before caption search starts; the scorer never sees optimizer-written
captions; caption diffs need human sign-off; and full gate vectors are logged
every run so a Goodhart walk is visible early rather than at the end.

## Validation against ears

A blind 2AFC session is built and live at `eval/listen/abtest/` (107 trials,
~20 min). Every trial compares a slider clip against **the same seed's zero
clip** — the only comparison seed instability does not poison. Chance is exactly
50%, so "inaudible" and "backwards" fall out of the same trials. Controls:
forced full playback before answering, 6 repeats for self-consistency, 4
reference-vs-reference trials, and 2 synthetic broken clips that must be rated
broken or the session is void.

Per run, 9/11 correct certifies audible (p=.033) and <=2/11 certifies reversed.
**Responses are metric-agnostic**: listen once, and every future candidate score
is validated against those judgements for free. Ongoing, a ~12-trial 3-minute
sentinel per promoted batch, including deliberately low-scoring candidates —
without those, discrimination is unmeasurable.

## Known blind spots, stated plainly

Concept versus proxy (an EQ tilt and real "gloss" are the same vector in these
features); musical damage at constant level and spectrum beyond what G4 catches;
anything past one denoise window (~8 s) unless explicitly rendered; slider
stacking; and the audibility constants are judgement calls, not measurements.

## Amendments from the pipeline build (2026-08-21, evening)

Implemented in `scripts/slider_pipeline.py` + `slider_pipeline/`; the goodhart
attack table above is now an executable test (`slider_pipeline.py selftest`
feeds synthetic collapse / silence / hiss / song-replace / no-op / reversed /
dead-pole ladders to the gates and asserts every veto fires).

1. **Phase-1 ranking is a PAIRED design on frozen seeds, and that is a gate on
   the experiment, not a preference.** Measured on disk: between-seed sd of the
   +1 rms delta on the trip-hop pair is ~1.0 ln while every recipe effect ever
   measured is <= 0.25 ln — unpaired ranking is unfalsifiable. Paired on the
   same three seeds, E-gp2 - R-final = +0.25 mean, sd 0.18, sign-consistent
   3/3; the dust pair's two variants tie at 0.001 +- 0.011. The pipeline
   refuses specs with unpinned seeds, checks the config echo of every adopted
   checkpoint, and verifies zero-scale clips agree across variants (they are
   slider-off renders of the same seed, so divergence = broken pairing).
2. **The one rms convention is per-channel-power rms** (`sqrt(mean(all
   samples^2))` over the de-interleaved array), per the blind-spot audit:
   mono-downmix rms mixes level with stereo width through phase cancellation
   (up to 40 % on real renders). Mono rms and `mono_cancel_db` stay logged.
3. **G6's null distribution has a concrete construction**: pseudo-candidates
   whose "+1 clip" is a different seed's zero-scale render, all assignments of
   substitute seeds, 95th percentile of the resulting E. Zero-only null seeds
   are rendered per sweep because three compare seeds alone give too few
   combinations.
4. **Two candidate cheap signals were tested against ground truth and deleted**
   rather than shipped: probe-time `gain_frac` (no threshold separates the
   accepted slider from silence; mostly re-measures the gain_penalty knob) and
   `trajectory_cos` (ranks silence top at matched multiplier even within one
   pair; gain-corrected variant ranks a song replacement in a tie for first).
   Details in MUSIC3.md "Two more candidate metrics, tested and rejected".
   Consequence: there is currently NO verified in-training early-abort signal;
   the cheapest honest failure detector remains a rendered ladder.
5. **G7 is split by cost**: the over-range multiplier check renders with every
   ladder (no silence beyond the fit point); long-duration (20 s windowed) and
   held-out-seed confirmation run on the promoted winner only (`confirm`
   stage), before anything is called a winner out loud. Held-out seeds are
   named in the spec and never touched by ranking.

## Amendment 6 — scope: transformer sliders only (2026-08-21, night)

The contract was frozen on transformer-slider attacks, and a control run shows
it does not generalize to LM halves: shipped, ears-approved `energy-lm-v4`,
rendered through the identical paired 12 s ladder bed, fails G3/G4/G6 exactly
like seven deliberately varied dust LM candidates (onset corr 0.05–0.13 at
every scale; details in MUSIC3.md "Dust LM campaign"). The mechanism: LM
halves change the arrangement by design, so same-seed onset/envelope identity
against the scale-0 clip is ~0 for any working LM slider, and the G6 null is
built from the same seed-noise structure. **Do not gate or rank LM halves with
this instrument** — it measures the host, not the slider. An LM-specific
contract (identity defined against arrangement-preserving features, null from
re-rolled plans) must be frozen before any LM optimization loop starts. Until
then LM halves are judged by the axis probe, training metrics, ending A/B, and
ears only.
