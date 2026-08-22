# Listening validation for slider scores

One listener, one ~20-minute session, run once before the training loop trusts any
computed score; then a ~3-minute sentinel check while the loop runs. Built from the
4-second ladders already on disk (`eval/listen/pairs-4s/`, `eval/listen/curve-gp-4s/`).

Tooling:

```
python scripts/build_ab_session.py            # writes eval/listen/abtest/
# listener opens http://localhost:8901/abtest/  (existing http.server, root eval/listen)
# page downloads ab_responses_*.jsonl at the end -> drop into eval/listen/abtest/responses/
python scripts/score_ab_session.py --metric metric.csv [--damage-metric d.csv]
```

Metric CSV: header + rows keyed `run,value`, `run,side,value`, or `clip,value`
(clip = source path substring). Any candidate score can be re-scored against the same
responses forever — **the listening data is metric-agnostic; collect once, correlate
against every future score candidate for free.** That is the main lever against the
listening-time constraint.

---

## 0. What is being validated, and what is deliberately not

Validated: on 4-s clips from six sliders whose measured behaviour spans -76%..+4.5%
rms and -8.6%..+51.4% centroid at +1, does a candidate score rank
(1) hears-the-intended-attribute-move and (2) not-damaged the way one attentive
listener does?

Explicitly out of scope of this session, and unsupported by any result it produces:

- a calibrated mapping score -> audibility ("0.6 means 75% detectable");
- distinctions between two mid-pack sliders;
- transfer to 20-90 s renders (long-form structure can break while 4-s texture is
  fine — after a score passes here, spend one extra 5 minutes on two sliders from a
  `*-20s` folder as a transfer spot check);
- generality beyond this listener.

O1 is the cautionary tale: probe cosine moved 0.52 -> 0.81 across a recipe sweep
while predicting nothing audible. The session is designed to catch exactly that
class of score cheaply, not to certify sliders individually.

## 1. Trial design and why

### Direction: two-alternative forced choice on same-seed pairs

Each trial: clips A and B, same caption, same lyrics, **same denoise seed**, differing
only in merged LoRA scale — one at the ladder extreme (+1, or -1 / -0.5 where that is
the deepest rendered), one at scale 0. Question: "Which clip is more DENSE?" (the
pair's own positive-pole word plus a one-line gloss). Forced answer A or B; the clip
with the higher multiplier is correct by construction.

Why this and not the alternatives:

- **Same-seed pairing is what neutralizes O2.** Seed instability lives entirely in
  comparisons across renders with different conditioning (caption-vs-caption moved
  centroid anywhere from -51.6% to +110.6% on identical captions). Within a ladder
  folder everything shares seed and caption; only the LoRA scale differs, so the
  pair isolates the slider exactly. The plain-caption REF clips are therefore
  **never a scored comparison target** — they appear only as pooled vocabulary
  controls (below).
- **vs ABX:** ABX answers "is there any difference", which is the wrong question —
  several sliders differ hugely (rms -76%) while failing; the product claim is
  *directional*. ABX also costs three listens per judgement instead of two.
- **vs MUSHRA / absolute rating:** one untrained listener's ratings drift and
  anchor; MUSHRA needs a stable reference, which O2 rules out; and a rating scale
  has no objective correct answer, so a guessing or confused listener is invisible.
  2AFC has ground truth: chance is exactly 50%, so "can't hear it" shows up as
  chance and "hears it backwards" as *below* chance — two of the O5 failure modes
  fall out of the same trials at no extra cost.
- **vs pairwise preference:** "which is better" conflates attribute movement with
  damage; we need them separated (part b).
- **vs ranking the whole 7-step ladder:** slow, high working-memory load for 4-s
  clips, and rank errors don't convert cleanly into a per-condition audibility
  number. Monotonicity along the ladder is better checked by machine once the
  endpoint is validated.
- **Sides are tested separately** (+extreme vs 0, and 0 vs -extreme), which is what
  catches "+s and -s are not mirror images": pos audible + neg at chance = broken
  mirror, invisible to any pooled design.
- **No loudness normalization, deliberately.** Level is a legitimate part of some
  attributes (dense/sparse) and of shipped behaviour. Instead every pair's rms
  delta is logged and conditions with >6 dB are flagged LEVEL-SEPARABLE in the
  report: correctness there may be loudness-only, and the damage block judges
  whether that level change is pathological. If a flagged condition matters, a
  loudness-normalized rebuild of just that condition is the follow-up.

### Damage: single-clip 3-point categorical

Damage is an absolute property, not a comparison, so: one clip, forced rating
**1 fine / 2 degraded / 3 broken**, defined on the page ("degraded = audibly worse
than a normal render: too quiet/loud, artifacts, smeared; broken = silent,
collapsed, unusable"). The base model's own renders are inconsistent, so raw flag
rates mean nothing; the block therefore includes blinded **zero-scale clips and REF
renders as a baseline**, and slider damage is measured as flag-rate *above that
baseline* (Fisher exact).

### Blinding and anti-guessing controls

- Clips hardlinked under SHA-hashed names (`abtest/clips/<hash>.wav`); the page
  fetches `session.json` (ids, questions, hashed URLs only); ground truth stays in
  `key.json`, which the page never requests. Blinding is against incidental
  unblinding, not an adversarial listener.
- Trial order shuffled with no run twice in a row; A/B assignment randomized per
  trial; the listener never sees run, side, multiplier, or any metric.
- **Forced full playback:** answer buttons stay disabled until both clips have
  played to the end once. This is the hard attention floor — a zero-effort listener
  cannot exist; response times are logged (floor ~8 s/trial).
- **4 REF-vs-REF vocabulary controls** (positive-caption render vs negative-caption
  render, same seed, scored but analyzed separately): a listener at chance *here*
  either can't hear the axis or the vocabulary doesn't transfer to this model's
  renders — and in that case chance-level slider results are uninterpretable rather
  than evidence of dead sliders. This disambiguation is what the REF clips are for.
- **6 exact repeats** (same pair, re-randomized A/B): self-consistency. A guesser
  self-agrees at 50%; note this check is soft when the repeated condition is
  genuinely inaudible, which is why it is secondary to the two above.
- **2 synthetic broken controls** in the damage block (a real clip at 0.04x — the
  observed "96% of level gone" failure — and digital silence, the other observed
  failure). Both must be rated 3, or the damage ratings are suspect.
- Practice: one labeled REF pair with feedback, unscored, doubles as the
  volume-setting clip.

## 2. The session

| block | trials | content | time |
|---|---|---|---|
| practice | 1 | REF pair, feedback shown | 1 min incl. instructions |
| direction | 60 core | 6 runs x 2 sides x 5 seeds (seeds cycle where only 3 exist) | ~13 min |
| | 6 repeats + 4 REF controls | controls | ~2 min |
| damage | 30 slider/zero | per run: zero x1, +extreme x2, -extreme x2 | ~4 min |
| | 4 REF + 2 synthetic | baseline + broken anchors | ~1 min |

107 scored judgements, ~20 minutes. Listener instructions (on the page): headphones,
set volume once on the practice clip, one listen each is usually enough, **guess when
unsure** — forced guessing is part of the design and shows up as chance, "can't
tell" buttons are deliberately absent.

## 3. Budget split: why ~15 min on direction, ~5 min on damage

Direction gets 2/3 of the clock because it is the machine-blind judgement — nothing
computed currently stands in for "sounds more gritty", and it needs paired trials.

Damage gets 1/3 because it is *partially machine-visible*: every damage incident in
O1 (digital silence, 96% level loss) is a one-line rms check. Consequence for the
loop, independent of any validation outcome: **install a free rms/silence gate**
(render at ±extreme, reject if rms ratio vs zero-scale is outside ~[0.35, 2.0] —
thresholds already used by `build_listen_index.py`). Human damage time then covers
only what rms cannot see (artifacts, smearing, musical incoherence), and the block
doubles as validation of the rms gate itself (`--damage-metric` with per-clip rms
ratios: its Spearman against the ratings is reported).

## 4. What the numbers can and cannot say

All tests exact (binomial / Fisher / permutation), stdlib implementation in
`score_ab_session.py` — the env's scipy is numpy-2 broken.

**Per condition (n = 5-6):** only a perfect 5/5 (p=.031) or 6/6 (p=.016) is
individually significant. Condition accuracies are *inputs to the correlation*, not
certifications. Trials within a condition share the slider and often the seed pair
(3-seed runs), so they are percept-stability replicates, not fully independent
samples; treat per-condition CIs as optimistic.

**Per run (n = 11, sides pooled):** >=9/11 certifies audible (p=.033), 11/11 gives
p=.0005; <=2/11 certifies reversed. Between 4/11 and 8/11 the honest verdict is
"not established", and the report says so. Detecting a *specific* asymmetry (pos
audible, neg chance) needs 5/5 vs <=2/5 — visible only when gross.

**Score correlation (the actual goal):** 12 conditions x ~5.5 trials.

- Condition-level Spearman, permutation p: significance needs |rho| >~ 0.50. Power
  ~95% for a true rho ~0.8, ~65% at 0.6, poor below 0.5.
- Trial-level median split (33 vs 33 trials, Fisher): a 0.90-vs-0.40 accuracy gap
  comes out p<.001 (verified in simulation: planted per-run accuracies
  .95/.9/.8/.7/.5/.5 + a noisy correlated metric -> rho=0.81, p=.003, split 0.91 vs
  0.39, p<.001). A 0.75-vs-0.55 gap is ~coin-flip to reach significance.

So **one session distinguishes exactly three outcomes**:

1. *Adopt:* rho >= 0.5 with permutation p < .05 AND median-split gap >= 25 pp —
   claim: "the score ranks audible sliders above inaudible ones for this listener
   on this material". That is the full extent of the supportable claim.
2. *Reject:* rho <~ 0 or split gap ~0 while at least two runs are certified AUDIBLE
   and at least one is chance/REVERSED (i.e. the *listener* discriminated, the
   score didn't). This is the O1 outcome, caught in 20 minutes.
3. *Ambiguous:* everything else. The remedy is **not** another marathon session; it
   is adopting the score provisionally with the rms gate on, and letting the
   sentinel checks (below) accumulate trials — 24 more sentinel trials roughly
   doubles the trial-level n on the disputed region.

Degenerate case to watch: if fewer than 2 runs are AUDIBLE and REF controls passed,
there is no perceptual spread to correlate against — the finding is "current
sliders are mostly inaudible at these multipliers", which is a training problem,
not a metric problem, and no score can be validated on this material.

**Damage:** 10 baseline vs 24 slider clips detects pooled flag-rate elevations of
roughly >=40 pp; per-run only gross damage (4/4 flagged vs a quiet baseline,
Fisher p~.09-.005 depending on baseline noise). Fine-grained damage scoring is not
established by this session; the synthetic controls plus baseline-relative rates
are enough to validate a *gate*, not a *ranking*.

**Validity gates on all of the above:** synthetic controls rated 3; REF controls
>= 3/4; repeats consistency reported. A failed gate caps every downstream claim,
and the report prints it first.

## 5. The ongoing sentinel check

Cadence: every promoted batch or weekly, whichever comes first. ~12 trials, 3-4
minutes, same page/scorer:

```
python scripts/build_ab_session.py --out eval/listen/abtest-sentinel-<date> \
    --runs <2-4 newest loop outputs> --trials-per-side 1 --seed <date>
```

plus, kept in every sentinel set:

- 8 direction trials on new sliders sampled at the **score extremes** — 4 the score
  loves, 4 it hates. Always include the hated ones: without them discrimination is
  unmeasurable, and a loop that silently discards good sliders looks identical to a
  healthy one.
- 2 frozen anchors from the validated session (1 certified AUDIBLE, 1 certified
  chance) — unchanged clips, so they measure listener/playback drift, not model
  drift.
- 2 damage clips (1 new top-score clip, 1 frozen broken control).

Responses accumulate in `responses/`; the scorer runs per session and on the
concatenation.

**Triggers for a full re-validation session:**

1. A frozen anchor missed in two consecutive sentinels (the AUDIBLE one at chance,
   or the broken control not rated 3) -> listener/pipeline drift; results since the
   last clean sentinel are suspect.
2. Rolling last-24 *top-score* direction trials: <=14/24 correct (below the
   validated ~85% level at p<.05, and Wilson lower bound crossing 0.5) -> the score
   has drifted away from perception.
3. Rolling last-24 *bottom-score* trials: >=17/24 correct -> the score's "bad" is
   audibly good; the loop is throwing away working sliders.
4. **No listening needed, checked every batch:** >25% of new candidates score
   outside the validated score range (extrapolation guard), or any change to
   recipe, caption format, rank/alpha, or base model -> automatic full session.
   O1's 0.52->0.81 sweep-with-no-audible-change would have tripped this guard on
   range alone.

## 6. Files

- `scripts/build_ab_session.py` — builds `eval/listen/abtest/` (session.json,
  key.json, blinded clips/, responses/, index.html). Deterministic per `--seed`.
- `eval/listen/abtest/index.html` — trial runner: forced playback, keyboard
  (q/w play, 1/2 answer; space + 1/2/3 in damage), localStorage resume, downloads
  `ab_responses_*.jsonl` at the end.
- `scripts/score_ab_session.py` — validity gates, per-condition/per-run exact
  binomials, metric Spearman + median split, damage Fisher vs baseline. Stdlib only.
- Current build: 71 + 36 trials from G-energy, G-grit, G-space, G-vintage,
  F-dust-nopen, R-final (one variant per caption pair; `--runs` to change).
