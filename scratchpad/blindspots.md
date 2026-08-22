# Blind-spot audit: what the five DSP features cannot see

Date: 2026-08-21. Evidence base: `eval/listen/curve-gp-4s` (E, F, R runs), `eval/listen/pairs-4s`
(G-energy/grit/space/vintage), `eval/listen/seed-sweep-4s` (D-pole-uni, D-pop-uni), plus a
length check on `eval/listen/best-20s`. 481 4s clips + 61 20s clips scanned. All numbers below
are reproducible: `scripts/blindspot_metrics.py` (measures, with selftest),
`scripts/blindspot_whisper.py` (lyric/embedding pass, offline cached model),
`scripts/blindspot_analyze.py` (regenerates every table from `scratchpad/bs_*.csv`).

Current feature set audited: rms, spectral centroid, hi4k energy ratio, spectral flatness,
crest factor — whole-clip, mono downmix.

---

## 0. Instrument validation (done before any finding was trusted)

`python scripts/blindspot_metrics.py selftest` — 11 synthetic checks, all passing, including
one control per historical bug class:

* **Stereo-read-as-mono** (bug 1): loader asserts channel count from the header and de-interleaves;
  the selftest builds a stereo file (L=440 Hz, R=330 Hz) and shows the interleaved-as-mono read
  fabricates HF (centroid 385 Hz proper vs 11026 Hz bugged). Every scan row records `sr`, `n_ch`,
  `n_samples` per file. Note: the renders are **44.1 kHz**, not the 24 kHz in the model notes —
  the header is read per file, never assumed.
* **RMS convention** (bug 2): one stated convention (`sqrt(mean(mono^2))`, float in [-1,1]);
  selftest checks a unit sine gives 0.7071. The scan also cross-checks every LISTEN.md rms figure
  it can parse — and this control **fired** (see finding I1).
* **Alignment** (bug 3): every pairwise measure searches +-150 ms of lag and reports the best lag.
  Selftest recovers a deliberate 100 ms shift. On the real data, **254/254** same-seed pairs with
  onset_corr > 0.9 peak at exactly 0 ms lag — no hidden misalignment in this render set.

One bug in my own analysis was caught and fixed during the audit: Spearman rho without tie
handling produced a fake rho = 1.0 on the all-zero `silent_frac` column (numbered ranks on ties
correlate with anything sorted). Average-rank ties fixed it; all rho tables below are post-fix.

### I1. Live catch: the project already has two rms conventions in production

LISTEN.md tables were generated with **per-channel-power rms** (rms over the interleaved array);
`scripts/score_render_curve.py` (source of the O2 numbers) uses **mono-downmix rms**. On the same
445 files: median |difference| = 0.0117 (about 10% of typical rms 0.1), max = 0.049 (40%).
Matching against per-channel-power rms instead gives max |difference| = 5e-5, i.e. that is exactly
what LISTEN.md computed. The gap is out-of-phase stereo content cancelling in the downmix:
`mono_cancel_db` median -1.08 dB, 5th percentile -2.47 dB, worst -3.43 dB
(G-energy-seed404/01_slider_Sparse_minus1.wav) — and it varies across a ladder, so published rms
percentages silently mix level change with stereo-width change. Pick one convention (recommend
per-channel-power) and log `mono_cancel_db` as its own column.

---

## (a)+(b) Candidate blind spots, each tested on the renders

### B1. Song identity — did the edit keep the same piece of music? REAL, the largest gap.

Three cheap same-seed comparisons against each ladder's own scale-0 clip: `onset_corr`
(spectral-flux envelope correlation, best-lag), `env_corr` (frame-RMS envelope), `chroma_corr`
(12-dim pitch-class profile). Two anchors calibrate them: the same-seed caption-swap REF render
("what a genuinely different rendering of this seed looks like") and cross-seed zero-vs-zero pairs
("different song, same caption").

* Identity band: every healthy ladder step across E/F/G/R sits at onset_corr >= 0.92,
  env_corr >= 0.95, chroma_corr >= 0.97 (mild dip to 0.87 at E-gp05 +1).
* Different-song floor: REF anchor onset_corr 0.12-0.25, cross-seed null 0.11-0.19.
  The gap between bands is ~0.7 — enormous, and seed-stable (onset_corr at +1: std 0.007-0.034
  across seeds per group).
* Broken checkpoints cross the floor mid-ladder: D-pole-uni onset_corr 0.91 -> 0.63 -> 0.33 ->
  0.085 at scales 0.17/0.33/0.5/1.0; D-pop-uni hits 0.16 by +0.85.

**The demonstration that the current five cannot see this**: D-pop-uni at scale +1.2 has
rms +4.9%, centroid -6.5%, hi4k -27%, flatness -21% (each within or below the cross-seed CV of the
same feature on zero clips: 17%, 30%, 147%, 104%) — it looks like a mild, working slider. Its
onset_corr per seed is [0.088, 0.157, 0.183, 0.212, 0.239], entirely inside the different-song
null [0.038 .. 0.322]: the render is a different piece of music with normal summary statistics.
Meanwhile the same checkpoint at +0.5 shows rms -61% — so a loop optimizing "level preserved"
on the current features would prefer +1.2 (total song replacement) over +0.5 (a level dip).
Only crest (+30% vs CV 11%) hints anything changed, and crest cannot distinguish "song replaced"
from the legitimate "sparser arrangement".

Also note `logspec_corr` (average-spectrum-shape similarity, i.e. the family the current five
belong to) reads 0.87 for *completely different songs* vs 0.96-0.99 for same-song edits — average
spectral statistics fundamentally cannot separate "same song, edited" from "different song,
same genre".

### B2. Vocal / lyric survival. REAL, orthogonal to B1, needs the pretrained model.

Whisper-large-v3-turbo (cached locally, run offline on GPU 1) transcribed all 481 clips.
Only 10/56 folders have an audible lyric in the first 4 s at scale 0 — so recall is only usable
*paired* against the same-seed zero clip. In those 10 folders the result is binary:

* Every healthy slider keeps the transcript **word-identical** across the entire ladder
  (e.g. E-gp05-seed23: recall 0.77 at every scale from -0.5 to +1.0, including +1.0 where rms
  fell -65% — that seed's failure is level collapse, not song replacement; failure severity is
  seed-dependent).
* Broken checkpoints lose the words exactly where structure collapses: D-pop-uni-seed23
  0.77 -> 0.46 (+0.5) -> 0.00 (+0.85, transcript "Did you see me?"; +1.1 "Beepishmoo!").

The Whisper encoder embedding (mean-pooled over real frames, cosine vs zero clip) agrees with the
DSP structure trio (Spearman 0.855 over 419 pairs; same-song band >= 0.966, different-song median
0.895) — i.e. the cheap DSP metrics already capture most of what the embedding sees, but with a
much wider margin. Keep the embedding as a cross-check, not the primary gate.

### B3. Stereo image. REAL — and already contaminating current numbers (see I1).

Mono downmix destroys `width_db` (side/mid energy) and `lr_corr`:

* G-energy: width delta at +1 negative in **10/10 seeds** [-2.62 .. -0.28 dB] (denser -> narrower)
  while rms delta at +1 has sign agreement only 6/10. On the minus side width is non-mirror
  (4/10 seeds opposite sign) — the slider narrows on + but does not reliably widen on -.
* G-vintage: width +1.86 dB median at +1, 3/3 seeds, rho with lr_corr dropping 0.57 -> 0.40 —
  "modern" reliably widens; the minus (vintage) pole barely moves width (+0.44) — asymmetric.
* G-space: width rho -1.00 across the ladder (3 dB swing dry -> cavernous).

### B4. Time variation within the clip. REAL as a guardrail; whole-clip means hide the arc.

* `env_std_db` (loudness-trajectory spread) trends with |rho| >= 0.86 in 8/11 groups —
  sliders systematically change the dynamic arc, invisible to whole-clip rms.
* `tail_delta_db` (last quarter vs first quarter): G-vintage +1.1 dB median at +1 (rho +1.0);
  G-grit monotone within 2/3 seeds (a gritty +1 clip stops building toward its end).
* Digital silence: none in this 4s set (`silent_frac` = 0 everywhere; quietest clips are
  -40..-52 dBFS "quiet whine", not zeros), so the O4 silence mode needs the frame-level check
  as a guardrail with a perceptual threshold (~-45 dBFS), not the -60 dB digital one.

### B5. Envelope modulation (pumping / roughness). REAL, moves where the current five are flat.

* G-grit — the pair O2 calls flat (rms +0.5%, centroid +5.8%): `pump_frac` (1-8 Hz envelope
  modulation) falls monotonically 0.755 -> 0.590 across the ladder, rho -1.00, sign-consistent
  3/3 seeds. The grit slider's audible signature is amplitude-envelope texture, not spectrum.
* G-vintage pump +0.066 (3/3), G-space pump +0.050 (3/3), both rho >= 0.93.
* `rough_frac` (30-150 Hz modulation) is a strong artefact flag: 0.03 on healthy clips,
  0.70-0.96 on D-pole's collapsed +0.5/+1 clips (quiet tonal buzz), and the quiet-whine signature
  (rms < -40 dBFS with harmonicity > 0.9) marks every collapsed clip in the set.

### B6. Pitch/harmonic content. PARTIAL.

`chroma_corr` earns its place inside B1 (harmony survival; drops to 0.33-0.51 on replaced songs).
As an *attribute* measure, `harmonicity` moves cleanly only in G-space (rho 0.98, +0.023 median,
2/3 seeds) and G-energy (rho -0.94, denser -> noisier); elsewhere small. Keep chroma for identity,
harmonicity mainly for the artefact signature above.

### B7. Perceptual (K-weighted) loudness vs flat rms. MOSTLY NOT A BLIND SPOT — honest negative.

Frequency-domain K-weighting (BS.1770 approximation) disagrees in sign with flat rms dB on only
8/308 ladder steps (threshold 0.25 dB), and 6 of those are D-pop's already-broken high scales
(e.g. seed11 s=1.2: rms -1.05 dB vs K-weighted +2.42 dB — the "recovered" energy is HF artefact).
Useful as a tiebreaker in broken regimes; not worth much on healthy ladders.

### B8. Onset rate / transient density as an attribute. NOT DEMONSTRATED — honest negative.

`onset_rate` at 4 s is quantized to ~14-20 onsets and shows no reliable ladder trend anywhere
(|rho| <= 0.44 in the corrected table) — even dense/sparse (G-energy) does not change onset count;
it changes spectrum and width instead. Transient structure matters as *identity* (onset_corr),
not as a rate statistic on 4 s clips.

### B9. HF tonal artefacts (birdies/aliasing). WEAK in this set.

`hf_tonal_db` baseline in real music is noisy (5-8 dB); it flags some collapsed clips (8.8 dB)
but adds little beyond the B5 artefact signature. Keep as a cheap logged column, not a selector.

---

## Protocol blind spots (not features — how the numbers are used)

### P1. Mirror symmetry is never checked, and it fails in real checkpoints.

At |s| = 1: G-energy rms delta is *positive on both poles* (asym index 1.0 — both directions get
louder); G-grit and G-vintage width move the same direction on both poles; G-space centroid falls
on both poles. O4's "+s and -s not mirror images" is measurable today from existing ladders —
score both poles and penalize |d(+s) + d(-s)| — but the current pipeline only reports +1.

### P2. Seed noise dominates single-seed caption-referenced scores (O6 replicated).

The caption-swap ground-truth point swings wildly: REF-vs-zero centroid delta across seeds is
-33%..+45% (G-energy, 10 seeds, sign flips) and -37%..+136% (F-dust). Ladder deltas at +1 have
cross-seed std of 5-23 centroid points against medians of +5..+18 — SNR near or below 1.
The paired structure metrics are 10-30x more seed-stable (onset_corr std 0.007-0.034 against a
0.7-wide scale). Anything caption-referenced must be multi-seed median + sign-agreement;
anything same-seed-paired can be trusted at 3 seeds.

### P3. 4 s clips measure the worst window of a product-length render (O5 confirmed).

On 20 s renders of the same recipe (best-20s, 2 checkpoints), the slider effect is strongly
front-loaded: at +1 the first 4 s window shows -27.5 / -21.7 dB vs zero, the last windows only
-13.6 / -11.2 dB (about half, in dB). onset_corr is likewise lowest in window 1 (0.77) and
recovers to 0.92+ later. A 4 s eval clip coincides with exactly that first window, so 4 s scores
systematically overstate level effects and understate whole-song identity. Before the loop starts,
either evaluate at >= 20 s or add a windowed profile (per-4s-window delta curve) so front-loading
is visible.

---

## (c) Ranked shortlist to add before the automated loop starts

1. **Structure-preservation gate** — onset_corr + env_corr + chroma_corr vs same-seed zero, with
   best-lag reported, calibrated by the per-run cross-seed null. Reject any scale where the trio
   falls toward the null band; report the usable scale range, not just +1.
   *Cost: pure numpy, ~1 s/clip, already implemented.* This kills the D-pop failure class that the
   current features actively reward.
2. **Score both poles + mirror-asymmetry penalty** on every feature (P1). *Cost: zero new renders
   — minus-side clips already exist; analysis change only.*
3. **One rms convention + stereo columns** — per-channel-power rms as canonical, plus `width_db`,
   `lr_corr`, `mono_cancel_db` (I1, B3). *Cost: trivial.*
4. **Artefact guardrails** — quiet-whine signature (dBFS < -40 & harmonicity > 0.9), `rough_frac`
   jump, `silent_frac`/`frame_rms_min_db`, `hf_tonal_db` logged. *Cost: trivial.* Catches all of
   O4's collapse/whine modes explicitly instead of via rms side effects.
5. **Multi-seed protocol** — >= 3 seeds, medians + sign-agreement for caption-referenced numbers
   (P2). *Cost: 3x renders (already the practice in newer folders).*
6. **Length check** — at least one >= 20 s render per candidate with the windowed delta profile
   (P3). *Cost: ~5x render time on one setting, once per candidate.*
7. **Whisper lyric survival** — paired transcript recall on vocal seeds; encoder-embedding cosine
   as cross-check (B2). *Cost: ~0.5 s/clip on GPU, model already cached offline; only meaningful
   on seeds whose zero clip carries the vocal, so select 1-2 vocal seeds per run.*
8. **pump_frac** as an attribute feature (B5) — the only new *attribute* axis that moved where all
   five current features were flat. *Cost: trivial.*

Deliberately left out: K-weighted loudness as a primary metric (B7 negative), onset-rate (B8
negative), hf_tonal as selector (B9 weak).

One Goodhart warning for the loop design: the structure gate saturates for a do-nothing slider
(identity = 1.0 at every scale), and caption-similarity saturates for a caption-copying slider.
The gate (identity) and the objective (attribute movement, multi-seed sign-consistent, both poles)
must be scored jointly; either alone is trivially gameable.

---

## Files

* `scripts/blindspot_metrics.py` — measures + selftest + scan CLI (pure numpy; scipy in the base
  env is ABI-broken against numpy 2 and must not be imported).
* `scripts/blindspot_whisper.py` — lyric/embedding pass; run with the `minimax-music3` env python
  (`/home/mikkel/anaconda3/envs/minimax-music3/bin/python`, transformers 5.x); base-env
  transformers pulls sklearn -> broken scipy on `generate`.
* `scripts/blindspot_analyze.py` — regenerates tables A-E from the CSVs.
* `scratchpad/bs_clips.csv`, `bs_pairs.csv` — 4 s scan; `bs20_*.csv` — 20 s scan;
  `bs_whisper.csv`, `bs_whisper_emb.npz` — whisper pass.
