#!/usr/bin/env python3
"""One scalar per slider run: gated, direction-certified audible effect size.

WHAT THE NUMBER IS
    score in [0, 1)  the run renders audio at every setting, moves
                     monotonically in the caption pair's certified direction
                     on BOTH sides, keeps level inside a tolerance band, and
                     the value is a saturating function of how many clearly
                     audible steps of on-axis change the slider buys per unit
                     of scale, on its WEAKER side.  0.5 = two audible steps.
    score in (-2, 0) a render gate failed; magnitude = mean normalized
                     violation, so failures still rank by badness and an
                     optimizer keeps a slope instead of hitting a cliff.
    score = -3       the caption pair has no certifiable acoustic direction:
                     not a training failure -- fix the captions or render a
                     condition-interpolation reference before spending
                     GPU-hours (see AXIS below).

WHY GATE + SCALAR FOLDED INTO ONE ORDERED NUMBER
  * Velocity-space metrics are disqualified by measurement: probe cosine
    0.818 rendered digital silence while 0.771 rendered audio; cosine
    0.52-0.81 across a sweep changed nothing rendered (O1).  The adapter
    drives its own trajectory at inference.  Only renders are scored here.
  * The real failure modes are qualitative disqualifiers, not low quality:
    digital silence; -96..99% level collapse; +197..446% level explosion;
    movement opposite the caption (rank corr -1.0); one dead side (O5).
    In a weighted sum the optimizer buys spectrum with level; as gates, a
    run that trips one can never outrank a run that trips none.

WHY THE SCALAR IS *AUDIBLE STEPS*, NOT "FRACTION OF THE CAPTION SWAP"
    The obvious scalar -- slider effect as a fraction of the caption swap's
    own acoustic delta (O7) -- was implemented first and measured to be
    standing on noise.  The caption swap re-renders a DIFFERENT SONG:
    arrangement luck swamps the concept.  Measured on this corpus:
      - dusty/glossy swap, magnitude-spectrum centroid, one seed each:
        +92%, +55%, -25%.  Pooled over 9 seeds the median centroid span is
        ~0; the only seed-stable component is "dusty renders ~+50% louder".
        The celebrated "+55.4% centroid swap" was the luck of seed 7.
      - dense/sparse swap pooled over 10 seeds: no feature clears the
        sign-flip noise floor at all.
    Meanwhile the slider ladder is PAIRED -- scale s vs the same folder's
    scale 0 perturbs one trajectory -- and its deltas are precise.
    Dividing a precise numerator by an unmeasurable denominator manufactures
    exactly the kind of artifact metric this project keeps burning weeks on.
    So the caption swap contributes only what it can testify to -- the
    DIRECTION, certified by a sign-flip test -- and magnitude is normalized
    by fixed per-feature audibility steps (AUDIBLE below: ~3 dB level, ~20%
    centroid, ...).  Those constants are judgment calls, but they are
    STABLE judgment calls, immune to seed-count luck, and they make scores
    comparable across concepts, which a per-pair noisy denominator never was.

AXIS (concept direction) AND ITS CERTIFICATION
    Preferred source: condition-interpolation pole ladders
    (render_cond_interp.py, the prompt yamls' Gate B) via --condmix.  They
    perturb one cached AR plan, so their pole span carries far less
    arrangement noise than caption re-renders.  Measured: switching the
    dust axis from 10 pooled REF-swap seeds (uncertifiable, p=0.61) to a
    single cond-interp seed turned both dust runs from "no direction /
    backwards" into textbook monotone ladders (rho +1.00, endpoint sign
    10/10 seeds, near-symmetric steps), and dropped the energy runs'
    off-axis residual from 2.2 to 0.8 steps -- the sliders were never
    off-axis, the REF-swap axes were mis-aimed.
    Fallback: REF spans d_i = F(REF_pos) - F(REF_neg) pooled across every
    run of the pair (REF clips are slider-off, they belong to the pair, not
    the run), deduplicated by seed; u = normalize(median_i d_i/(2*AUDIBLE)).
    Certification: sign-flip test on ||median of flipped spans||; p <= 0.10
    certifies; p > 0.25 is a hard -3 at any n (the floor is 2/2^n <= 0.25,
    so p above it means most flips beat the observed direction); between,
    the pair is scored but tagged unresolved -- render more REF seeds.

THE SCALAR, PRECISELY
    Features per clip, natural-log: rms, magnitude-spectrum centroid,
    hi-freq fraction >4 kHz, spectral flatness, crest, stereo side/mid
    width, spectral-flux ratio (event density -- one axis under test is
    dense/sparse and no other feature sees event rate).  Magnitude not
    power spectrum: the audible signature of dust/gloss axes is low-energy
    HF content that power weighting buries (+55% mag vs -11% pow centroid
    on the same pair of clips).
    Per seed and setting: e(s) = (F(s) - F(0)) / AUDIBLE  (audible steps).
    p(s) = median over seeds of e(s).u  =  on-axis audible steps.
    Endpoint rate per side: E = p(s_end)/s_end, attenuated by off-axis
    movement in excess of ORTHO_ALLOW audible steps (movement the captions
    never asked for): E_att = E * E/(E + excess).
    score = min over sides of E_att / (E_att + E_HALF).   E_HALF = 2.
    Saturating, so the optimizer gains ~nothing past a handful of audible
    steps -- the observed endpoint of an unsaturated effect-size objective
    is the +197..446% level explosion, and gates already bound level.

GATES (working range |s| <= 1, median across seeds unless stated)
    G0 axis       certification above (-3 band, or tag when unresolvable)
    G1 silence    every clip, every seed: rms >= 0.02 x its own zero clip
    G2a level     every setting: |median delta-ln-rms| <= 8 dB.  Wide on
                  purpose: the live-composed perfect teacher dips -55%
                  (-6.9 dB) at net 1.0 and the prompt yamls record a mid-
                  ladder rms crater for every pair even in pure condition
                  space -- level is a nuisance channel (O4), tolerated in
                  moderation, vetoed at collapse.  The trip-hop run's -76%
                  (-12.4 dB) and the +197% explosion (+9.5 dB) both fail.
    G2b level     any single clip: |delta-ln-rms| <= 14 dB
    G3 direction  Spearman rho(s, p(s)) >= 0.8 AND the endpoint sign
                  agrees with the pole in >= 2/3 of seeds on each side

WHAT THIS NUMBER DOES NOT CAPTURE (keep saying it out loud)
  * Whether the movement IS the concept or a spectrally identical proxy:
    an EQ tilt and real "gloss" can be the same vector in these features.
  * Musical damage at constant level and spectrum: garbled vocals, smeared
    transients, harmonic wrongness, lyric loss.  A slider that degrades the
    music while moving the right direction scores well.  This is the main
    residual risk and it is a listening problem, not a feature problem.
  * Anything past 4 s / one denoise window: song structure, slider
    behavior across AR chunk boundaries, envelope/ramp behavior.
  * Slider-slider interaction when stacked.
  * Whether AUDIBLE constants are truly iso-audible across features.

LEAST-LISTENING VALIDATION PLAN
    1. Zero listening: permutation sanity (shuffle setting labels within
       folders -> scores must collapse); seed-split stability (score with
       disjoint seed halves -> run ranking must agree).  Both were run on
       2026-08-21: permuted ladders scored -0.7..-2.0 vs +0.36/+0.49 real;
       seed halves gave 0.360/0.351 (dust) and 0.466/0.581 (energy).
    2. ~10 min: gate audit -- listen to 3 gated clips and 3 near-threshold
       passes; every gate firing must correspond to audio you would reject.
    3. ~20 min: blind 2AFC on the survivors: zero clip vs +1 clip, "which
       is more <concept word>?", 3 seeds x top runs; per-run accuracy must
       rank like the score (Kendall tau).  If not, the axis is a proxy.
    4. ~10 min: quality spot-check of the single best run at +/-1 for the
       musical-damage blind spot before promoting anything.

USAGE
    python3 scripts/slider_score.py eval/listen/curve-gp-4s/* eval/listen/pairs-4s/*
    Folders differing only by a -seedN suffix are one run.  Each folder:
    *_slider_<Label>_{minus|plus}S.wav ladder, *_neutral_base_zero.wav, and
    REF_prompt_<Label>_no_slider.wav for both labels, same seed and AR plan.
    --condmix <PlusLabel>=<dir_pos>,<dir_neg> overrides the axis for one
    pair from condition-interpolation folders (condmix_{pos,neg}_plus1.wav).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import wave
from pathlib import Path

import numpy as np

FEATURES = ("rms", "centroid", "hi4k", "flatness", "crest", "width", "flux")
# One "clearly audible step" per feature, in ln units.  Judgment calls,
# deliberately fixed: ~3 dB level; ~20% centroid; ~50% HF fraction /
# flatness / width; ~20% crest; ~35% flux ratio.
AUDIBLE = np.array([0.35, 0.18, 0.41, 0.41, 0.18, 0.41, 0.30])
EPS = 1e-9
E_HALF = 2.0                # audible steps/unit at which the scalar = 0.5
ORTHO_ALLOW = 1.0           # free off-axis audible steps per unit of scale
P_CERT = 0.10               # sign-flip p to certify an axis
P_FAIL = 0.25               # p above this with >=6 REF seeds: hard -3
N_CERT = 6                  # REF seeds needed for the test to resolve
SILENCE_RATIO = 0.02        # G1
LEVEL_MED_DB = 8.0          # G2a
LEVEL_ANY_DB = 14.0         # G2b
RHO_MIN = 0.8               # G3
SIGN_FRAC_MIN = 2.0 / 3.0   # G3
WORK_RANGE = 1.0


# ---------------------------------------------------------------- audio

def load(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mono, side, sr).  side is (L-R)/2, zeros if mono."""
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if sw != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got sampwidth={sw}")
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    a = a.astype(np.float64) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)
        return a.mean(1), (a[:, 0] - a[:, 1]) / 2.0, sr
    return a, np.zeros_like(a), sr


def feats(mono: np.ndarray, side: np.ndarray, sr: int) -> np.ndarray:
    # Magnitude spectrum, not power: the audible signature of several axes
    # (dust crackle, gloss sheen) is low-energy HF content that power
    # weighting buries under the bass.  Measured: the dusty swap at seed 7
    # is +55% centroid in magnitude, -11% in power; +55% is the audible one.
    N, H = 2048, 512
    idx = np.arange(0, max(1, len(mono) - N), H)
    fr = np.stack([mono[i:i + N] for i in idx]) * np.hanning(N)
    S = np.abs(np.fft.rfft(fr, axis=1)) + EPS
    P = S ** 2
    fq = np.fft.rfftfreq(N, 1 / sr)
    rms = math.sqrt(float(np.mean(mono ** 2)) + EPS)
    centroid = float((S * fq).sum(1).mean() / S.sum(1).mean())
    hi4k = float((S[:, fq > 4000].sum(1) / S.sum(1)).mean())
    flat = float(np.mean(np.exp(np.log(P).mean(1)) / P.mean(1)))
    crest = float(np.max(np.abs(mono)) + EPS) / rms
    width = math.sqrt(float(np.mean(side ** 2)) + EPS) / rms
    flux = float(np.maximum(np.diff(S, axis=0), 0.0).mean() / S.mean())
    vals = [rms, centroid, hi4k, flat, crest, width, flux]
    return np.array([math.log(v + EPS) for v in vals])


# ---------------------------------------------------------------- parsing

SCALE_RE = re.compile(r"_slider_(?P<label>.+)_(?P<sign>minus|plus)(?P<mag>[0-9.]+)$")
ZERO_RE = re.compile(r"_slider_neutral_base_zero$")
REF_RE = re.compile(r"_REF_prompt_(?P<label>.+)_no_slider$")
SEED_RE = re.compile(r"-(seed|s)(\d+)$")


def parse_folder(folder: Path) -> dict | None:
    lad: dict[float, np.ndarray] = {}
    raw_rms: dict[float, float] = {}
    refs: dict[str, np.ndarray] = {}
    plus_label = minus_label = None
    for wav_path in sorted(folder.glob("*.wav")):
        stem = wav_path.stem
        m = SCALE_RE.search(stem)
        if m:
            s = float(m.group("mag")) * (-1 if m.group("sign") == "minus" else 1)
            mono, sd, sr = load(wav_path)
            lad[s] = feats(mono, sd, sr)
            raw_rms[s] = math.sqrt(float(np.mean(mono ** 2)))
            if s > 0:
                plus_label = m.group("label")
            else:
                minus_label = m.group("label")
            continue
        if ZERO_RE.search(stem):
            mono, sd, sr = load(wav_path)
            lad[0.0] = feats(mono, sd, sr)
            raw_rms[0.0] = math.sqrt(float(np.mean(mono ** 2)))
            continue
        m = REF_RE.search(stem)
        if m:
            mono, sd, sr = load(wav_path)
            refs[m.group("label")] = feats(mono, sd, sr)
    if 0.0 not in lad or plus_label is None:
        return None
    if plus_label not in refs or (minus_label and minus_label not in refs):
        return None
    return {"lad": lad, "raw_rms": raw_rms,
            "pos": refs[plus_label], "neg": refs[minus_label],
            "plus_label": plus_label, "minus_label": minus_label}


# ---------------------------------------------------------------- stats

def spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


# ---------------------------------------------------------------- axis

def sign_flip_p(spans: np.ndarray) -> float:
    """P(||median of sign-flipped spans|| >= observed) under H0: no direction.

    Spans are whitened by AUDIBLE before the norm.  Full enumeration up to
    2^12 flips, else 4096 random flips."""
    w = spans / AUDIBLE
    t_obs = float(np.linalg.norm(np.median(w, axis=0)))
    n = len(w)
    if n <= 12:
        flips = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        rng = np.random.default_rng(0)
        flips = rng.choice([-1, 1], size=(4096, n))
    t_null = np.linalg.norm(np.median(flips[:, :, None] * w[None], axis=1), axis=1)
    return float(np.mean(t_null >= t_obs - 1e-12))


def concept_axis(ref_rows: list[dict]) -> dict:
    """Direction from REF spans pooled per caption pair across runs.

    REF clips are rendered slider-off: they belong to the pair, not the run.
    Pooling matters -- 3-seed estimates of the same dust pair disagreed with
    its own 9-seed estimate in sign on the dominant feature."""
    seen: set[str] = set()
    span = []
    for row in ref_rows:
        if row["seed"] in seen:
            continue
        seen.add(row["seed"])
        span.append(row["pos"] - row["neg"])
    span = np.stack(span)
    g = np.median(span, axis=0) / (2.0 * AUDIBLE)   # audible steps per unit
    norm = float(np.linalg.norm(g))
    return {"u": g / max(norm, EPS), "steps_per_unit": norm,
            "n_seeds": len(seen), "p": sign_flip_p(span), "source": "refswap"}


def condmix_axis(dir_pos: Path, dir_neg: Path) -> dict:
    """Axis from condition-interpolation pole ladders (render_cond_interp.py).

    Preferred when rendered: cond-interp perturbs one cached AR plan, so its
    span carries far less arrangement noise than a caption re-render."""
    def endpoint(d: Path, pole: str) -> np.ndarray:
        return feats(*load(d / f"condmix_{pole}_plus1.wav"))
    span = endpoint(dir_pos, "pos") - endpoint(dir_neg, "neg")
    g = span / (2.0 * AUDIBLE)
    norm = float(np.linalg.norm(g))
    return {"u": g / max(norm, EPS), "steps_per_unit": norm,
            "n_seeds": 1, "p": float("nan"), "source": "condmix"}


# ---------------------------------------------------------------- scoring

def score_group(name: str, seeds: list[dict], axis: dict) -> dict:
    out: dict = {"name": name, "n_seeds": len(seeds),
                 "plus_label": seeds[0]["plus_label"],
                 "minus_label": seeds[0]["minus_label"],
                 "axis_source": axis["source"], "axis_p": axis["p"],
                 "n_ref_seeds": axis["n_seeds"],
                 "axis": {f: float(v) for f, v in zip(FEATURES, axis["u"])}}
    u = axis["u"]

    # -- axis certification (G0)
    p = axis["p"]
    if axis["source"] == "condmix":
        cert = "condmix(1 seed, uncertified)"
    elif math.isnan(p):
        cert = "uncertified"
    elif p <= P_CERT:
        cert = "certified"
    elif p > P_FAIL:
        # Not a small-sample floor artifact: the floor is 2/2^n <= 0.25, so
        # p above it means a majority of sign-flips beat the observed
        # direction -- genuine absence of evidence at any n.
        cert = "absent"
    else:
        cert = f"unresolved(n={axis['n_seeds']},p={p:.2f})"
    out["axis_cert"] = cert
    if cert == "absent":
        out.update(score=-3.0, verdict="UNSCOREABLE: no certifiable REF-swap "
                   f"direction (n={axis['n_seeds']}, p={p:.2f}); render more "
                   "REF seeds or a cond-interp reference, or fix the captions")
        return out

    # -- per-setting on-axis audible steps
    scales = sorted({s for sd in seeds for s in sd["lad"]
                     if s != 0.0 and abs(s) <= WORK_RANGE + 1e-9})
    p_med, ortho_med, dln_med, dln_worst = {}, {}, {}, {}
    end_pos_signs, end_neg_signs = [], []
    smax = max((s for s in scales if s > 0), default=None)
    smin = min((s for s in scales if s < 0), default=None)
    silence_worst = math.inf
    for s in scales:
        ps, orth, dr = [], [], []
        for sd in seeds:
            if s not in sd["lad"]:
                continue
            e = (sd["lad"][s] - sd["lad"][0.0]) / AUDIBLE
            pr = float(e @ u)
            ps.append(pr)
            orth.append(float(np.linalg.norm(e - pr * u)))
            dr.append(float(sd["lad"][s][0] - sd["lad"][0.0][0]))
            silence_worst = min(silence_worst,
                                sd["raw_rms"][s] / max(sd["raw_rms"][0.0], EPS))
            if s == smax:
                end_pos_signs.append(pr > 0)
            if s == smin:
                end_neg_signs.append(pr < 0)
        p_med[s] = float(np.median(ps))
        ortho_med[s] = float(np.median(orth))
        dln_med[s] = float(np.median(dr))
        dln_worst[s] = float(np.max(np.abs(dr)))

    # -- gates
    margins: dict[str, float] = {}
    ln_med = LEVEL_MED_DB * math.log(10) / 20
    ln_any = LEVEL_ANY_DB * math.log(10) / 20
    worst_med = max((abs(v) for v in dln_med.values()), default=0.0)
    worst_any = max(dln_worst.values(), default=0.0)
    margins["G1_silence"] = ((math.log(max(silence_worst, EPS))
                              - math.log(SILENCE_RATIO)) / abs(math.log(SILENCE_RATIO)))
    margins["G2a_level_med"] = (ln_med - worst_med) / ln_med
    margins["G2b_level_any"] = (ln_any - worst_any) / ln_any
    rho = spearman([0.0] + scales, [0.0] + [p_med[s] for s in scales])
    fracs = []
    if end_pos_signs:
        fracs.append(sum(end_pos_signs) / len(end_pos_signs))
    if end_neg_signs:
        fracs.append(sum(end_neg_signs) / len(end_neg_signs))
    sign_frac = min(fracs) if fracs else 0.0
    margins["G3_direction"] = min((rho - RHO_MIN) / (1 - RHO_MIN),
                                  (sign_frac - SIGN_FRAC_MIN) / (1 - SIGN_FRAC_MIN))
    out.update(rho=rho, sign_frac=sign_frac,
               worst_level_med_db=20 * worst_med / math.log(10),
               worst_level_any_db=20 * worst_any / math.log(10),
               silence_worst_ratio=silence_worst, gates=margins)

    # -- scalar: saturating audible-step rate on the weaker side
    sides = {}
    for tag, s_end in (("plus", smax), ("minus", smin)):
        if s_end is None:
            continue
        E = p_med[s_end] / s_end                      # steps per unit, signed
        excess = max(0.0, ortho_med[s_end] / abs(s_end) - ORTHO_ALLOW)
        E_att = max(0.0, E) * max(0.0, E) / (max(0.0, E) + excess + EPS)
        sides[tag] = {"steps_per_unit": E, "ortho_per_unit": ortho_med[s_end] / abs(s_end),
                      "attenuated": E_att, "sat": E_att / (E_att + E_HALF)}
    out["sides"] = sides
    out["rungs"] = {s: {"p": p_med[s], "ortho": ortho_med[s],
                        "d_rms_db": 20 * dln_med[s] / math.log(10)} for s in scales}
    strength = min((v["sat"] for v in sides.values()), default=0.0)
    out["strength"] = strength

    if min(margins.values()) < 0:
        bad = [m for m in margins.values() if m < 0]
        out["score"] = max(-2.0, float(np.mean(bad)))
        out["verdict"] = "GATED: " + ",".join(k for k, m in margins.items() if m < 0)
    else:
        out["score"] = strength
        out["verdict"] = "PASS" if cert == "certified" else f"PASS ({cert} axis)"
    return out


# ---------------------------------------------------------------- report

def report(res: dict, verbose: bool) -> None:
    print(f"=== {res['name']}  ({res['n_seeds']} seeds, "
          f"{res['minus_label']} -1 .. +1 {res['plus_label']})")
    print(f"  score {res['score']:+.3f}   {res['verdict']}")
    ptxt = "-" if math.isnan(res["axis_p"]) else f"{res['axis_p']:.3f}"
    print(f"  axis [{res['axis_source']}, {res['n_ref_seeds']} seeds, "
          f"p={ptxt}, {res['axis_cert']}]  " + "  ".join(
              f"{f} {v:+.2f}" for f, v in res["axis"].items()))
    if "sides" in res:
        for tag, v in res["sides"].items():
            print(f"  {tag:>5}: {v['steps_per_unit']:+.2f} audible steps/unit "
                  f"(ortho {v['ortho_per_unit']:.2f})  -> sat {v['sat']:.3f}")
        print(f"  rho {res['rho']:+.2f}  sign-agree {res['sign_frac']:.2f}  "
              f"level med/any {res['worst_level_med_db']:.1f}/"
              f"{res['worst_level_any_db']:.1f} dB")
        bad = [k for k, m in res["gates"].items() if m < 0]
        if bad:
            print("  failed gates: " + "  ".join(
                f"{k} (margin {res['gates'][k]:+.2f})" for k in bad))
        if verbose:
            print(f"  {'scale':>7} {'steps':>7} {'ortho':>6} {'d_rms':>8}")
            for s in sorted(res["rungs"]):
                g = res["rungs"][s]
                print(f"  {s:7.2f} {g['p']:+7.2f} {g['ortho']:6.2f} "
                      f"{g['d_rms_db']:+7.1f}dB")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print per-rung step/ortho/level tables")
    ap.add_argument("--condmix", action="append", default=[],
                    metavar="PLUSLABEL=DIR_POS,DIR_NEG",
                    help="axis for one pair from condition-interp folders "
                    "containing condmix_{pos,neg}_plus1.wav")
    ap.add_argument("--json", type=Path, help="dump full results as JSON")
    args = ap.parse_args(argv)

    groups: dict[str, list[dict]] = {}
    pair_refs: dict[tuple[str, str], list[dict]] = {}
    for folder in args.folders:
        if not folder.is_dir():
            continue
        parsed = parse_folder(folder)
        if parsed is None:
            print(f"skip {folder}: missing zero clip or matching REF pair",
                  file=sys.stderr)
            continue
        m = SEED_RE.search(folder.name)
        seed = m.group(2) if m else folder.name
        groups.setdefault(SEED_RE.sub("", folder.name), []).append(parsed)
        pair_refs.setdefault(
            (parsed["plus_label"], parsed["minus_label"]),
            []).append({"seed": seed, "pos": parsed["pos"], "neg": parsed["neg"]})

    axes = {pair: concept_axis(rows) for pair, rows in pair_refs.items()}
    for spec in args.condmix:
        label, _, dirs = spec.partition("=")
        d_pos, _, d_neg = dirs.partition(",")
        for pair in list(axes):
            if pair[0] == label:
                axes[pair] = condmix_axis(Path(d_pos), Path(d_neg))

    results = []
    for name, seeds in sorted(groups.items()):
        if len(seeds) < 3:
            print(f"skip {name}: {len(seeds)} seeds (< 3)", file=sys.stderr)
            continue
        pair = (seeds[0]["plus_label"], seeds[0]["minus_label"])
        results.append(score_group(name, seeds, axes[pair]))

    results.sort(key=lambda r: r["score"], reverse=True)
    for res in results:
        report(res, args.verbose)
    print("ranking: " + "  ".join(f"{r['name']} {r['score']:+.3f}"
                                  for r in results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
