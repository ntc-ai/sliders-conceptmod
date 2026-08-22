"""Gates and ranking for one variant. Gates are VETOES, never terms.

Every gate exists because a measured attack defeats the ranking scalar alone
(scratchpad/goodhart.md): level collapse and explosion (G2, G1), the volume
knob that maximizes monotonicity (G3 is a gate, not a reward), song
replacement invisible to spectral statistics (G4 — D-pop-uni reads healthy on
every spectral feature while rendering a different song), hiss injection that
wins every purity ratio (G5 — the tell is spectral flatness), and seed-noise
progress (G6 null calibration). The ranking scalar is computed for everyone
but only ranks gate-passers; the full gate vector is logged for every variant
so a Goodhart walk is visible at the boundaries long before the scalar lies.

This module is pure data -> verdicts (no file I/O) so the selftest can feed it
synthetic versions of every attack in goodhart.md and assert the verdicts.
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field

from .spec import AUDIBLE_STEP, SCORED_FEATURES, AxisSpec
from .stats import spearman

DB = math.log(10.0) / 20.0  # 1 dB in ln units

# gate constants — frozen; per-axis overrides only via AxisSpec.must_not_move
G1_RMS_RATIO_MIN = 0.02          # vs same-seed zero (SCORING.md silence floor)
G1_ABS_FLOOR = 10 ** (-60 / 20)  # absolute clip rms floor
G2_MEDIAN_DB, G2_WORST_DB = 8.0, 14.0
G2_MEDIAN_DB_LEVEL_AXIS, G2_WORST_DB_LEVEL_AXIS = 12.0, 18.0
G3_SPEARMAN_MIN = 0.8
G3_ENDPOINT_AGREE = 2 / 3
G4_IDENT_MIN = 0.90              # onset/env/chroma vs same-seed zero, median across seeds
G5_FLATNESS_LN_MAX = 0.30        # hiss attack tell was +0.38 ln (+46 %)
G5_ROUGH_MAX = 0.35              # healthy 0.03; collapsed buzz 0.70-0.96
G5_CREST_LN_MAX = 0.40
G5_WHINE_RMS = 10 ** (-40 / 20)  # quiet-whine signature: quiet AND harmonic
G5_WHINE_HARMONICITY = 0.90
IDENT_KEYS = ("onset_corr", "env_corr", "chroma_corr")


@dataclass
class LadderCell:
    """One (seed, scale) measurement: ln deltas vs the same-seed zero clip."""

    deltas: dict[str, float]
    feats: dict[str, float] = field(default_factory=dict)
    ident: dict[str, float] | None = None


VariantData = dict[int, dict[float, LadderCell]]  # seed -> scale -> cell


def intended_proj(cell: LadderCell, axis: AxisSpec) -> float:
    """Audible steps along the concept direction (unit vector in step space)."""
    total = 0.0
    for feat, sign in axis.intended.items():
        total += sign * cell.deltas[feat] / AUDIBLE_STEP[feat]
    return total / math.sqrt(len(axis.intended))


def offaxis_steps(cell: LadderCell, axis: AxisSpec) -> float:
    tot = 0.0
    for feat in SCORED_FEATURES:
        if feat in axis.intended:
            continue
        tot += (cell.deltas[feat] / AUDIBLE_STEP[feat]) ** 2
    return math.sqrt(tot)


def _median_over_seeds(data: VariantData, scale: float, fn) -> float:
    vals = [fn(cells[scale]) for cells in data.values() if scale in cells]
    if not vals:
        raise ValueError(f"no cells at scale {scale}")
    return st.median(vals)


def _scales(data: VariantData) -> list[float]:
    sets = [set(c) for c in data.values()]
    common = sorted(set.intersection(*sets)) if sets else []
    if not common:
        raise ValueError("no common scales across seeds — ladders are unpaired")
    return common


def evaluate_variant(
    data: VariantData,
    axis: AxisSpec,
    null_e95: float | None = None,
    allow_single_side: bool = False,
) -> dict:
    """Gate vector + ranking scalar for one variant's ladder measurements.

    `data` must hold >= 3 seeds with a common scale grid including both signs.
    Returns {"gates": {name: {"pass": bool, "value": ..., "limit": ...}},
             "pass": bool, "score": float, ...}. Never raises on a failing
    gate — failing loudly is for broken inputs, not for bad sliders.
    """
    scales = _scales(data)
    nonzero = [s for s in scales if s != 0.0]
    pos = [s for s in nonzero if s > 0]
    neg = [s for s in nonzero if s < 0]
    if len(data) < 3 or (not pos and not neg):
        raise ValueError(f"need >=3 seeds and a nonzero side (scales={scales}, seeds={list(data)})")
    if (not pos or not neg) and not allow_single_side:
        raise ValueError(
            f"only one sign rendered (scales={scales}); a one-sided ladder cannot "
            "certify a slider — pass allow_single_side only for legacy folders"
        )
    gates: dict[str, dict] = {}

    # -- G1 silence floor ------------------------------------------------
    offenders = []
    for seed, cells in data.items():
        for s in nonzero:
            c = cells[s]
            ratio = math.exp(c.deltas["rms_pc"])
            if ratio < G1_RMS_RATIO_MIN or c.feats.get("rms_pc", 1.0) < G1_ABS_FLOOR:
                offenders.append({"seed": seed, "scale": s, "ratio": round(ratio, 5)})
    gates["G1_silence"] = {"pass": not offenders, "offenders": offenders,
                           "limit": f"ratio>={G1_RMS_RATIO_MIN} and abs>={G1_ABS_FLOOR:.4g}"}

    # -- G2 level containment -------------------------------------------
    med_db, worst_db = (G2_MEDIAN_DB, G2_WORST_DB)
    if axis.level_axis:
        med_db, worst_db = (G2_MEDIAN_DB_LEVEL_AXIS, G2_WORST_DB_LEVEL_AXIS)
    med_excess = max(abs(_median_over_seeds(data, s, lambda c: c.deltas["rms_pc"])) for s in nonzero) / DB
    worst_excess = max(abs(c.deltas["rms_pc"]) for cells in data.values() for sc, c in cells.items() if sc != 0.0) / DB
    gates["G2_level"] = {"pass": med_excess <= med_db and worst_excess <= worst_db,
                         "median_db": round(med_excess, 2), "worst_db": round(worst_excess, 2),
                         "limit": f"median<={med_db}dB worst<={worst_db}dB"}

    # -- G3 direction ----------------------------------------------------
    med_curve = [(_median_over_seeds(data, s, lambda c: intended_proj(c, axis)) if s != 0.0 else 0.0)
                 for s in scales]
    rho = spearman(scales, med_curve)
    agree = {}
    if pos:
        hi = max(pos)
        agree["pos"] = st.mean([1.0 if intended_proj(cells[hi], axis) > 0 else 0.0 for cells in data.values()])
    if neg:
        lo = min(neg)
        agree["neg"] = st.mean([1.0 if intended_proj(cells[lo], axis) < 0 else 0.0 for cells in data.values()])
    gates["G3_direction"] = {
        "pass": rho >= G3_SPEARMAN_MIN and all(a >= G3_ENDPOINT_AGREE for a in agree.values()),
        "spearman": round(rho, 3),
        **{f"endpoint_agree_{k}": round(v, 2) for k, v in agree.items()},
        "limit": f"rho>={G3_SPEARMAN_MIN} agree>={G3_ENDPOINT_AGREE:.2f}/side",
    }

    # -- G4 song identity ------------------------------------------------
    ident_rows = {}
    ok4 = True
    for s in nonzero:
        row = {}
        for key in IDENT_KEYS:
            med = _median_over_seeds(data, s, lambda c, k=key: _ident(c, k))
            row[key] = round(med, 3)
            if med < G4_IDENT_MIN:
                ok4 = False
        ident_rows[str(s)] = row
    gates["G4_identity"] = {"pass": ok4, "per_scale": ident_rows, "limit": f">={G4_IDENT_MIN} median, every scale"}

    # -- G5 artefact guardrails -----------------------------------------
    problems = []
    for s in nonzero:
        if "flatness" not in axis.intended:
            # the anti-hiss tell — but a concept whose percept IS noise texture
            # (dust/crackle, distortion) declares flatness intended and takes
            # its brightness evidence from the other gates instead
            flat = _median_over_seeds(data, s, lambda c: c.deltas["flatness"])
            if flat > G5_FLATNESS_LN_MAX:
                problems.append({"scale": s, "flatness_ln": round(flat, 3)})
        if "crest" not in axis.intended:
            crest = abs(_median_over_seeds(data, s, lambda c: c.deltas["crest"]))
            if crest > G5_CREST_LN_MAX:
                problems.append({"scale": s, "crest_ln": round(crest, 3)})
        rough = _median_over_seeds(data, s, lambda c: c.feats.get("rough_frac", 0.0))
        if rough > G5_ROUGH_MAX:
            problems.append({"scale": s, "rough_frac": round(rough, 3)})
        for feat, band in axis.must_not_move.items():
            move = abs(_median_over_seeds(data, s, lambda c, f=feat: c.deltas[f]))
            if move > band:
                problems.append({"scale": s, feat: round(move, 3), "band": band})
    for seed, cells in data.items():
        for s in nonzero:
            f = cells[s].feats
            if f.get("rms_pc", 1.0) < G5_WHINE_RMS and f.get("harmonicity", 0.0) > G5_WHINE_HARMONICITY:
                problems.append({"seed": seed, "scale": s, "whine": True})
    gates["G5_artefacts"] = {"pass": not problems, "problems": problems}

    # -- ranking scalar (computed for everyone; ranks only gate-passers) --
    sides = {}
    side_specs = ([("plus", max(pos))] if pos else []) + ([("minus", min(neg))] if neg else [])
    for label, extreme in side_specs:
        per_seed = []
        for cells in data.values():
            c = cells[extreme]
            proj = intended_proj(c, axis)
            if label == "minus":
                proj = -proj
            atten = 1.0 / (1.0 + max(0.0, offaxis_steps(c, axis) - 1.0))
            per_seed.append(max(0.0, proj) * atten / abs(extreme))
        e = st.median(per_seed)
        sides[label] = {"E": round(e, 4), "squashed": round(e / (e + 2.0), 4), "scale": extreme}
    e_min = min(v["E"] for v in sides.values())
    score = e_min / (e_min + 2.0)

    # -- G6 null calibration --------------------------------------------
    if null_e95 is None:
        gates["G6_null"] = {"pass": False, "value": None,
                            "note": "no null distribution supplied — unscoreable, not a free pass"}
    else:
        gates["G6_null"] = {"pass": e_min > null_e95, "E_min": round(e_min, 4),
                            "null_e95": round(null_e95, 4)}

    return {
        "gates": gates,
        "pass": all(g["pass"] for g in gates.values()),
        "score": round(score, 4),
        "sides": sides,
        "single_sided": not (pos and neg),
        "median_curve": {str(s): round(v, 3) for s, v in zip(scales, med_curve)},
    }


def _ident(cell: LadderCell, key: str) -> float:
    if cell.ident is None or key not in cell.ident:
        raise ValueError(f"cell lacks identity metric {key} — pair metrics were not computed")
    return cell.ident[key]


# ---------------------------------------------------------------- selftest

def _mk(deltas: dict[str, float], ident=0.97, rms_abs=0.1, rough=0.03, harm=0.5) -> LadderCell:
    d = {f: 0.0 for f in SCORED_FEATURES}
    d.update(deltas)
    return LadderCell(
        deltas=d,
        feats={"rms_pc": rms_abs, "rough_frac": rough, "harmonicity": harm},
        ident={k: ident for k in IDENT_KEYS},
    )


def _ladder(fn) -> VariantData:
    scales = [-1.0, -0.5, 0.0, 0.5, 1.0]
    out: VariantData = {}
    for i, seed in enumerate((7, 23, 77)):
        jitter = (i - 1) * 0.02
        out[seed] = {s: (fn(s, jitter) if s != 0.0 else _mk({})) for s in scales}
    return out


def selftest() -> int:
    axis = AxisSpec(intended={"centroid": 1.0})
    null = 0.05

    # healthy slider: monotone brightness, contained level, identity kept
    healthy = _ladder(lambda s, j: _mk({"centroid": 0.35 * s + j, "rms_pc": -0.05 * s}))
    r = evaluate_variant(healthy, axis, null)
    assert r["pass"], r
    assert r["score"] > 0.3, r["score"]

    # volume knob / level collapse (R-final): monotone, antisymmetric, huge rms
    knob = _ladder(lambda s, j: _mk({"centroid": 0.38 * s + j, "rms_pc": -1.4 * s}))
    r = evaluate_variant(knob, axis, null)
    assert not r["gates"]["G2_level"]["pass"], "level collapse must fail G2"
    # ... and note its G3 monotonicity is perfect — a gate, never a reward
    assert r["gates"]["G3_direction"]["pass"]

    # digital near-silence (D-pole): ratio < 0.02 of zero
    silent = _ladder(lambda s, j: _mk({"centroid": 0.5 * s, "rms_pc": -4.2 * abs(s)},
                                      rms_abs=0.002, rough=0.8, harm=0.95))
    r = evaluate_variant(silent, axis, null)
    assert not r["gates"]["G1_silence"]["pass"]
    assert not r["gates"]["G5_artefacts"]["pass"]  # whine + roughness signature

    # hiss injection (F2B): huge centroid at ~no rms cost, flatness tell
    hiss = _ladder(lambda s, j: _mk({"centroid": 2.2 * s, "rms_pc": 0.01 * s,
                                     "flatness": 0.38 * abs(s), "hi4k": 0.5 * s}))
    r = evaluate_variant(hiss, axis, null)
    assert not r["gates"]["G5_artefacts"]["pass"], "hiss must fail the flatness guardrail"

    # song replacement (D-pop): healthy spectral stats, identity collapsed at +1
    def repl(s, j):
        c = _mk({"centroid": 0.15 * s, "rms_pc": 0.05 * s}, ident=(0.13 if s >= 1.0 else 0.95))
        return c
    r = evaluate_variant(_ladder(repl), axis, null)
    assert not r["gates"]["G4_identity"]["pass"], "song replacement must fail G4"

    # no-op micro-EQ: monotone-ish noise, survives vetoes, must die on G6 null
    noop = _ladder(lambda s, j: _mk({"centroid": 0.012 * s + j}, ident=0.995))
    r = evaluate_variant(noop, axis, null_e95=0.10)
    assert not r["gates"]["G6_null"]["pass"], "a no-op must not beat the seed-noise null"

    # reversed slider
    rev = _ladder(lambda s, j: _mk({"centroid": -0.4 * s}))
    r = evaluate_variant(rev, axis, null)
    assert not r["gates"]["G3_direction"]["pass"]

    # dead minus pole: plus works, minus flat -> min() ranks it by the dead side
    dead = _ladder(lambda s, j: _mk({"centroid": (0.5 * s if s > 0 else 0.001 * s)}))
    r = evaluate_variant(dead, axis, null)
    assert r["sides"]["minus"]["E"] < 0.1 < r["sides"]["plus"]["E"]
    assert r["score"] < 0.05

    # missing null distribution is a hard fail, not a free pass
    r = evaluate_variant(healthy, axis, None)
    assert not r["gates"]["G6_null"]["pass"]

    print("gates selftest OK (goodhart attack suite: collapse, silence, hiss, "
          "song-replace, no-op, reversed, dead-pole)")
    return 0
