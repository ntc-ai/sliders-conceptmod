#!/usr/bin/env python3
"""Score a blind listening session against ground truth and a candidate metric.

    python scripts/score_ab_session.py                       # newest responses file
    python scripts/score_ab_session.py --responses r.jsonl \
        --metric metric.csv --damage-metric dmetric.csv

Reads eval/listen/abtest/{key.json,responses/*.jsonl} and prints:
  1. listener validity  - repeat self-consistency, REF vocabulary controls,
                          synthetic broken controls, response times
  2. per-condition      - accuracy + exact binomial p per (run, side)
  3. per-run pooled     - AUDIBLE / CHANCE / REVERSED classification
  4. metric agreement   - condition-level Spearman + trial-level median split
  5. damage             - flag rates vs the zero/REF baseline, per-run Fisher

Metric CSV: header + rows keyed by  run,value  |  run,side,value  |  clip,value
(clip = source path as stored in key.json, substring match allowed).
A metric may be per-run even if the truth varies by side; the report says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---- exact small-n statistics, stdlib only (env scipy is numpy-2 broken) ----


def binom_tail(k: int, n: int, alt: str) -> float:
    """Exact binomial tail vs p=0.5. alt='greater': P(X>=k); alt='less': P(X<=k)."""
    if n == 0:
        return 1.0
    rng = range(k, n + 1) if alt == "greater" else range(0, k + 1)
    return sum(math.comb(n, i) for i in rng) / 2 ** n


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact for [[a,b],[c,d]] by summing tables as or less likely."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def p_of(x: int) -> float:
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)

    p_obs = p_of(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-9)))


def rankdata(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float], n_perm: int = 20000) -> tuple[float, float]:
    """Spearman rho with a two-sided permutation p (deterministic seed)."""

    def pearson(a: list[float], b: list[float]) -> float:
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        da = math.sqrt(sum((p - ma) ** 2 for p in a))
        db_ = math.sqrt(sum((q - mb) ** 2 for q in b))
        return num / (da * db_) if da and db_ else 0.0

    rx, ry = rankdata(x), rankdata(y)
    rho = pearson(rx, ry)
    rng = random.Random(0)
    hits = 0
    ry2 = ry[:]
    for _ in range(n_perm):
        rng.shuffle(ry2)
        if abs(pearson(rx, ry2)) >= abs(rho) - 1e-12:
            hits += 1
    return rho, hits / n_perm


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def binom_p(k: int, n: int, alt: str) -> float:
    return binom_tail(k, n, alt) if n else 1.0


def db(x: float | None, y: float | None) -> float | None:
    if not x or not y:
        return None
    return 20 * math.log10(x / y)


def load_metric(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "value" not in rows[0]:
        raise SystemExit(f"{path}: need a CSV with a 'value' column")
    return rows


def metric_for(rows: list[dict], run: str, side: str | None, clip: str | None) -> float | None:
    best = None
    for r in rows:
        if clip and r.get("clip") and (r["clip"] in clip or clip in r["clip"]):
            return float(r["value"])
        if r.get("run") == run:
            if side and r.get("side") and r["side"] != side:
                continue
            best = float(r["value"])
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=REPO / "eval" / "listen" / "abtest")
    ap.add_argument("--responses", type=Path, default=None,
                    help="responses .jsonl (default: newest in <session>/responses/)")
    ap.add_argument("--metric", type=Path, default=None, help="direction metric CSV")
    ap.add_argument("--damage-metric", type=Path, default=None, help="damage metric CSV")
    args = ap.parse_args()

    key = {k["id"]: k for k in json.loads((args.session / "key.json").read_text())["trials"]}
    rpath = args.responses
    if rpath is None:
        cands = sorted((args.session / "responses").glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("no responses found; pass --responses")
        rpath = cands[-1]
    resp = {}
    for line in rpath.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            resp[r["trial_id"]] = r  # last answer wins
    print(f"responses: {rpath}  ({len(resp)} answers over {len(key)} trials)\n")

    dir_resp = [(key[t], r) for t, r in resp.items()
                if t in key and key[t]["block"] == "direction"]
    dam_resp = [(key[t], r) for t, r in resp.items()
                if t in key and key[t]["block"] == "damage"]

    # ---------------- 1. listener validity ----------------
    print("== 1. listener validity")
    rep = [(k, r) for k, r in dir_resp if k["kind"] == "repeat"]
    agree = 0
    for k, r in rep:
        orig = resp.get(k["repeat_of"])
        if orig is None:
            continue
        # same underlying pair; compare which multiplier was chosen, not a/b label
        ko = key[k["repeat_of"]]
        chose_hi = r["response"] == k["correct"]
        chose_hi_orig = orig["response"] == ko["correct"]
        agree += chose_hi == chose_hi_orig
    if rep:
        print(f"  repeat self-consistency: {agree}/{len(rep)}"
              f"  (a guesser matches ~{len(rep) / 2:.0f}; "
              f"P(>= {agree}) under guessing = {binom_p(agree, len(rep), 'greater'):.3f})")
    refs = [(k, r) for k, r in dir_resp if k["kind"] == "ref_control"]
    kr = sum(r["response"] == k["correct"] for k, r in refs)
    if refs:
        print(f"  REF vocabulary controls: {kr}/{len(refs)} correct"
              f"  (chance p = {binom_p(kr, len(refs), 'greater'):.3f})")
        if kr <= len(refs) / 2:
            print("  !! REF controls at chance: either the vocabulary does not transfer"
                  " to this model's renders, or the listener could not hear the axis."
                  " Slider results below are uninterpretable on this axis-check basis.")
    synth = [(k, r) for k, r in dam_resp if k["kind"].startswith("synth_")]
    for k, r in synth:
        ok = r["response"] == "3"
        print(f"  synthetic {k['kind'].split('_', 1)[1]:9s} rated {r['response']} "
              f"({'ok' if ok else 'MISSED — validity concern'})")
    rts = sorted(r["rt_ms"] for _, r in dir_resp)
    if rts:
        print(f"  direction median rt: {rts[len(rts) // 2] / 1000:.1f}s "
              f"(both clips force-played before answering; floor ~8s)")
    print()

    # ---------------- 2. per-condition ----------------
    print("== 2. per-condition accuracy  (core trials: extreme-vs-zero, same seed)")
    core = [(k, r) for k, r in dir_resp if k["kind"] in ("core", "repeat")]
    conds: dict[tuple[str, str], list] = {}
    for k, r in core:
        conds.setdefault((k["run"], k["side"]), []).append((k, r))
    mrows = load_metric(args.metric) if args.metric else None
    table = []
    print(f"  {'run':16s} {'side':4s} {'n':>2s} {'acc':>5s} {'p>':>6s} {'p<':>6s} "
          f"{'lvl dB':>7s} {'metric':>8s}")
    for (run, side), items in sorted(conds.items()):
        n = len(items)
        kc = sum(r["response"] == k["correct"] for k, r in items)
        dbs = [abs(d) for k, _ in items
               if (d := db(k["rms_a"], k["rms_b"])) is not None]
        lvl = sum(dbs) / len(dbs) if dbs else float("nan")
        mv = metric_for(mrows, run, side, None) if mrows else None
        table.append({"run": run, "side": side, "n": n, "k": kc, "acc": kc / n,
                      "lvl": lvl, "metric": mv})
        flag = " LEVEL-SEPARABLE" if lvl > 6 else ""
        print(f"  {run:16s} {side:4s} {n:2d} {kc / n:5.2f} "
              f"{binom_p(kc, n, 'greater'):6.3f} {binom_p(kc, n, 'less'):6.3f} "
              f"{lvl:7.1f} {mv if mv is not None else float('nan'):8.3f}{flag}")
    print("  (p> tests audible-in-intended-direction, p< tests reversed;"
          " LEVEL-SEPARABLE = pair differs by >6 dB, loudness alone could answer)\n")

    # ---------------- 3. per-run pooled ----------------
    print("== 3. per-run verdict  (both sides pooled)")
    for run in sorted({run for run, _ in conds}):
        items = [it for (r_, s_), v in conds.items() if r_ == run for it in v]
        n = len(items)
        kc = sum(r["response"] == k["correct"] for k, r in items)
        pg, pl = binom_p(kc, n, "greater"), binom_p(kc, n, "less")
        lo, hi = wilson(kc, n)
        verdict = "AUDIBLE" if pg < 0.05 else ("REVERSED" if pl < 0.05 else "chance/weak")
        print(f"  {run:16s} {kc:2d}/{n:2d}  acc {kc / n:.2f}  "
              f"CI [{lo:.2f},{hi:.2f}]  {verdict}")
    print("  (n=10-12 per run: >=9/10 certifies audible at p<=.011;"
          " <=1/10 certifies reversed; anything between is 'not established')\n")

    # ---------------- 4. metric agreement ----------------
    if mrows:
        print("== 4. metric agreement (direction)")
        have = [t for t in table if t["metric"] is not None]
        if len(have) >= 4:
            rho, p = spearman([t["metric"] for t in have], [t["acc"] for t in have])
            print(f"  condition-level Spearman (N={len(have)}): rho={rho:.2f}  p={p:.3f}")
            med = sorted(t["metric"] for t in have)[len(have) // 2]
            hi_ = [(t["k"], t["n"]) for t in have if t["metric"] >= med]
            lo_ = [(t["k"], t["n"]) for t in have if t["metric"] < med]
            kh, nh = sum(k for k, _ in hi_), sum(n for _, n in hi_)
            kl, nl = sum(k for k, _ in lo_), sum(n for _, n in lo_)
            pf = fisher_exact_p(kh, nh - kh, kl, nl - kl)
            print(f"  trial-level median split: high-metric acc {kh}/{nh}={kh / nh:.2f}"
                  f"  low-metric acc {kl}/{nl}={kl / nl:.2f}  Fisher p={pf:.3f}")
            print("  supported claim if rho>=0.5 & split gap >=25pp & p<.05: the score"
                  " ranks audible sliders above inaudible ones on this material.")
            print("  never supported by this design: calibrated score->audibility mapping,"
                  " differences between mid-pack conditions, or transfer to 20-90s renders.")
        else:
            print("  metric matched too few conditions; check CSV keys")
        print()

    # ---------------- 5. damage ----------------
    print("== 5. damage  (rating 1 fine / 2 degraded / 3 broken)")
    base = [(k, r) for k, r in dam_resp if k["kind"].startswith("baseline")]
    sld = [(k, r) for k, r in dam_resp if k["kind"] == "slider"]
    bflag = sum(r["response"] != "1" for _, r in base)
    sflag = sum(r["response"] != "1" for _, r in sld)
    print(f"  baseline (zero-scale + REF) flagged: {bflag}/{len(base)}"
          f"   slider extremes flagged: {sflag}/{len(sld)}")
    if base and sld:
        p = fisher_exact_p(sflag, len(sld) - sflag, bflag, len(base) - bflag)
        print(f"  pooled slider-vs-baseline Fisher p = {p:.3f}"
              " (damage attributable to sliders, over base-model quirks)")
    per_run: dict[str, list] = {}
    for k, r in sld:
        per_run.setdefault(k["run"], []).append(r["response"])
    for run, rs in sorted(per_run.items()):
        f = sum(x != "1" for x in rs)
        b = sum(x == "3" for x in rs)
        p = fisher_exact_p(f, len(rs) - f, bflag, len(base) - bflag) if base else 1.0
        note = "  DAMAGED" if p < 0.05 else ""
        print(f"  {run:16s} flagged {f}/{len(rs)} (broken {b})  vs-baseline p={p:.3f}{note}")
    if args.damage_metric:
        drows = load_metric(args.damage_metric)
        pairs = []
        for k, r in dam_resp:
            if k["kind"].startswith("synth"):
                continue
            mv = metric_for(drows, k["run"], None, k.get("path"))
            if mv is not None:
                pairs.append((mv, int(r["response"])))
        if len(pairs) >= 6:
            rho, p = spearman([a for a, _ in pairs], [b for _, b in pairs])
            print(f"  damage-metric Spearman over {len(pairs)} clips: rho={rho:.2f} p={p:.3f}")
    print("\n(any validity failure in section 1 caps what sections 2-5 can claim;"
          " see scratchpad/listening_protocol.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
