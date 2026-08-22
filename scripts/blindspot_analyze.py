#!/usr/bin/env python3
"""Regenerate the blind-spot audit tables (scratchpad/blindspots.md) from the scan CSVs.

Inputs (produced by blindspot_metrics.py scan and blindspot_whisper.py):
  scratchpad/bs_clips.csv, scratchpad/bs_pairs.csv        4s folders
  scratchpad/bs20_clips.csv, scratchpad/bs20_pairs.csv    20s folders (optional)
  scratchpad/bs_whisper.csv                                whisper pass (optional)

  python scripts/blindspot_analyze.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NUM = set('''rms rms_std centroid hi4k flatness crest silent_frac tail_delta_db env_std_db
frame_rms_min_db width_db lr_corr mono_cancel_db onset_rate flux_p90_med harmonicity
pump_frac rough_frac loud_kw_db hf_tonal_db onset_corr onset_lag_ms env_corr env_lag_ms
chroma_corr chromaseq_cos logspec_corr'''.split())
SKIP_GROUPS = {"refs", "teacher"}


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k in list(r):
            if k in NUM and r[k] != '':
                r[k] = float(r[k])
        r['scale'] = float(r['scale']) if r.get('scale') else None
        rows.append(r)
    return [r for r in rows if r.get('group') not in SKIP_GROUPS]


def by_group(rows):
    g = {}
    for r in rows:
        g.setdefault(r['group'], []).append(r)
    return g


def _rank(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    i = 0
    sv = v[order]
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def ladder_rho(rows, metric):
    seeds = {}
    for r in rows:
        if r['role'] == 'ladder' and r['scale'] is not None:
            seeds.setdefault(r['seed'], []).append((r['scale'], r[metric]))
    rhos = [spearman([p[0] for p in sorted(pts)], [p[1] for p in sorted(pts)])
            for pts in seeds.values() if len(pts) >= 4]
    return float(np.median(rhos)) if rhos else float('nan')


def delta_vs_zero(rows, metric):
    seeds = {}
    for r in rows:
        if r['role'] == 'ladder' and r['scale'] is not None:
            seeds.setdefault(r['seed'], {})[r['scale']] = r[metric]
    out = {}
    for lad in seeds.values():
        if 0.0 not in lad:
            continue
        for sc, v in lad.items():
            out.setdefault(sc, []).append(v - lad[0.0])
    return {sc: v for sc, v in sorted(out.items())}


def main():
    clips = load(ROOT / 'scratchpad/bs_clips.csv')
    pairs = load(ROOT / 'scratchpad/bs_pairs.csv')
    groups, pgroups = by_group(clips), by_group(pairs)

    metrics = ['rms', 'centroid', 'hi4k', 'flatness', 'crest',
               'width_db', 'lr_corr', 'harmonicity', 'pump_frac', 'rough_frac',
               'loud_kw_db', 'tail_delta_db', 'env_std_db']
    print("== A. trend strength: median Spearman rho (metric vs slider scale) ==")
    print(f"{'group':<14}" + "".join(f"{m[:9]:>10}" for m in metrics))
    for g in sorted(groups):
        print(f"{g:<14}" + "".join(f"{ladder_rho(groups[g], m):>10.2f}" for m in metrics))

    print("\n== B. structure preservation vs same-seed zero (medians) ==")
    pm = ['onset_corr', 'env_corr', 'chroma_corr', 'chromaseq_cos', 'logspec_corr']
    for g in sorted(pgroups):
        rows = pgroups[g]
        lad = {}
        for r in rows:
            if r['kind'] == 'vs_zero' and r['role'] == 'ladder':
                lad.setdefault(r['scale'], []).append(r)
        print(f"{g}:")
        for s in sorted(lad):
            print(f"  {s:>6} " + " ".join(f"{m}={np.median([v[m] for v in lad[s]]):.3f}" for m in pm))
        for label, kind, role in [("REF anchor", 'vs_zero', 'ref'), ("xseed null", 'null_xseed', None)]:
            rr = [r for r in rows if r['kind'] == kind and (role is None or r['role'] == role)]
            if rr:
                print(f"  {label}: " + " ".join(f"{m}={np.median([v[m] for v in rr]):.3f}" for m in pm))

    good = [r for r in pairs if r['kind'] == 'vs_zero' and r['onset_corr'] > 0.9]
    lag0 = sum(1 for r in good if r['onset_lag_ms'] == 0.0)
    print(f"\n== C. alignment control: {lag0}/{len(good)} healthy pairs peak at lag 0 ms ==")

    print("\n== D. stereo: width_db delta at +1 per seed ==")
    for g in sorted(groups):
        d = delta_vs_zero(groups[g], 'width_db')
        if 1.0 in d:
            v = sorted(d[1.0])
            agree = np.mean(np.sign(v) == np.sign(np.median(v)))
            print(f"  {g:<14} {[round(x, 2) for x in v]}  sign-agree {agree:.2f}")

    wcsv = ROOT / 'scratchpad/bs_whisper.csv'
    if wcsv.exists():
        wr = {(r['folder'], r['file']): r for r in csv.DictReader(open(wcsv))}
        zeros = {f: float(r['lyric_recall']) for (f, fn), r in wr.items() if fn.endswith('_zero.wav')}
        vocal = {f: v for f, v in zeros.items() if v >= 0.5}
        SCALE = re.compile(r'_(minus|plus)([0-9.]+)\.wav$')
        print(f"\n== E. lyric survival ({len(vocal)}/{len(zeros)} folders have audible lyric at 0) ==")
        for f, zv in sorted(vocal.items()):
            steps = []
            for (ff, fn), r in wr.items():
                if ff != f:
                    continue
                m = SCALE.search(fn)
                if m:
                    s = (-1 if m.group(1) == 'minus' else 1) * float(m.group(2))
                    steps.append((s, float(r['lyric_recall'])))
            print(f"  {f:<24} z={zv:.2f} " + " ".join(f"{s:+.2f}:{v:.2f}" for s, v in sorted(steps)))


if __name__ == "__main__":
    main()
