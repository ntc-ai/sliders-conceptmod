#!/usr/bin/env python3
"""Score a slider by where its render curve goes, not by velocity cosine.

The probe scores an adapter at x_t drawn from *unperturbed* trajectories, which
is exactly where an open-loop adapter looks fine; at inference it drives its own
trajectory and the error compounds. So acceptance has to be a render.

The concept axis is a mixture: the caption swap moves brightness a lot and level
a little. Sweeping the slider traces a curve through (delta rms, delta centroid)
relative to the ladder's own scale-0 clip. Ground truth is a single point --
REF_pos relative to REF_neu. A good slider's curve passes near that point; the
open-loop failure is a curve that dives in level long before it arrives in
brightness.

Reported per folder:
  * the ladder, as percentages against its own zero clip
  * the setting where brightness matches ground truth, and the LEVEL ERROR there
    (the number that separates a working slider from a collapsing one)

    python scripts/score_render_curve.py eval/listen/traj-4s/*/ \
      --refs eval/listen/teacher-4s
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    if ch > 1:  # interleaved stereo read as mono fabricates high-frequency content
        a = a.reshape(-1, ch).mean(1)
    return a, sr


def feats(a: np.ndarray, sr: int) -> dict[str, float]:
    N, H = 2048, 512
    idx = np.arange(0, max(1, len(a) - N), H)
    fr = np.stack([a[i:i + N] for i in idx]) * np.hanning(N)
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-12
    fq = np.fft.rfftfreq(N, 1 / sr)
    return {
        "rms": float(a.std()),
        "centroid": float((S * fq).sum(1).mean() / S.sum(1).mean()),
        "hi4k": float((S[:, fq > 4000].sum(1) / S.sum(1)).mean()),
    }


def scale_of(stem: str) -> float | None:
    tag = stem.split("_")[-1]
    if tag == "zero":
        return 0.0
    if tag.startswith("minus"):
        return -float(tag[len("minus"):])
    if tag.startswith("plus"):
        return float(tag[len("plus"):])
    return None


def ladder(folder: Path, pattern: str) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for wav in folder.glob(pattern):
        s = scale_of(wav.stem)
        if s is not None:
            out[s] = feats(*load(wav))
    return out


def pct(cur: dict[str, float], base: dict[str, float], key: str) -> float:
    return (cur[key] - base[key]) / base[key] * 100.0


def crossing(points: list[tuple[float, float]], target: float) -> float | None:
    """First scale where the (monotone-ish) curve reaches `target`, linearly."""
    for (s0, y0), (s1, y1) in zip(points, points[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            return s0 + (target - y0) * (s1 - s0) / (y1 - y0)
    return None


GROUP_RE = __import__("re").compile(r"-(seed|s)\d+$")


def grouped_report(folders: list[Path], pattern: str) -> None:
    """Aggregate same-run folders that differ only in a -seedN / -sN suffix.

    The 2026-08-21 five-seed panel showed the caption-swap GT point swings more
    across denoise/AR seeds than the effects under study (REF_pos rms -15.5%
    to -69%, centroid +110% to -52% for one caption pair), so matching a
    single-seed GT point is noise. What survives aggregation: the median
    ladder, the consistency of the spectral trend's SIGN across seeds, and the
    worst median level dip. Those are what this mode reports."""
    groups: dict[str, list[Path]] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        groups.setdefault(GROUP_RE.sub("", folder.name), []).append(folder)
    for name, members in sorted(groups.items()):
        per_seed = [ladder(m, pattern) for m in members]
        per_seed = [l for l in per_seed if 0.0 in l]
        if not per_seed:
            print(f"=== {name}: no zero clips in {len(members)} folders ===")
            continue
        scales = sorted(set().union(*[set(l) for l in per_seed]))
        print(f"=== {name}  ({len(per_seed)} seeds) ===")
        print(f"  {'scale':>7} {'rms med':>9} {'rms range':>15} {'cen med':>9} {'cen range':>15}")
        med_curve = []
        cen_signs_pos, cen_signs_neg = [], []
        for l in per_seed:
            base = l[0.0]
            ups = [pct(l[s], base, "centroid") for s in sorted(l) if s > 0]
            dns = [pct(l[s], base, "centroid") for s in sorted(l) if s < 0]
            if ups:
                cen_signs_pos.append(np.sign(ups[-1]))
            if dns:
                cen_signs_neg.append(np.sign(dns[0]))
        for scale in scales:
            r = [pct(l[scale], l[0.0], "rms") for l in per_seed if scale in l]
            c = [pct(l[scale], l[0.0], "centroid") for l in per_seed if scale in l]
            if not r:
                continue
            med_curve.append((scale, float(np.median(r)), float(np.median(c))))
            print(f"  {scale:7.3f} {np.median(r):+8.1f}% [{min(r):+6.1f},{max(r):+6.1f}] "
                  f"{np.median(c):+8.1f}% [{min(c):+6.1f},{max(c):+6.1f}]")
        pos_med = [(s, r) for s, r, _ in med_curve if s > 0]
        if pos_med:
            worst = min(r for _, r in pos_med)
            print(f"  worst median level dip (+side): {worst:+.1f}%")
        if cen_signs_pos:
            agree = max(cen_signs_pos.count(1), cen_signs_pos.count(-1))
            print(f"  centroid endpoint sign agreement (+side): {agree}/{len(cen_signs_pos)}")
        if cen_signs_neg:
            agree = max(cen_signs_neg.count(1), cen_signs_neg.count(-1))
            print(f"  centroid endpoint sign agreement (-side): {agree}/{len(cen_signs_neg)}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--refs", type=Path, default=None,
                    help="folder holding REF_neu.wav and REF_pos.wav at the SAME duration "
                    "and seed. CAUTION: the five-seed panel showed this GT point is "
                    "seed-noise; prefer --group for any decision")
    ap.add_argument("--pattern", default="*slider*.wav")
    ap.add_argument("--group", action="store_true",
                    help="aggregate folders differing only by -seedN/-sN suffix: median "
                    "ladders, trend-sign consistency, worst level dip. No GT matching.")
    args = ap.parse_args(argv)
    if args.group:
        grouped_report(list(args.folders), args.pattern)
        return 0
    if args.refs is None:
        ap.error("--refs is required unless --group is given")

    neu = feats(*load(args.refs / "REF_neu.wav"))
    pos = feats(*load(args.refs / "REF_pos.wav"))
    gt = {k: pct(pos, neu, k) for k in ("rms", "centroid", "hi4k")}
    print(f"ground truth (caption swap, neutral -> + pole, {args.refs.name}): "
          f"rms {gt['rms']:+.1f}%  centroid {gt['centroid']:+.1f}%  hi4k {gt['hi4k']:+.1f}%")
    gt_neg = None
    neg_ref = args.refs / "REF_neg.wav"
    if neg_ref.exists():
        neg = feats(*load(neg_ref))
        gt_neg = {k: pct(neg, neu, k) for k in ("rms", "centroid", "hi4k")}
        print(f"ground truth (caption swap, neutral -> - pole): "
              f"rms {gt_neg['rms']:+.1f}%  centroid {gt_neg['centroid']:+.1f}%  "
              f"hi4k {gt_neg['hi4k']:+.1f}%")
    print()

    rows = []
    for folder in args.folders:
        if not folder.is_dir():
            print(f"missing {folder}")
            continue
        lad = ladder(folder, args.pattern)
        if 0.0 not in lad:
            print(f"{folder.name}: no zero clip")
            continue
        base = lad[0.0]
        print(f"=== {folder.name} ===")
        print(f"  {'scale':>7} {'rms':>9} {'centroid':>10} {'hi4k':>9}")
        curve = []
        for s in sorted(lad):
            d = {k: pct(lad[s], base, k) for k in ("rms", "centroid", "hi4k")}
            print(f"  {s:7.3f} {d['rms']:+8.1f}% {d['centroid']:+9.1f}% {d['hi4k']:+8.1f}%")
            curve.append((s, d))
        # + side only: a broken run's minus side can spuriously cross the
        # + target, and the curve is not assumed monotone across zero.
        pos_curve = [(s, d) for s, d in curve if s >= 0]
        neg_curve = sorted(((s, d) for s, d in curve if s <= 0), reverse=True)
        if gt_neg is not None and len(neg_curve) > 1:
            # - side: level is the discriminating feature there (the - caption is
            # louder, barely darker); find where rms matches and report the
            # brightness error at that setting, scanning outward from 0.
            pts_rn = [(s, d["rms"]) for s, d in neg_curve]
            s_neg = crossing(pts_rn, gt_neg["rms"])
            if s_neg is None:
                print(f"  - side: rms never reaches ground truth ({gt_neg['rms']:+.1f}%)")
            else:
                cent_at = np.interp(
                    s_neg,
                    [s for s, _ in sorted(neg_curve)],
                    [d["centroid"] for _, d in sorted(neg_curve)],
                )
                print(f"  - side level-matched at scale {s_neg:.3f}: centroid {cent_at:+.1f}% "
                      f"(ground truth {gt_neg['centroid']:+.1f}%)")
        pts_c = [(s, d["centroid"]) for s, d in pos_curve]
        s_star = crossing(pts_c, gt["centroid"])
        if s_star is None:
            reach = max((d["centroid"] for _, d in pos_curve), default=float("-inf"))
            print(f"  brightness never reaches ground truth (+{gt['centroid']:.1f}%); "
                  f"tops out at {reach:+.1f}%\n")
            rows.append((folder.name, None, None))
            continue
        pts_r = [(s, d["rms"]) for s, d in pos_curve]
        rms_at = np.interp(s_star, [s for s, _ in pts_r], [y for _, y in pts_r])
        err = rms_at - gt["rms"]
        if rms_at < -90.0:
            # Spectral features of a clip at <10% of the base level are hiss,
            # not music; a "brightness match" there is an artifact.
            print(f"  collapsed before reaching brightness: centroid crosses at scale "
                  f"{s_star:.3f} but rms is already {rms_at:+.1f}%\n")
            rows.append((folder.name, None, None))
            continue
        print(f"  brightness-matched at scale {s_star:.3f}: rms {rms_at:+.1f}% "
              f"(ground truth {gt['rms']:+.1f}%, level error {err:+.1f} pts)\n")
        rows.append((folder.name, s_star, err))

    if len(rows) > 1:
        print(f"{'run':34s} {'match scale':>11s} {'level error':>12s}")
        for name, s, err in sorted(rows, key=lambda r: abs(r[2]) if r[2] is not None else 1e9):
            if err is None:
                print(f"{name:34s} {'-':>11s} {'never bright':>12s}")
            else:
                print(f"{name:34s} {s:11.3f} {err:+11.1f}p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
