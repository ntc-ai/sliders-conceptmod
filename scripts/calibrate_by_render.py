#!/usr/bin/env python3
"""Calibrate a slider's unit_scale from rendered audio, not velocity deltas.

MUSIC3.md records that `calibrate_scale.py`'s target ("one trained concept" from
3-timestep velocity deltas on a 4s clip) over-drives these checkpoints: at face
value the shipped triphop +2 render came out near-silent. Renders are the only
strength signal this project has trusted.

A slider's effect on level is multiplicative and compounds through the stack, so
across a scale ladder the rendered level follows

    rms(s) = rms(0) * exp(-k * s)

`k` is the slider's gain per unit. Fitting it on a rendered ladder and comparing
against a reference slider's `k` gives

    unit_scale = k_reference / k_measured

so user +-1 costs the same amount of level as the reference's +-1. The trip-hop
axis genuinely carries some level change (caption swap alone: 1.22x pole to
pole), so the goal is matching a known-good slider's curve, not flattening it.

    python scripts/calibrate_by_render.py eval/listen/best-20s/I-r32full-s2000 \
      --reference eval/listen/loss-rank-ab-20s/shipped-v4-unit-calibrated

Add --write to record unit_scale into the checkpoint's sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from pathlib import Path

import numpy as np


def clip_rms(path: Path) -> float:
    with wave.open(str(path)) as w:
        frames = w.readframes(w.getnframes())
    a = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return float(a.std())


def ladder(folder: Path) -> dict[float, float]:
    """scale -> rms, from a generate_listen output folder."""
    out: dict[float, float] = {}
    for wav in folder.glob("*slider*.wav"):
        tag = wav.stem.split("_")[-1]
        if tag == "zero":
            scale = 0.0
        elif tag.startswith("minus"):
            scale = -float(tag[len("minus"):])
        elif tag.startswith("plus"):
            scale = float(tag[len("plus"):])
        else:
            continue
        out[scale] = clip_rms(wav)
    return out


def fit_k(points: dict[float, float], max_abs_scale: float | None = None) -> tuple[float, float]:
    """Least-squares k for rms = rms0*exp(-k*s), plus worst relative residual.

    Fit on the inner ladder by default: the far poles leave the manifold, where
    the level change stops being a clean gain and becomes collapse.
    """
    base = points.get(0.0)
    if not base:
        raise SystemExit("ladder has no zero-scale clip to normalise against")
    items = [(s, r) for s, r in points.items() if s != 0.0]
    if max_abs_scale is not None:
        items = [(s, r) for s, r in items if abs(s) <= max_abs_scale]
    if not items:
        raise SystemExit("no non-zero scales within the fit range")
    xs = np.array([s for s, _ in items], dtype=np.float64)
    ys = np.log(np.array([r for _, r in items], dtype=np.float64) / base)
    k = float(-(xs * ys).sum() / (xs * xs).sum())
    worst = max(abs(math.exp(-k * s) - r / base) / (r / base) for s, r in items)
    return k, worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path, help="generate_listen output folders")
    ap.add_argument("--reference", type=Path, required=True, help="a calibrated slider's folder")
    ap.add_argument("--fit_range", type=float, default=1.0,
                    help="fit |scale| <= this; beyond it the render leaves the manifold")
    ap.add_argument("--write", action="store_true", help="write unit_scale into each sidecar")
    ap.add_argument("--models_root", type=Path, default=Path("/ml2/music/sliders-conceptmod/models/overnight"))
    args = ap.parse_args(argv)

    k_ref, err_ref = fit_k(ladder(args.reference), args.fit_range)
    print(f"reference {args.reference.name}: k={k_ref:.3f} ({math.exp(k_ref):.2f}x per unit, "
          f"fit err {err_ref*100:.1f}%)\n")
    print(f"{'slider':30s} {'k':>7s} {'gain/unit':>10s} {'fit err':>8s} {'unit_scale':>11s}")
    for folder in args.folders:
        if not folder.is_dir():
            print(f"missing {folder}")
            continue
        try:
            k, err = fit_k(ladder(folder), args.fit_range)
        except SystemExit as exc:
            print(f"{folder.name:30s} {exc}")
            continue
        unit = k_ref / k if k else float("nan")
        print(f"{folder.name:30s} {k:7.3f} {math.exp(k):9.2f}x {err*100:7.1f}% {unit:11.4f}")
        if args.write:
            run_dir = args.models_root / folder.name
            wrote = 0
            for sidecar in run_dir.glob("*_last.json"):
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                meta["unit_scale"] = round(unit, 6)
                meta["unit_scale_source"] = "render_gain_fit"
                meta["unit_scale_reference"] = args.reference.name
                meta["render_gain_k"] = round(k, 6)
                meta["calibrated"] = True
                sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                wrote += 1
            print(f"{'':30s} wrote unit_scale to {wrote} sidecar(s) in {run_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
