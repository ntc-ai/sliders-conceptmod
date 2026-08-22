#!/usr/bin/env python3
"""Does the rendered slider move the audio the way the caption swap does?

The concept axis is not only a level change: swapping the caption alone moves
RMS -14% but spectral centroid +67%. A slider can score well on velocity-space
cosine and still deliver the wrong mixture, because the ~50-step solve weights
the delta by temporal coherence rather than by per-step energy: components
parallel to the velocity compound multiplicatively, shape components partly
cancel.

This compares each slider's rendered direction against the caption-swap ground
truth taken from the same folder's REF clips, so the comparison is per-render
and needs no external reference.

    python scripts/probe_render_direction.py eval/listen/calibrated-20s/*/

Quote effects at the slider setting whose level change matches the ground
truth's, not at +-2: with guidance g the trained edit is g times the natural
one, so +-2 on a g=3 slider is a 6x edit and will overshoot level badly.
"""

from __future__ import annotations

import argparse
import glob
import wave
from pathlib import Path

import numpy as np

FEATURES = ("rms", "centroid", "flatness", "hi4k", "crest")


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    if ch > 1:  # interleaved stereo read as mono fabricates high-frequency content
        a = a.reshape(-1, ch).mean(1)
    return a, sr


def feats(a: np.ndarray, sr: int) -> dict[str, float]:
    N, H = 2048, 512
    if len(a) < N * 2:
        return {}
    idx = np.arange(0, len(a) - N, H)
    fr = np.stack([a[i:i + N] for i in idx]) * np.hanning(N)
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-12
    fq = np.fft.rfftfreq(N, 1 / sr)
    return {
        "rms": float(a.std()),
        "centroid": float((S * fq).sum(1).mean() / S.sum(1).mean()),
        "flatness": float((np.exp(np.log(S).mean(1)) / S.mean(1)).mean()),
        "hi4k": float((S[:, fq > 4000].sum(1) / S.sum(1)).mean()),
        "crest": float(20 * np.log10(np.abs(a).max() / (a.std() + 1e-12))),
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    args = ap.parse_args(argv)

    for folder in args.folders:
        if not folder.is_dir():
            continue
        refs = {}
        for wav in folder.glob("*REF*.wav"):
            key = "plus" if "Trip-hop" in wav.stem else "minus"
            refs[key] = feats(*load(wav))
        sliders = {}
        for wav in folder.glob("*slider*.wav"):
            s = scale_of(wav.stem)
            if s is not None:
                sliders[s] = feats(*load(wav))
        if not sliders or 0.0 not in sliders:
            print(f"{folder.name}: no neutral clip")
            continue

        print(f"\n=== {folder.name} ===")
        if len(refs) == 2:
            gt = {f: (refs['plus'][f] - refs['minus'][f]) / refs['minus'][f] * 100 for f in FEATURES}
            print("  ground truth (caption swap, - pole -> + pole): " +
                  "  ".join(f"{f} {gt[f]:+.1f}%" for f in FEATURES if f != "crest"))
        else:
            gt = None
            print("  (no REF pair in this folder — cannot state ground truth)")

        base = sliders[0.0]
        print(f"  {'scale':>7} " + " ".join(f"{f:>10s}" for f in FEATURES) +
              ("   level-matched" if gt else ""))
        best = None
        for s in sorted(sliders):
            f = sliders[s]
            row = " ".join(f"{f[k]:10.4f}" if k != "centroid" else f"{f[k]:10.0f}" for k in FEATURES)
            mark = ""
            if gt and s != 0:
                # the setting whose level change matches the ground truth's
                d = abs((f["rms"] - base["rms"]) / base["rms"] * 100 - gt["rms"])
                if best is None or d < best[0]:
                    best = (d, s)
            print(f"  {s:7.2f} {row}{mark}")
        if gt and best:
            s = best[1]
            f = sliders[s]
            print(f"\n  at s={s:g} (the setting matching the ground truth's level change):")
            for k in FEATURES:
                if k == "crest":
                    continue
                got = (f[k] - base[k]) / base[k] * 100
                want = gt[k] / 2.0  # ground truth spans pole to pole; neutral is mid
                ratio = got / want if abs(want) > 1e-6 else float("nan")
                print(f"    {k:9s} got {got:+7.1f}%   want {want:+7.1f}%   ({ratio:5.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
