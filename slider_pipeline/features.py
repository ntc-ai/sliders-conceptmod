"""Feature extraction for the pipeline: one stated convention, loud failures.

Wraps scripts/blindspot_metrics.py (the audited instrument: pure numpy, its
selftest carries one control per historical measurement-bug class — stereo
interleave, rms convention, lag alignment). This module adds:

  * the canonical level number `rms_pc` — per-channel-power rms,
    sqrt(mean(all_samples^2)) over the de-interleaved (n, ch) array. Chosen per
    the blind-spot audit (I1): the mono-downmix rms silently mixes level change
    with stereo-width change through phase cancellation (up to 40 % on real
    renders). The mono rms and mono_cancel_db stay logged so the difference is
    visible, never implicit.
  * folder scanning with header validation (sr / channel count must agree
    across a ladder folder) and same-seed pair metrics against the folder's
    own zero clip.

The renders themselves are full-pipeline outputs (generate_listen), so the
>8 s chunked-conditioning corruption class does not apply here; durations are
still checked against the request and mismatches raise.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import blindspot_metrics as bm  # noqa: E402  (audited instrument)


@dataclass
class Clip:
    path: Path
    role: str                 # ladder | ref | other
    scale: float | None       # None for refs
    sr: int
    n_ch: int
    duration: float
    feats: dict[str, float]   # clip metrics + rms_pc
    ident: dict[str, float] | None = None   # pair metrics vs same-seed zero


def measure_clip(path: Path) -> tuple[np.ndarray, int, dict[str, float]]:
    a, sr = bm.load(path)
    feats = bm.clip_metrics(a, sr)
    feats["rms_pc"] = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    return a, sr, feats


def scan_folder(folder: Path, expect_duration: float | None = None) -> dict[float, Clip]:
    """Measure every ladder clip in a listen folder, paired against its zero clip.

    Returns {scale: Clip}. Raises loudly on: missing zero clip, unreadable or
    inconsistent audio (sr / channels), duration below 90 % of the request.
    """
    folder = Path(folder)
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"{folder}: no wav files")
    clips: dict[float, Clip] = {}
    arrays: dict[float, np.ndarray] = {}
    sr_seen: set[int] = set()
    ch_seen: set[int] = set()
    for wav in wavs:
        role, scale = bm.parse_role(wav.stem)
        if role != "ladder" or scale is None:
            continue
        a, sr, feats = measure_clip(wav)
        sr_seen.add(sr)
        ch_seen.add(a.shape[1])
        dur = a.shape[0] / sr
        if expect_duration is not None and dur < 0.9 * expect_duration:
            raise ValueError(f"{wav}: duration {dur:.2f}s < 90% of requested {expect_duration}s")
        clips[scale] = Clip(path=wav, role=role, scale=scale, sr=sr, n_ch=a.shape[1], duration=dur, feats=feats)
        arrays[scale] = a
    if len(sr_seen) > 1 or len(ch_seen) > 1:
        raise ValueError(f"{folder}: inconsistent audio headers sr={sr_seen} ch={ch_seen}")
    if 0.0 not in clips:
        raise FileNotFoundError(f"{folder}: no zero-scale clip — every delta is paired against it")
    zero = arrays[0.0]
    sr = clips[0.0].sr
    for scale, clip in clips.items():
        if scale == 0.0:
            continue
        clip.ident = bm.pair_metrics(zero, arrays[scale], sr)
    return clips


def ln_delta(feat: str, val: float, zero_val: float) -> float:
    """Signed delta of one feature against the same-seed zero clip.

    ln-ratio for positive quantities; plain difference for width_db (already dB,
    sign-carrying). Non-finite results raise: a NaN here means a broken clip or
    a broken measurement, and both must be seen, not averaged away.
    """
    if feat == "width_db":
        d = float(val - zero_val)
    else:
        if zero_val <= 0 or val <= 0:
            # true digital silence: represent as a huge but finite drop so the
            # silence gate fires instead of NaNs propagating
            return -6.0 if val <= zero_val else 6.0
        d = math.log(val / zero_val)
    if not math.isfinite(d):
        raise ValueError(f"non-finite delta for {feat}: {val} vs {zero_val}")
    return d


def selftest() -> int:
    """Pipeline-features selftest on top of the audited instrument's own."""
    rc = bm.selftest()
    if rc != 0:
        print("blindspot_metrics selftest FAILED", file=sys.stderr)
        return rc
    # rms_pc convention: stereo file with silent right channel. Per-channel
    # power halves the energy (rms_pc = rms_L/sqrt(2)); the mono downmix
    # halves the amplitude (rms_mono = rms_L/2). Distinct by construction.
    sr = 44100
    t = np.arange(sr) / sr
    left = np.sqrt(2.0) * np.sin(2 * np.pi * 440 * t) * 0.5
    a = np.stack([left, np.zeros_like(left)], axis=1)
    rms_pc = float(np.sqrt(np.mean(a**2)))
    rms_mono = float(np.sqrt(np.mean(a.mean(axis=1) ** 2)))
    assert abs(rms_pc - 0.5 / math.sqrt(2)) < 1e-3, rms_pc      # 0.354
    assert abs(rms_mono - 0.25) < 1e-3, rms_mono                # 0.250
    assert rms_pc / rms_mono > 1.3, "per-channel vs downmix must differ on wide content"
    # ln_delta sign conventions
    assert ln_delta("rms_pc", 2.0, 1.0) > 0
    assert ln_delta("width_db", -3.0, -5.0) == 2.0
    assert ln_delta("rms_pc", 0.0, 0.1) == -6.0
    print("features selftest OK (incl. blindspot_metrics controls)")
    return 0
