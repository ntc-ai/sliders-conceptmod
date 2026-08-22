#!/usr/bin/env python3
"""Blind-spot audit metrics: what a listener hears that rms/centroid/hi4k/flatness/crest miss.

Pure numpy (scipy in this env is ABI-broken against numpy 2). All per-clip and
pairwise measures used by the 2026-08 blind-spot audit live here.

Conventions (stated so the next rms-convention bug is impossible):
  * loader returns float64 in [-1, 1], shape (n_samples, n_channels); sample
    rate is read from the header, never assumed (these renders are 44.1 kHz,
    not the 24 kHz the model card suggests).
  * "mono" = mean over channels AFTER de-interleaving.
  * rms = sqrt(mean(mono**2)) over the whole clip. rms_std = mono.std() is
    also emitted because LISTEN.md tables were generated with std(); the scan
    cross-checks our value against every LISTEN.md row it can parse.
  * every pairwise measure reports the best cross-correlation LAG it found;
    a nonzero lag between same-seed renders means misalignment, which is the
    control that catches chunked-sampler/conditioning offset bugs.

CLI:
  python scripts/blindspot_metrics.py selftest
  python scripts/blindspot_metrics.py scan FOLDER [FOLDER...] --out-clips clips.csv --out-pairs pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import wave
from pathlib import Path

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- loading

def load(path: Path) -> tuple[np.ndarray, int]:
    """Return (samples float64 (n, ch), sr). Never silently downmixes."""
    with wave.open(str(path)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sampwidth={sw}")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if len(a) != n * ch:
        raise ValueError(f"{path}: frame count mismatch")
    return a.reshape(n, ch), sr


def mono(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=1)


# ---------------------------------------------------------------- STFT base

def stft_mag(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Magnitude STFT, frames x bins, hann window."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    idx = np.arange(0, len(x) - n_fft + 1, hop)
    frames = np.stack([x[i:i + n_fft] for i in idx]) * np.hanning(n_fft)
    return np.abs(np.fft.rfft(frames, axis=1))


# ---------------------------------------------------------------- per-clip

def frame_rms_db(x: np.ndarray, sr: int, win_s: float = 0.046, hop_s: float = 0.0116) -> np.ndarray:
    win, hop = max(1, int(win_s * sr)), max(1, int(hop_s * sr))
    idx = np.arange(0, max(1, len(x) - win), hop)
    r = np.sqrt(np.array([np.mean(x[i:i + win] ** 2) for i in idx]) + EPS)
    return 20 * np.log10(r + EPS)


def k_weight(freqs: np.ndarray) -> np.ndarray:
    """Approximate BS.1770 K-weighting power response (freq domain)."""
    f2 = freqs ** 2
    hp = (f2 / (f2 + 38.0 ** 2)) ** 2                       # RLB-ish highpass
    shelf = 1.0 + (10 ** 0.4 - 1.0) * f2 / (f2 + 1500.0 ** 2)  # ~+4 dB high shelf
    return hp * shelf


def chroma_gram(S: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Frames x 12 pitch-class power, each frame L1-normalized."""
    P = S ** 2
    sel = (freqs >= 55.0) & (freqs <= 5512.0)
    pc = (np.round(12.0 * np.log2(freqs[sel] / 440.0)).astype(int)) % 12
    C = np.zeros((S.shape[0], 12))
    np.add.at(C.T, pc, P[:, sel].T)
    return C / (C.sum(axis=1, keepdims=True) + EPS)


def onset_env(S: np.ndarray) -> np.ndarray:
    """Half-wave rectified log-magnitude spectral flux, z-scored."""
    L = np.log1p(1000.0 * S)
    flux = np.maximum(0.0, np.diff(L, axis=0)).sum(axis=1)
    return (flux - flux.mean()) / (flux.std() + EPS)


def harmonicity(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 1024) -> float:
    """Median over active frames of max normalized autocorrelation, 60-1000 Hz lags."""
    if len(x) < n_fft:
        return 0.0
    idx = np.arange(0, len(x) - n_fft + 1, hop)
    frames = np.stack([x[i:i + n_fft] for i in idx])
    active = np.sqrt((frames ** 2).mean(axis=1)) > 10 ** (-50 / 20)
    if not active.any():
        return 0.0
    frames = frames[active] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, n=2 * n_fft, axis=1)) ** 2
    ac = np.fft.irfft(spec, axis=1)[:, :n_fft]
    ac = ac / (ac[:, :1] + EPS)
    lo, hi = int(sr / 1000.0), int(sr / 60.0)
    return float(np.median(ac[:, lo:hi].max(axis=1)))


def env_mod_fracs(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(pump 1-8 Hz, roughness 30-150 Hz) fraction of envelope-modulation power."""
    win, hop = 256, 128
    idx = np.arange(0, max(1, len(x) - win), hop)
    env = np.sqrt(np.array([np.mean(x[i:i + win] ** 2) for i in idx]) + EPS)
    env = env - env.mean()
    rate = sr / hop
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    f = np.fft.rfftfreq(len(env), 1.0 / rate)
    tot = spec[(f >= 0.3) & (f <= 160.0)].sum() + EPS
    pump = spec[(f >= 1.0) & (f <= 8.0)].sum() / tot
    rough = spec[(f >= 30.0) & (f <= 150.0)].sum() / tot
    return float(pump), float(rough)


def hf_tonal_peak_db(S: np.ndarray, freqs: np.ndarray, f_lo: float = 8000.0) -> float:
    """Max dB of a persistent narrowband tone above f_lo vs its local spectral floor."""
    P = (S ** 2).mean(axis=0)
    sel = np.where(freqs >= f_lo)[0]
    if len(sel) < 80:
        return 0.0
    best = 0.0
    for i in sel[40:-40]:
        floor = np.median(np.concatenate([P[i - 40:i - 5], P[i + 5:i + 40]]))
        best = max(best, 10 * np.log10((P[i] + EPS) / (floor + EPS)))
    return float(best)


def clip_metrics(a: np.ndarray, sr: int) -> dict[str, float]:
    m = mono(a)
    S = stft_mag(m)
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    P = S ** 2

    out: dict[str, float] = {}
    # -- the current five (replicated, single stated convention) --
    out["rms"] = float(np.sqrt(np.mean(m ** 2)))
    out["rms_std"] = float(m.std())
    cen = (S * freqs).sum(axis=1) / (S.sum(axis=1) + EPS)
    out["centroid"] = float(cen.mean())
    out["hi4k"] = float((P[:, freqs > 4000].sum(axis=1) / (P.sum(axis=1) + EPS)).mean())
    out["flatness"] = float(np.median(np.exp(np.mean(np.log(P + EPS), axis=1)) / (P.mean(axis=1) + EPS)))
    out["crest"] = float(np.max(np.abs(m)) / (out["rms"] + EPS))

    # -- time structure --
    fdb = frame_rms_db(m, sr)
    q = len(fdb) // 4
    out["silent_frac"] = float((fdb < -60.0).mean())
    out["tail_delta_db"] = float(fdb[-q:].mean() - fdb[:q].mean())
    out["env_std_db"] = float(fdb.std())
    out["frame_rms_min_db"] = float(fdb.min())

    # -- stereo --
    if a.shape[1] >= 2:
        L, R = a[:, 0], a[:, 1]
        mid, side = (L + R) / 2, (L - R) / 2
        out["width_db"] = float(10 * np.log10((np.mean(side ** 2) + EPS) / (np.mean(mid ** 2) + EPS)))
        denom = L.std() * R.std() + EPS
        out["lr_corr"] = float(np.mean((L - L.mean()) * (R - R.mean())) / denom)
        rms_pc = np.sqrt(np.mean(a ** 2))  # per-channel power average
        out["mono_cancel_db"] = float(20 * np.log10((out["rms"] + EPS) / (rms_pc + EPS)))
    else:
        out["width_db"], out["lr_corr"], out["mono_cancel_db"] = 0.0, 1.0, 0.0

    # -- onsets / transients --
    oe = onset_env(S)
    thr = np.percentile(oe, 90)
    peaks = (oe[1:-1] > thr) & (oe[1:-1] >= oe[:-2]) & (oe[1:-1] >= oe[2:])
    dur = len(m) / sr
    out["onset_rate"] = float(peaks.sum() / dur)
    out["flux_p90_med"] = float((np.percentile(oe, 90) - np.percentile(oe, 50)))

    # -- pitch / harmony / noise --
    out["harmonicity"] = harmonicity(m, sr)

    # -- envelope modulation --
    out["pump_frac"], out["rough_frac"] = env_mod_fracs(m, sr)

    # -- perceptual weighting --
    kw = k_weight(freqs)
    out["loud_kw_db"] = float(10 * np.log10((P * kw).sum(axis=1).mean() + EPS))

    # -- HF artefact tone --
    out["hf_tonal_db"] = hf_tonal_peak_db(S, freqs)
    return out


# ---------------------------------------------------------------- pairwise

def best_lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[float, int]:
    """Max Pearson correlation of two 1-D z-scored series over lags in [-max_lag, max_lag]."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    best, blag = -1.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[lag:], y[:n - lag]
        else:
            a, b = x[:n + lag], y[-lag:]
        if len(a) < 8:
            continue
        c = float(np.dot(a - a.mean(), b - b.mean()) / ((a.std() * b.std() + EPS) * len(a)))
        if c > best:
            best, blag = c, lag
    return best, blag


def pair_metrics(a1: np.ndarray, a2: np.ndarray, sr: int) -> dict[str, float]:
    """Structure-preservation measures between two same-seed renders."""
    m1, m2 = mono(a1), mono(a2)
    n = min(len(m1), len(m2))
    m1, m2 = m1[:n], m2[:n]
    S1, S2 = stft_mag(m1), stft_mag(m2)
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)

    out: dict[str, float] = {}
    hop_ms = 512 / sr * 1000.0
    max_lag = max(1, int(round(150.0 / hop_ms)))

    c, lag = best_lag_corr(onset_env(S1), onset_env(S2), max_lag)
    out["onset_corr"], out["onset_lag_ms"] = c, lag * hop_ms

    e1, e2 = frame_rms_db(m1, sr), frame_rms_db(m2, sr)
    c, lag = best_lag_corr(e1, e2, int(150.0 / 11.6))
    out["env_corr"], out["env_lag_ms"] = c, lag * 11.6

    C1, C2 = chroma_gram(S1, freqs), chroma_gram(S2, freqs)
    v1, v2 = C1.mean(axis=0), C2.mean(axis=0)
    out["chroma_corr"] = float(np.corrcoef(v1, v2)[0, 1])
    nf = min(len(C1), len(C2))
    num = (C1[:nf] * C2[:nf]).sum(axis=1)
    den = np.linalg.norm(C1[:nf], axis=1) * np.linalg.norm(C2[:nf], axis=1) + EPS
    out["chromaseq_cos"] = float((num / den).mean())

    ls1 = np.log(np.maximum((S1 ** 2).mean(axis=0), EPS))
    ls2 = np.log(np.maximum((S2 ** 2).mean(axis=0), EPS))
    out["logspec_corr"] = float(np.corrcoef(ls1, ls2)[0, 1])
    return out


# ---------------------------------------------------------------- scan CLI

SCALE_RE = re.compile(r"_(minus|plus)([0-9.]+)$")


def parse_role(stem: str) -> tuple[str, float | None]:
    if "REF_prompt" in stem:
        # first-listed REF in every folder is the positive-pole caption
        return "ref", None
    if stem.endswith("_zero"):
        return "ladder", 0.0
    mt = SCALE_RE.search(stem)
    if mt:
        v = float(mt.group(2))
        return "ladder", -v if mt.group(1) == "minus" else v
    return "other", None


def listen_md_rms(folder: Path) -> dict[str, float]:
    md = folder / "LISTEN.md"
    out: dict[str, float] = {}
    if not md.exists():
        return out
    for line in md.read_text().splitlines():
        mt = re.search(r"`([^`]+\.wav)`\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|", line)
        if mt:
            out[mt.group(1)] = float(mt.group(2))
    return out


GROUP_RE = re.compile(r"-(seed|s)(\d+)$")


def scan(folders: list[Path], out_clips: Path, out_pairs: Path) -> None:
    clip_rows: list[dict] = []
    pair_rows: list[dict] = []
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    md_diffs: list[tuple[str, float, float]] = []

    for folder in sorted(folders):
        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            continue
        gm = GROUP_RE.search(folder.name)
        group = GROUP_RE.sub("", folder.name)
        seed = int(gm.group(2)) if gm else -1
        md_rms = listen_md_rms(folder)
        zero = None
        loaded: dict[str, tuple[np.ndarray, int]] = {}
        for wav in wavs:
            a, sr = load(wav)
            loaded[wav.name] = (a, sr)
            role, scale = parse_role(wav.stem)
            met = clip_metrics(a, sr)
            if wav.name in md_rms:
                md_diffs.append((f"{folder.name}/{wav.name}", md_rms[wav.name], met["rms_std"]))
            row = dict(folder=folder.name, group=group, seed=seed, file=wav.name,
                       role=role, scale="" if scale is None else scale,
                       sr=sr, n_samples=a.shape[0], n_ch=a.shape[1], **met)
            clip_rows.append(row)
            if scale == 0.0:
                zero = wav.name
        if zero is not None:
            za, zsr = loaded[zero]
            cache[folder / zero] = (za, zsr)
            for name, (a, sr) in loaded.items():
                if name == zero:
                    continue
                role, scale = parse_role(Path(name).stem)
                pm = pair_metrics(a, za, sr)
                pair_rows.append(dict(folder=folder.name, group=group, seed=seed,
                                      file=name, role=role,
                                      scale="" if scale is None else scale,
                                      kind="vs_zero", **pm))

    # cross-seed zero-vs-zero null pairs ("different song" anchor)
    by_group: dict[str, list[Path]] = {}
    for p in cache:
        gm = GROUP_RE.sub("", p.parent.name)
        by_group.setdefault(gm, []).append(p)
    for group, ps in by_group.items():
        ps = sorted(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                (a1, sr1), (a2, sr2) = cache[ps[i]], cache[ps[j]]
                if sr1 != sr2:
                    continue
                pm = pair_metrics(a1, a2, sr1)
                pair_rows.append(dict(folder=f"{ps[i].parent.name}|{ps[j].parent.name}",
                                      group=group, seed=-1, file="zero_x_seed",
                                      role="null", scale="", kind="null_xseed", **pm))

    for path, rows in [(out_clips, clip_rows), (out_pairs, pair_rows)]:
        if not rows:
            continue
        with open(path, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"wrote {len(rows)} rows -> {path}")

    if md_diffs:
        worst = max(md_diffs, key=lambda t: abs(t[1] - t[2]))
        print(f"LISTEN.md rms cross-check: {len(md_diffs)} rows, "
              f"worst |md - ours(std)| = {abs(worst[1]-worst[2]):.4f} on {worst[0]}")


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    import tempfile
    sr = 44100
    t = np.arange(2 * sr) / sr
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
        ok = ok and cond

    # 1. rms convention: full-scale sine -> 0.7071
    sine = np.sin(2 * np.pi * 440 * t)
    r = float(np.sqrt(np.mean(sine ** 2)))
    check("rms(sine amp 1.0) == 1/sqrt(2)", abs(r - 0.70710678) < 1e-3, f"got {r:.5f}")

    # 2. stereo loader: L=440 Hz sine, R=quiet noise; catches interleave-as-mono
    rng = np.random.default_rng(0)
    L, R = 0.5 * sine, 0.1 * rng.standard_normal(len(t))
    inter = np.empty(2 * len(t), dtype=np.int16)
    inter[0::2] = (L * 32767).astype(np.int16)
    inter[1::2] = (R * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        with wave.open(fh, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(inter.tobytes())
        tmp = Path(fh.name)
    a, sr2 = load(tmp)
    check("loader shape/sr", a.shape == (len(t), 2) and sr2 == sr, f"got {a.shape} sr={sr2}")
    check("channels not swapped/mixed",
          abs(a[:, 0].std() - L.std()) < 1e-3 and abs(a[:, 1].std() - R.std()) < 1e-2)
    tmp.unlink()

    # 2b. interleave-as-mono control on musical (low-freq) content in both channels:
    # reading interleaved stereo as mono fabricates content near Nyquist.
    Lm, Rm = 0.5 * np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 330 * t)
    inter2 = np.empty(2 * len(t))
    inter2[0::2], inter2[1::2] = Lm, Rm
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    S_good = stft_mag((Lm + Rm) / 2)
    cen_good = float(((S_good * freqs).sum(1) / (S_good.sum(1) + EPS)).mean())
    S_bug = stft_mag(inter2[: len(t)])
    cen_bug = float(((S_bug * freqs).sum(1) / (S_bug.sum(1) + EPS)).mean())
    check("interleave-as-mono control detects fabricated HF", cen_bug > 3 * cen_good,
          f"proper={cen_good:.0f} Hz, bugged={cen_bug:.0f} Hz")

    # 3. lag recovery: shifted copy of a click train
    m = np.zeros(4 * sr)
    m[:: sr // 2] = 1.0
    m = np.convolve(m, np.hanning(64), mode="same") + 0.01 * rng.standard_normal(len(m))
    shift = int(0.100 * sr)
    m2 = np.roll(m, shift)
    pm = pair_metrics(m[:, None], m2[:, None], sr)
    check("onset lag recovery ~100 ms", abs(abs(pm["onset_lag_ms"]) - 100) < 25,
          f"got {pm['onset_lag_ms']:.0f} ms, corr {pm['onset_corr']:.2f}")
    pm_self = pair_metrics(m[:, None], m[:, None], sr)
    check("self-pair corr==1 @ lag 0",
          pm_self["onset_corr"] > 0.999 and pm_self["onset_lag_ms"] == 0.0)

    # 4. chroma: A major triad vs itself and vs a disjoint (Eb major) triad
    triad = sum(np.sin(2 * np.pi * f * t) for f in (440.0, 554.37, 659.25))
    triad_eb = sum(np.sin(2 * np.pi * f * t) for f in (622.25, 783.99, 932.33))
    noise = rng.standard_normal(len(t))
    pm_t = pair_metrics(triad[:, None], triad[:, None], sr)
    pm_x = pair_metrics(triad[:, None], triad_eb[:, None], sr)
    check("chroma self high, disjoint triad low",
          pm_t["chroma_corr"] > 0.99 and pm_x["chroma_corr"] < 0.0,
          f"self={pm_t['chroma_corr']:.3f} disjoint={pm_x['chroma_corr']:.3f}")

    # 5. harmonicity: sine ~1, noise ~low
    h_sine = harmonicity(sine, sr)
    h_noise = harmonicity(noise, sr)
    check("harmonicity sine>0.9, noise<0.4", h_sine > 0.9 and h_noise < 0.4,
          f"sine={h_sine:.2f} noise={h_noise:.2f}")

    # 6. pumping: 4 Hz amplitude-modulated noise scores higher than steady noise
    am = (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)) * noise
    p_am, _ = env_mod_fracs(am, sr)
    p_st, _ = env_mod_fracs(noise, sr)
    check("pump_frac AM(4Hz) >> steady", p_am > 5 * max(p_st, 1e-4),
          f"am={p_am:.3f} steady={p_st:.4f}")

    # 7. silence detection
    half = np.concatenate([0.3 * noise[: len(t) // 2], np.zeros(len(t) // 2)])
    met = clip_metrics(half[:, None], sr)
    check("silent_frac sees half-dead clip", 0.35 < met["silent_frac"] < 0.65,
          f"got {met['silent_frac']:.2f}")

    # 8. HF tonal peak: 12 kHz whistle in noise
    whistle = 0.05 * np.sin(2 * np.pi * 12000 * t) + 0.1 * noise
    met_w = clip_metrics(whistle[:, None], sr)
    met_n = clip_metrics((0.1 * noise)[:, None], sr)
    check("hf_tonal_db flags 12 kHz whistle", met_w["hf_tonal_db"] > met_n["hf_tonal_db"] + 10,
          f"whistle={met_w['hf_tonal_db']:.1f} dB noise={met_n['hf_tonal_db']:.1f} dB")

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sc = sub.add_parser("scan")
    sc.add_argument("folders", nargs="+", type=Path)
    sc.add_argument("--out-clips", type=Path, required=True)
    sc.add_argument("--out-pairs", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    scan([f for f in args.folders if f.is_dir()], args.out_clips, args.out_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
