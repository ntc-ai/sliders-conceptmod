#!/usr/bin/env python3
"""Range-sample one shipped slider and mix a continuous sweep.

Each scale is a real LoRA render (same caption, lyrics, seed). The mix plays
the same song-time from every take and equal-power crossfades between adjacent
scales, so you hear the axis move instead of a two-pole dip.

  CUDA_VISIBLE_DEVICES=1 $PY scripts/render_slider_sweep.py energy \
      --prompts_file conceptmod/textsliders/data/prompts-energy-v4.yaml \
      --out_dir eval/listen/energy-sweep-8s \
      --start -1 --stop 1 --step 0.25 --duration 8 --device 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path("/ml2/music")


def _early_visible_device(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


os.environ["CUDA_VISIBLE_DEVICES"] = _early_visible_device(sys.argv[1:])
os.environ.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTHONPATH", str(ROOT))

for path in (str(ROOT), str(APP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from conceptmod.textsliders.generate_gender_stack import _apply, _wrap_sidecar  # noqa: E402
from conceptmod.textsliders.generate_listen import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    _accept_wav,
    _load_prompt_row,
    _write_wav,
)
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402


def _scale_grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round((stop - start) / step))
    scales = [round(start + i * step, 6) for i in range(n + 1)]
    if abs(scales[-1] - stop) > 1e-6:
        scales.append(round(stop, 6))
    return scales


def _scale_name(index: int, scale: float, plus: str, minus: str) -> str:
    if scale > 0:
        return f"{index:02d}_scale_{plus}_plus{scale:g}.wav"
    if scale < 0:
        return f"{index:02d}_scale_{minus}_minus{abs(scale):g}.wav"
    return f"{index:02d}_scale_zero.wav"


def mix_sweep(
    paths: list[Path],
    scales: list[float],
    dest: Path,
    hold: float,
    sweep: float,
) -> None:
    """Play the same song-time from every take; crossfade by slider position."""
    arrays = []
    sr = None
    for path in paths:
        audio, this_sr = sf.read(str(path), always_2d=True)
        if sr is None:
            sr = int(this_sr)
        elif int(this_sr) != sr:
            raise RuntimeError(f"sample-rate mismatch {path}: {this_sr} vs {sr}")
        arrays.append(audio.astype(np.float64))
    n_samples = min(a.shape[0] for a in arrays)
    n_ch = arrays[0].shape[1]
    stacked = np.stack([a[:n_samples] for a in arrays], axis=0)  # S, T, C

    total = hold + sweep + hold
    out_n = min(n_samples, int(round(total * sr)))
    n_scales = len(scales)
    out = np.zeros((out_n, n_ch), dtype=np.float64)
    # Map output time -> slider position in [0, n-1], hold at the poles.
    for i in range(out_n):
        t = i / sr
        if t <= hold:
            pos = 0.0
        elif t >= hold + sweep:
            pos = float(n_scales - 1)
        else:
            pos = (t - hold) / sweep * (n_scales - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, n_scales - 1)
        frac = pos - lo
        # equal-power
        w_hi = np.sin(frac * np.pi / 2.0)
        w_lo = np.cos(frac * np.pi / 2.0)
        out[i] = w_lo * stacked[lo, i] + w_hi * stacked[hi, i]

    peak = np.max(np.abs(out))
    if peak > 0.98:
        out *= 0.98 / peak
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), out.astype(np.float32), sr, format="WAV")
    rms = float(np.sqrt(np.mean(np.square(out))))
    print(f"wrote sweep {dest} duration={out_n / sr:.2f}s rms={rms:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slider_id")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--start", type=float, default=-1.0)
    parser.add_argument("--stop", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hold", type=float, default=1.5)
    parser.add_argument("--sweep", type=float, default=5.0)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mix_only", action="store_true")
    args = parser.parse_args()

    from app import sliders as registry

    catalog = {s["id"]: s for s in registry.catalog()["sliders"]}
    if args.slider_id not in catalog:
        raise SystemExit(f"unknown slider '{args.slider_id}' (have: {', '.join(sorted(catalog))})")
    definition = catalog[args.slider_id]
    plus, minus = definition["label_plus"], definition["label_minus"]

    row = _load_prompt_row(Path(args.prompts_file), row=args.row)
    neutral = str(row.get("neutral") or row["target"])
    lyrics = str(row.get("lyrics") or "")
    scales = _scale_grid(args.start, args.stop, args.step)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dests: list[Path] = []
    rows_md = []

    if not args.mix_only:
        device = "cuda:0"
        pipe = _load_pipeline(Path(args.model_dir), device)
        networks: dict[str, object] = {}

        for index, scale in enumerate(scales, start=1):
            name = _scale_name(index, scale, plus, minus)
            dest = out_dir / name
            dests.append(dest)

            comps = registry.resolve([{"id": args.slider_id, "scale": scale}]) if scale else []
            pairs = []
            for comp in comps:
                key = comp["weights"]
                if key not in networks:
                    networks[key], _meta = _wrap_sidecar(pipe, device, Path(key), comp["kind"])
                pairs.append((networks[key], comp["multiplier"]))
            applied = (
                ", ".join(
                    f"{Path(c['weights']).name.split('_')[0]}x{c['multiplier']:+.2f}" for c in comps
                )
                or "none"
            )
            print(f"{name}: scale={scale:+g}  {applied}", flush=True)

            ok, _reason, duration, rms = _accept_wav(dest, float(args.duration))
            if ok and not args.force:
                print(f"skip existing {name}", flush=True)
                rows_md.append(f"| `{name}` | {scale:+g} | {duration:.2f} | {rms:.4f} | {applied} |")
                continue

            generator = torch.Generator(device).manual_seed(int(args.seed))
            with _apply(*pairs):
                audio = pipe(
                    prompt=neutral,
                    lyrics=lyrics,
                    audio_duration=float(args.duration),
                    generator=generator,
                    output="audios",
                )[0]
            duration, rms = _write_wav(dest, audio, int(pipe.sampling_rate), float(args.duration))
            rows_md.append(f"| `{name}` | {scale:+g} | {duration:.2f} | {rms:.4f} | {applied} |")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        for index, scale in enumerate(scales, start=1):
            dests.append(out_dir / _scale_name(index, scale, plus, minus))

    missing = [p for p in dests if not p.exists()]
    if missing:
        raise SystemExit(f"missing takes: {', '.join(p.name for p in missing)}")

    sweep_path = out_dir / f"{args.slider_id}_sweep_{args.start:g}_to_{args.stop:g}.wav"
    mix_sweep(dests, scales, sweep_path, hold=float(args.hold), sweep=float(args.sweep))

    meta = {
        "slider_id": args.slider_id,
        "scales": scales,
        "duration": args.duration,
        "seed": args.seed,
        "row": args.row,
        "prompts_file": str(args.prompts_file),
        "hold": args.hold,
        "sweep": args.sweep,
        "files": [p.name for p in dests],
        "mix": sweep_path.name,
    }
    (out_dir / "sweep.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if rows_md:
        (out_dir / "LISTEN.md").write_text(
            "\n".join(
                [
                    f"# {args.slider_id} LoRA range sweep",
                    "",
                    "Same caption, lyrics and seed. Only the shipped slider scale changes.",
                    "",
                    "| file | scale | seconds | rms | applied |",
                    "|------|------:|--------:|----:|---------|",
                    *rows_md,
                    "",
                    f"Mix: `{sweep_path.name}`  hold={args.hold}s sweep={args.sweep}s",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    print(f"done  {len(dests)} takes  mix={sweep_path}", flush=True)


if __name__ == "__main__":
    main()
