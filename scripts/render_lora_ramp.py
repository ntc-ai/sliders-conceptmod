#!/usr/bin/env python3
"""One generate, LoRA gain ramps over audio time.

Only the transformer half is ramped: the flow transformer sees the whole clip
every denoise step, so a per-frame envelope is a real fader. The LM half stays
off so the plan does not change mid-song.

  CUDA_VISIBLE_DEVICES=1 $PY scripts/render_lora_ramp.py energy \
      --prompts_file conceptmod/textsliders/data/prompts-energy-v4.yaml \
      --out eval/listen/energy-ramp-8s/energy_ramp.wav \
      --start -1 --stop 1 --duration 8 --device 1
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

for path in (str(ROOT), str(APP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

from conceptmod.textsliders.generate_gender_stack import _apply, _wrap_sidecar  # noqa: E402
from conceptmod.textsliders.generate_listen import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    _load_prompt_row,
    _write_wav,
)
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402


def hold_lerp_hold(n: int, start: float, stop: float, hold: float, sweep: float) -> torch.Tensor:
    total = hold + sweep + hold
    t = torch.linspace(0.0, total, n)
    env = torch.empty(n)
    for i, ti in enumerate(t):
        if ti <= hold:
            env[i] = start
        elif ti >= hold + sweep:
            env[i] = stop
        else:
            u = (ti - hold) / sweep
            u = u * u * (3 - 2 * u)
            env[i] = start + (stop - start) * u
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slider_id")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=-1.0)
    parser.add_argument("--stop", type=float, default=1.0)
    parser.add_argument("--hold", type=float, default=1.5)
    parser.add_argument("--sweep", type=float, default=5.0)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()

    from app import sliders as registry

    catalog = {s["id"]: s for s in registry.catalog()["sliders"]}
    if args.slider_id not in catalog:
        raise SystemExit(f"unknown slider '{args.slider_id}'")

    row = _load_prompt_row(Path(args.prompts_file), row=args.row)
    neutral = str(row.get("neutral") or row["target"])
    lyrics = str(row.get("lyrics") or "")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve at user-scale 1 to get the baked unit/ratio, then the envelope
    # carries the sign and the slide. LM halves are dropped: ramping them
    # over AR tokens would rewrite the plan, not the mix.
    comps = registry.resolve([{"id": args.slider_id, "scale": 1.0}])
    tf_comps = [c for c in comps if c["kind"] != "language_model"]
    if not tf_comps:
        raise SystemExit(f"{args.slider_id} has no transformer half to ramp")

    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)

    pairs = []
    applied = []
    curve = hold_lerp_hold(256, args.start, args.stop, args.hold, args.sweep)
    for comp in tf_comps:
        net, _meta = _wrap_sidecar(pipe, device, Path(comp["weights"]), comp["kind"])
        # multiplier at user-scale 1; envelope is the user-scale path
        net.set_seq_gain(curve)
        pairs.append((net, comp["multiplier"]))
        applied.append(f"{Path(comp['weights']).name} x{comp['multiplier']:+.3f} * env[{args.start:g}→{args.stop:g}]")
        print(applied[-1], flush=True)

    generator = torch.Generator(device).manual_seed(int(args.seed))
    with _apply(*pairs):
        audio = pipe(
            prompt=neutral,
            lyrics=lyrics,
            audio_duration=float(args.duration),
            generator=generator,
            output="audios",
        )[0]
    duration, rms = _write_wav(out, audio, int(pipe.sampling_rate), float(args.duration))

    meta = {
        "slider_id": args.slider_id,
        "method": "transformer_seq_gain",
        "start": args.start,
        "stop": args.stop,
        "hold": args.hold,
        "sweep": args.sweep,
        "duration": duration,
        "rms": rms,
        "seed": args.seed,
        "lm": "off",
        "applied": applied,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} duration={duration:.2f}s rms={rms:.4f}", flush=True)


if __name__ == "__main__":
    main()
