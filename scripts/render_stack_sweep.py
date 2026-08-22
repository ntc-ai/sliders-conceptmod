#!/usr/bin/env python3
"""Render whole slider *stacks* the way the studio resolves them, and measure them.

`render_shipped_slider.py` answers "does one slider sound right at ±2". This
answers the next question: how many normalized sliders can a user pile on before
the render falls apart. It resolves a list of `slider_id:fader` pairs through
`app/sliders.json` exactly as a job would, renders one clip per stack against a
fixed neutral caption/lyrics/seed, and reports rms and crest factor (same
convention as `probe_axis.py` / `probe.json`) relative to the neutral baseline.

The combined budget is deliberately *off* by default — this tool is what the
budget gets measured with. Pass --apply_budget to hear the budgeted result.

  python scripts/render_stack_sweep.py \
      --prompts_file conceptmod/textsliders/data/prompts-triphop-v3-single.yaml \
      --out_dir eval/listen/stack-sweep-20s \
      --stack "" \
      --stack "energy:+1,space:-0.5" \
      --stack "energy:+1,distortion:-1,space:-0.5"
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

for path in (str(ROOT), str(APP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from conceptmod.textsliders.generate_gender_stack import _apply, _wrap_sidecar  # noqa: E402
from conceptmod.textsliders.generate_listen import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    _load_prompt_row,
    _write_wav,
)
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402
from scripts.compare_slider import load_mono  # noqa: E402
from scripts.probe_axis import crest_db  # noqa: E402


def parse_stack(spec: str) -> list[dict]:
    """'energy:+1,distortion:-1.5' -> [{'id': 'energy', 'scale': 1.0}, ...]"""
    settings = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"bad stack item {part!r} (want slider_id:fader)")
        slider_id, _, fader = part.partition(":")
        settings.append({"id": slider_id.strip(), "scale": float(fader)})
    return settings


def measure(path: Path) -> tuple[float, float]:
    """(rms, crest_db) with probe_axis.py's convention."""
    y, _sr = load_mono(path)
    return float(np.sqrt(np.mean(np.square(y)))), crest_db(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--stack",
        action="append",
        default=[],
        metavar="ID:FADER,...",
        help="one render per --stack; an empty string is the neutral baseline",
    )
    parser.add_argument("--label", action="append", default=[], help="optional name per --stack")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="", help="filename/label suffix, e.g. a second caption+seed pass")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--apply_budget", action="store_true", help="honour sliders.json combined_budget")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from app import sliders as registry

    if not args.stack:
        raise SystemExit("need at least one --stack")

    row = _load_prompt_row(Path(args.prompts_file), row=args.row)
    neutral = str(row.get("neutral") or row["target"])
    lyrics = str(row.get("lyrics") or "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve every stack up front: a typo should fail before the model loads.
    jobs = []
    for index, spec in enumerate(args.stack):
        settings = registry.validate_settings(parse_stack(spec))
        kwargs = {} if args.apply_budget else {"budget": None}
        comps = registry.resolve(settings, **kwargs) if settings else []
        total = sum(abs(comp["multiplier"]) for comp in comps)
        label = args.label[index] if index < len(args.label) else (spec or "baseline")
        suffix = f"_{args.tag}" if args.tag else ""
        name = f"{index:02d}_total{total:04.1f}{suffix}.wav"
        jobs.append((name, label, spec, settings, comps, total))

    print(f"{'file':28s} {'total':>6s}  stack")
    for name, _label, spec, _settings, _comps, total in jobs:
        print(f"{name:28s} {total:6.2f}  {spec or '(neutral)'}")

    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    networks: dict[str, object] = {}
    results = []

    for name, label, spec, settings, comps, total in jobs:
        dest = out_dir / name
        pairs = []
        for comp in comps:
            key = comp["weights"]
            if key not in networks:
                networks[key], _meta = _wrap_sidecar(pipe, device, Path(key), comp["kind"])
            pairs.append((networks[key], comp["multiplier"]))
        applied = ", ".join(
            f"{Path(c['weights']).name.split('_')[0]}x{c['multiplier']:+.2f}" for c in comps
        ) or "none"
        print(f"{name}: total={total:.2f} {applied}", flush=True)

        if dest.exists() and not args.force:
            print(f"skip existing {name}", flush=True)
        else:
            generator = torch.Generator(device).manual_seed(int(args.seed))
            with _apply(*pairs):
                audio = pipe(
                    prompt=neutral,
                    lyrics=lyrics,
                    audio_duration=float(args.duration),
                    generator=generator,
                    output="audios",
                )[0]
            _write_wav(dest, audio, int(pipe.sampling_rate), float(args.duration))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rms, crest = measure(dest)
        results.append(
            {
                "file": name,
                "label": label,
                "stack": spec,
                "settings": settings,
                "total": round(total, 4),
                "components": len(comps),
                "rms": round(rms, 5),
                "crest_db": round(crest, 2),
                "applied": applied,
            }
        )
        print(f"  rms={rms:.5f} crest={crest:.2f} dB", flush=True)

    base = next((r for r in results if r["total"] == 0.0), None)
    for row_out in results:
        if base and base["rms"] > 0:
            row_out["rms_ratio"] = round(row_out["rms"] / base["rms"], 3)
            row_out["crest_delta_db"] = round(row_out["crest_db"] - base["crest_db"], 2)

    suffix = f"-{args.tag}" if args.tag else ""
    report = out_dir / f"sweep{suffix}.json"
    report.write_text(
        json.dumps(
            {
                "prompts_file": args.prompts_file,
                "row": args.row,
                "seed": args.seed,
                "duration": args.duration,
                "apply_budget": args.apply_budget,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n{'file':28s} {'total':>6s} {'rms':>8s} {'x base':>7s} {'crest':>7s} {'d crest':>8s}  stack")
    for r in results:
        print(
            f"{r['file']:28s} {r['total']:6.2f} {r['rms']:8.5f} "
            f"{r.get('rms_ratio', float('nan')):7.3f} {r['crest_db']:7.2f} "
            f"{r.get('crest_delta_db', float('nan')):8.2f}  {r['stack'] or '(neutral)'}"
        )
    print(f"wrote {report}", flush=True)


if __name__ == "__main__":
    main()
