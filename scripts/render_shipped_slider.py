#!/usr/bin/env python3
"""Render a slider exactly as the studio ships it.

Listen sets from generate_listen.py exercise one weights file at a time, but the
app resolves a user-facing slider through app/sliders.json: ratios, gains, the
sidecar unit_scale, and possibly several LoRA networks at once. This renders the
resolved multipliers so what you hear is what the app produces.

  python scripts/render_shipped_slider.py energy \
      --prompts_file conceptmod/textsliders/data/prompts-energy-v3.yaml \
      --out_dir eval/listen/energy-v3-shipped-20s --scales=-2,0,2
"""

from __future__ import annotations

import argparse
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

import torch  # noqa: E402

from conceptmod.textsliders.generate_gender_stack import _apply, _wrap_sidecar  # noqa: E402
from conceptmod.textsliders.generate_listen import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    _accept_wav,
    _glue_negative_option,
    _load_prompt_row,
    _write_wav,
)
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402


def main() -> None:
    argv = _glue_negative_option(list(sys.argv[1:]), "--scales")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slider_id")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scales", default="-2,0,2")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    from app import sliders as registry

    catalog = {s["id"]: s for s in registry.catalog()["sliders"]}
    if args.slider_id not in catalog:
        raise SystemExit(f"unknown slider '{args.slider_id}' (have: {', '.join(sorted(catalog))})")
    definition = catalog[args.slider_id]
    plus, minus = definition["label_plus"], definition["label_minus"]

    row = _load_prompt_row(Path(args.prompts_file), row=args.row)
    neutral = str(row.get("neutral") or row["target"])
    lyrics = str(row.get("lyrics") or "")
    scales = [float(part) for part in args.scales.split(",") if part.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    networks: dict[str, object] = {}
    rows_md = []

    for index, scale in enumerate(scales, start=1):
        if scale > 0:
            name = f"{index:02d}_shipped_{plus}_plus{scale:g}.wav"
        elif scale < 0:
            name = f"{index:02d}_shipped_{minus}_minus{abs(scale):g}.wav"
        else:
            name = f"{index:02d}_shipped_zero.wav"
        dest = out_dir / name

        comps = registry.resolve([{"id": args.slider_id, "scale": scale}]) if scale else []
        pairs = []
        for comp in comps:
            key = comp["weights"]
            if key not in networks:
                networks[key], _meta = _wrap_sidecar(pipe, device, Path(key), comp["kind"])
            pairs.append((networks[key], comp["multiplier"]))
        applied = ", ".join(f"{Path(c['weights']).name.split('_')[0]}x{c['multiplier']:+.2f}" for c in comps) or "none"
        print(f"{name}: {applied}", flush=True)

        ok, _reason, duration, rms = _accept_wav(dest, float(args.duration))
        if ok and not args.force:
            print(f"skip existing {name}", flush=True)
            rows_md.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | {applied} |")
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
        rows_md.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | {applied} |")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    readme = out_dir / "LISTEN.md"
    readme.write_text(
        "\n".join(
            [
                f"# {args.slider_id} — exactly what the studio ships",
                "",
                f"Resolved through `app/sliders.json` ({minus} <-> {plus}), so the multipliers",
                "below include each component's ratio, gain and sidecar `unit_scale`.",
                "Same neutral caption, lyrics and seed throughout.",
                "",
                "| file | seconds | rms | applied |",
                "|------|--------:|----:|---------|",
                *rows_md,
                "",
                f"- seed: {args.seed}  duration: {args.duration}s  prompt row: {args.row}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {readme}", flush=True)


if __name__ == "__main__":
    main()
