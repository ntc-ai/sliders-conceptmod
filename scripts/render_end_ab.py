#!/usr/bin/env python3
"""Same-seed A/B of song endings: base model vs LoRA variants.

Renders one caption+lyrics with one fixed seed, once without any slider and
once per --variant, and reports each cut's length and tail loudness. A natural
ending finishes short of the requested duration with a near-silent tail
(tail/overall well under 0.1); a truncated cut sits at the frame cap and stops
at full loudness. Built to verify the audio-end regularizer in
train_lm_slider_music3.py against the un-regularized shipped checkpoints:

  python scripts/render_end_ab.py --song 5ec87fed --duration 60 --seed 7 \
      --variant "live-v3+2=models/live-lm-v3/live-lm-v3_last.safetensors:2" \
      --variant "live-v4+2=models/live-lm-v4/live-lm-v4_last.safetensors:2" \
      --out_dir eval/listen/live-endreg-ab-60s --device 1

--song takes a library song id (prefix is enough) and reuses its caption and
lyrics; --prompts_file/--row works like the other render scripts. One seed is
one sample — endings are sampled, so sweep a few seeds before concluding.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path("/ml2/music")
LIBRARY = APP_ROOT / "library"


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
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from conceptmod.textsliders.generate_gender_stack import _apply, _wrap_sidecar  # noqa: E402
from conceptmod.textsliders.generate_listen import DEFAULT_MODEL_DIR, _load_prompt_row  # noqa: E402
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402


def _parse_variant(spec: str) -> tuple[str, Path, float]:
    """'label=path:scale' -> (label, absolute weights path, scale)."""
    label, _, rest = spec.partition("=")
    path_str, _, scale_str = rest.rpartition(":")
    if not label or not path_str or not scale_str:
        raise SystemExit(f"--variant must look like label=path:scale, got {spec!r}")
    weights = Path(path_str)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        raise SystemExit(f"variant '{label}': weights not found at {weights}")
    return label, weights, float(scale_str)


def _song_text(song_prefix: str) -> tuple[str, str, float]:
    matches = sorted(glob.glob(str(LIBRARY / f"{song_prefix}*" / "meta.json")))
    if len(matches) != 1:
        raise SystemExit(f"--song {song_prefix!r} matched {len(matches)} library songs, need exactly 1")
    meta = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    return str(meta["caption"]), str(meta["lyrics"]), float(meta.get("duration_sec") or 60.0)


def _save_and_measure(dest: Path, audio, sample_rate: int, requested: float) -> dict:
    if torch.is_tensor(audio):
        wav = audio.detach().T.float().cpu().numpy()
    else:
        wav = np.asarray(audio, dtype=np.float32).T
    wav = np.ascontiguousarray(wav)
    sf.write(str(dest), wav, sample_rate)
    mono = wav.mean(axis=1) if wav.ndim > 1 else wav
    overall = float(np.sqrt(np.mean(mono**2)))
    tail = mono[-int(sample_rate * 0.5) :]
    tail_ratio = float(np.sqrt(np.mean(tail**2)) / overall) if overall > 0 else 0.0
    duration = float(mono.shape[0] / sample_rate)
    return {
        "seconds": duration,
        "rms": overall,
        "tail_ratio": tail_ratio,
        # The frame cap leaves ~0.1 s of decoder padding past the request, so
        # anything not clearly short of it counted as capped.
        "ended": duration < requested - 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--song", help="library song id (prefix ok): reuse its caption and lyrics")
    source.add_argument("--prompts_file", help="prompt YAML, same format as the other render scripts")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--variant", action="append", default=[], help="label=weights_path:scale (repeatable)")
    parser.add_argument("--duration", type=float, default=None, help="requested seconds (default: the song's length)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_base", action="store_true", help="render only the variants")
    args = parser.parse_args()

    if args.song:
        caption, lyrics, song_seconds = _song_text(args.song)
        requested = float(args.duration or round(song_seconds))
    else:
        row = _load_prompt_row(Path(args.prompts_file), row=args.row)
        caption = str(row.get("neutral") or row["target"])
        lyrics = str(row.get("lyrics") or "")
        requested = float(args.duration or 60.0)

    variants = [_parse_variant(spec) for spec in args.variant]
    if not variants and args.skip_base:
        raise SystemExit("nothing to render: no --variant and --skip_base")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    sample_rate = int(pipe.sampling_rate)

    jobs: list[tuple[str, object, float]] = [] if args.skip_base else [("base", None, 0.0)]
    for label, weights, scale in variants:
        network, _meta = _wrap_sidecar(pipe, device, weights, "language_model")
        jobs.append((label, network, scale))

    rows_md = []
    for index, (label, network, scale) in enumerate(jobs, start=1):
        name = f"{index:02d}_{label}.wav"
        print(f"rendering {name} (scale {scale:+g}, seed {args.seed}, {requested:g}s cap)…", flush=True)
        generator = torch.Generator(device).manual_seed(int(args.seed))
        pairs = [] if network is None else [(network, scale)]
        with _apply(*pairs):
            audio = pipe(
                prompt=caption,
                lyrics=lyrics,
                audio_duration=requested,
                generator=generator,
                output="audios",
            )[0]
        stats = _save_and_measure(out_dir / name, audio, sample_rate, requested)
        verdict = "ended naturally" if stats["ended"] else "TRUNCATED at cap"
        print(
            f"  {name}: {stats['seconds']:.2f}s tail/overall {stats['tail_ratio']:.3f} — {verdict}",
            flush=True,
        )
        rows_md.append(
            f"| `{name}` | {scale:+g} | {stats['seconds']:.2f} | {stats['tail_ratio']:.3f} "
            f"| {'yes' if stats['ended'] else '**no — cap**'} |"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_line = f"library song `{args.song}`" if args.song else f"`{args.prompts_file}` row {args.row}"
    (out_dir / "LISTEN.md").write_text(
        "\n".join(
            [
                "# Same-seed ending A/B",
                "",
                f"Source: {source_line}. Seed {args.seed}, requested {requested:g}s — identical for every row,",
                "so the only difference is the applied LoRA. tail/overall well under 0.1 means the cut",
                "faded out on its own; a hot tail at the cap means the composer never sampled",
                "`<|audio_end|>` and was guillotined.",
                "",
                "| file | scale | seconds | tail/overall | ended naturally |",
                "|------|------:|--------:|-------------:|-----------------|",
                *rows_md,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'LISTEN.md'}", flush=True)


if __name__ == "__main__":
    main()
