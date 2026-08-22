"""Generate a labeled listen set: slider scales plus prompt-only references.

Filenames are numbered and spell out the intended effect so a folder listing
is enough to A/B the slider.

Existing valid wavs are reused (resume-safe). Pass --force to regenerate.
Each written wav is checked for duration and silence before the run is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _early_visible_device(argv: list[str]) -> str:
    """Which physical GPU to expose, honouring a caller-set CUDA_VISIBLE_DEVICES.

    --device used to be written straight into CUDA_VISIBLE_DEVICES, so
    `CUDA_VISIBLE_DEVICES=1 ... --device 0` silently ran on physical GPU 0 --
    the exact opposite of what the caller asked for, and a very quiet failure
    when that GPU is busy. --device indexes the visible list (as the load error
    already claims it does); it does not override it.
    """
    requested = None
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            requested = argv[i + 1]
            break
        if arg.startswith("--device="):
            requested = arg.split("=", 1)[1]
            break
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return requested if requested is not None else "0"
    devices = [d for d in visible.split(",") if d.strip()]
    if requested is None:
        return devices[0]
    try:
        return devices[int(requested)]
    except (ValueError, IndexError):
        return requested


os.environ["CUDA_VISIBLE_DEVICES"] = _early_visible_device(sys.argv[1:])

import numpy as np
import soundfile as sf
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.infer_music3 import (  # noqa: E402
    _load_pipeline,
    _to_wav_array,
)
from conceptmod.textsliders.lora import LoRANetwork  # noqa: E402

DEFAULT_MODEL_DIR = Path("/ml2/music/models/MiniMax-Music3")
TRANSFORMER_REPLACE = ["MiniMaxMusic3Attention"]
LM_REPLACE = ["Qwen3Attention"]
MIN_RMS = 1e-3
DURATION_TOLERANCE = 0.90


def _glue_negative_option(argv: list[str], flag: str) -> list[str]:
    glued: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            glued.append(f"{flag}={argv[i + 1]}")
            i += 2
            continue
        glued.append(argv[i])
        i += 1
    return glued


def _load_prompt_row(path: Path, row: int = 0) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows")
    if not raw:
        raise ValueError(f"empty prompts: {path}")
    return raw[int(row) % len(raw)]


def _sidecar(weights: Path) -> dict:
    # Only the final save writes a sidecar, so `_best` / `_step123` checkpoints
    # fall back to the run's `_last.json`. Without this the rank/alpha/
    # target_replace defaults silently win and the LoRA fails to load.
    import re as _re

    candidates = [weights.with_suffix(".json"), Path(str(weights) + ".json")]
    base = _re.sub(r"_(best|step\d+)$", "_last", weights.stem)
    if base != weights.stem:
        candidates.append(weights.parent / f"{base}.json")
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _scale_name(index: int, scale: float, plus_label: str, minus_label: str) -> str:
    if scale > 0:
        effect = plus_label
        tag = f"plus{scale:g}"
    elif scale < 0:
        effect = minus_label
        tag = f"minus{abs(scale):g}"
    else:
        effect = "neutral_base"
        tag = "zero"
    return f"{index:02d}_slider_{effect}_{tag}.wav"


def _inspect_wav(path: Path) -> tuple[float, float]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    duration = float(audio.shape[0]) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    return duration, rms


def _accept_wav(
    path: Path, requested: float, accept_silent: bool = False, accept_short: bool = False
) -> tuple[bool, str, float, float]:
    if not path.exists() or path.stat().st_size < 1024:
        return False, "missing or tiny", 0.0, 0.0
    try:
        duration, rms = _inspect_wav(path)
    except Exception as exc:  # noqa: BLE001 — surface any soundfile failure
        return False, f"unreadable ({exc})", 0.0, 0.0
    if duration < requested * DURATION_TOLERANCE and not accept_short:
        return False, f"short {duration:.2f}s < {requested * DURATION_TOLERANCE:.2f}s", duration, rms
    if rms < MIN_RMS and not accept_silent:
        return False, f"silent rms={rms:.6f}", duration, rms
    return True, "ok", duration, rms


def _write_wav(
    path: Path, audio, sample_rate: int, requested: float,
    accept_silent: bool = False, accept_short: bool = False,
) -> tuple[float, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _to_wav_array(audio)
    tmp = path.with_name(path.name + ".tmp.wav")
    sf.write(str(tmp), array, sample_rate, format="WAV")
    ok, reason, duration, rms = _accept_wav(tmp, requested, accept_silent=accept_silent, accept_short=accept_short)
    if not ok:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"rejected {path.name}: {reason}")
    if rms < MIN_RMS:
        # --accept_silent: a silent render is EVIDENCE for the scorer (the
        # silence gate must see it), not an error and never a seed retry.
        print(f"KEEPING SILENT RENDER {path.name} rms={rms:.6f}", flush=True)
    if duration < requested * DURATION_TOLERANCE:
        # --accept_short: an early <|audio_end|> is the model ending the song
        # naturally before the cap — sampled variation, not a broken render.
        print(f"KEEPING SHORT RENDER {path.name} duration={duration:.2f}s", flush=True)
    tmp.replace(path)
    print(f"wrote {path} duration={duration:.2f}s rms={rms:.4f}", flush=True)
    return duration, rms


def _jobs(args: argparse.Namespace, row: dict) -> list[tuple[Path, str, float | None]]:
    out_dir = Path(args.out_dir)
    scales = [float(part) for part in args.scales.split(",") if part.strip()]
    plus_label = args.plus_label
    minus_label = args.minus_label
    lyrics = str(row.get("lyrics") or "")
    neutral = str(row.get("neutral") or row.get("target"))
    jobs: list[tuple[Path, str, float | None]] = []
    for index, scale in enumerate(scales, start=1):
        dest = out_dir / _scale_name(index, scale, plus_label, minus_label)
        jobs.append((dest, neutral, float(scale)))
    jobs.append(
        (
            out_dir / f"{len(scales)+1:02d}_REF_prompt_{plus_label}_no_slider.wav",
            str(row["positive"]),
            None,
        )
    )
    jobs.append(
        (
            out_dir / f"{len(scales)+2:02d}_REF_prompt_{minus_label}_no_slider.wav",
            str(row["negative"]),
            None,
        )
    )
    return jobs


def _unit_scale(meta: dict, raw_scales: bool) -> float | None:
    if raw_scales:
        return None
    value = meta.get("unit_scale")
    if value is None:
        return None
    scale = float(value)
    if scale <= 0 or scale != scale:  # noqa: PLR0124 — NaN
        return None
    if meta.get("axis_tracking_low") or scale > 4.0:
        print(
            f"WARNING: sidecar unit_scale={scale:g} looks untrustworthy "
            f"(axis_tracking_low={bool(meta.get('axis_tracking_low'))}); using raw scales",
            flush=True,
        )
        return None
    return scale


def _write_readme(
    args: argparse.Namespace,
    weights: Path,
    row: dict,
    scales: list[float],
    stats: dict[str, tuple[float, float]],
    rank: int,
    alpha: float,
    unit_scale: float | None = None,
) -> None:
    plus_label = args.plus_label
    minus_label = args.minus_label
    lyrics = str(row.get("lyrics") or "").replace("\n", " / ")
    plus_scale = max(scales)
    minus_scale = min(scales)
    plus_name = _scale_name(scales.index(plus_scale) + 1, plus_scale, plus_label, minus_label)
    minus_name = _scale_name(scales.index(minus_scale) + 1, minus_scale, plus_label, minus_label)
    lines = [
        f"# {args.name} slider — play in order",
        "",
        "Same lyrics and same seed. Slider clips use the **neutral** caption;",
        "only the LoRA scale changes. REF clips change the **prompt** with the slider off.",
        "",
        "| play | file | seconds | rms | what it should do |",
        "|-----:|------|--------:|----:|-------------------|",
    ]
    for index, scale in enumerate(scales, start=1):
        dest = _scale_name(index, scale, plus_label, minus_label)
        duration, rms = stats.get(dest, (float(args.duration), 0.0))
        if scale > 0:
            expect = f"more {plus_label} (slider +{scale:g})"
        elif scale < 0:
            expect = f"more {minus_label} (slider {scale:g})"
        else:
            expect = "neutral base (slider off)"
        if unit_scale is not None and scale != 0:
            expect = f"{expect}; LoRA ×{scale * unit_scale:g}"
        lines.append(f"| {index} | `{dest}` | {duration:.2f} | {rms:.4f} | {expect} |")
    refs = [
        (len(scales) + 1, f"{len(scales)+1:02d}_REF_prompt_{plus_label}_no_slider.wav", plus_label),
        (len(scales) + 2, f"{len(scales)+2:02d}_REF_prompt_{minus_label}_no_slider.wav", minus_label),
    ]
    for index, fname, label in refs:
        duration, rms = stats.get(fname, (float(args.duration), 0.0))
        lines.append(
            f"| {index} | `{fname}` | {duration:.2f} | {rms:.4f} | "
            f"no slider; prompt is the {label} caption |"
        )
    lines.extend(
        [
            "",
            f"- weights: `{weights}`",
            f"- lyrics: `{lyrics}`",
            f"- requested: {args.duration}s  seed: {args.seed}  rank: {rank}  alpha: {alpha}  kind: {args.kind}",
        ]
    )
    if unit_scale is not None:
        lines.append(
            f"- unit_scale: {unit_scale:g}  (user ±1 is one trained concept; "
            f"LoRA multiplier = user_scale × {unit_scale:g})"
        )
    lines.extend(
        [
            "",
            f"If it works, `{plus_name}` should lean toward the {plus_label} REF, "
            f"and `{minus_name}` toward the {minus_label} REF.",
            "",
        ]
    )
    readme = Path(args.out_dir) / "LISTEN.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {readme}", flush=True)


def generate(args: argparse.Namespace) -> list[Path]:
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    meta = _sidecar(weights)
    # Sidecar-driven defaults; CLI flags are overrides only.
    if args.kind is None:
        args.kind = "lm" if str(meta.get("kind") or "") == "language_model" else "transformer"
    if args.plus_label is None:
        args.plus_label = str(meta.get("plus_label") or "") or None
    if args.minus_label is None:
        args.minus_label = str(meta.get("minus_label") or "") or None
    if not args.plus_label or not args.minus_label:
        raise SystemExit("--plus_label/--minus_label required (sidecar has no labels)")
    rank = int(args.rank if args.rank is not None else meta.get("rank", 8))
    alpha = float(args.alpha if args.alpha is not None else meta.get("alpha", 8.0))
    unit_scale = _unit_scale(meta, raw_scales=bool(getattr(args, "raw_scales", False)))
    row = _load_prompt_row(Path(args.prompts_file), row=int(getattr(args, "row", 0) or 0))
    scales = [float(part) for part in args.scales.split(",") if part.strip()]
    requested = float(args.duration)
    jobs = _jobs(args, row)
    if unit_scale is not None:
        mapped = ", ".join(f"{scale:g}->{scale * unit_scale:g}" for scale in scales)
        print(f"unit_scale={unit_scale:g}; user->LoRA {mapped}", flush=True)
    elif getattr(args, "raw_scales", False) and meta.get("unit_scale") is not None:
        print(f"--raw_scales: ignoring sidecar unit_scale={meta['unit_scale']}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, tuple[float, float]] = {}
    pending: list[tuple[Path, str, float | None]] = []
    for dest, prompt, scale in jobs:
        ok, reason, duration, rms = _accept_wav(
            dest, requested, accept_silent=bool(getattr(args, "accept_silent", False)),
            accept_short=bool(getattr(args, "accept_short", False)))
        if ok and not args.force:
            print(f"skip existing {dest.name} duration={duration:.2f}s rms={rms:.4f}", flush=True)
            stats[dest.name] = (duration, rms)
            continue
        if dest.exists() and not args.force:
            print(f"re-render {dest.name}: {reason}", flush=True)
        pending.append((dest, prompt, scale))

    if pending:
        device = "cuda:0"
        pipe = _load_pipeline(Path(args.model_dir), device)
        kind = args.kind
        if kind == "lm":
            host = pipe.language_model
            replace = list(meta.get("target_replace") or LM_REPLACE)
        else:
            host = pipe.transformer
            replace = list(meta.get("target_replace") or TRANSFORMER_REPLACE)
        network = LoRANetwork(
            host,
            rank=rank,
            alpha=alpha,
            multiplier=1.0,
            target_replace=replace,
            train_method=str(meta.get("train_method") or "full"),
            delimiter=str(meta.get("delimiter") or "-"),
            prefix=str(meta.get("prefix") or ("lora_te" if kind == "lm" else "lora_unet")),
        )
        network.to(device)
        from safetensors.torch import load_file

        state = load_file(str(weights), device="cpu")
        missing, unexpected = network.load_state_dict(state, strict=False)
        print(
            f"loaded {weights.name} kind={kind} modules={len(network.unet_loras)} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        if len(network.unet_loras) == 0:
            raise RuntimeError(f"LoRA wrapped 0 modules for {replace}")
        if missing:
            raise RuntimeError(f"LoRA load missed {len(missing)} keys (first={missing[0]})")

        sample_rate = int(pipe.sampling_rate)
        seed = int(args.seed)
        lyrics = str(row.get("lyrics") or "")

        for dest, prompt, scale in pending:
            if scale is None:
                for lora in network.unet_loras:
                    lora.multiplier = 0.0
                label = dest.name
            else:
                multiplier = float(scale) if unit_scale is None else float(scale) * unit_scale
                network.set_lora_slider(multiplier)
                if unit_scale is None:
                    label = f"slider scale={scale:g}"
                else:
                    label = f"slider scale={scale:g} multiplier={multiplier:g}"
            ctx = network if scale is not None else _NullContext()
            retries = max(0, int(getattr(args, "retry_seeds", 2)))
            for attempt in range(retries + 1):
                # AR can early-EOS on some prompt/seed pairs; retry the clip on
                # a bumped seed rather than failing the whole listen set.
                use_seed = seed + 1000 * attempt
                generator = torch.Generator(device).manual_seed(use_seed)
                print(f"{label} seed={use_seed} prompt={prompt[:72]!r}", flush=True)
                with ctx:
                    audio = pipe(
                        prompt=prompt,
                        lyrics=lyrics,
                        audio_duration=requested,
                        generator=generator,
                        output="audios",
                    )[0]
                try:
                    duration, rms = _write_wav(
                        dest, audio, sample_rate, requested,
                        accept_silent=bool(getattr(args, "accept_silent", False)),
                        accept_short=bool(getattr(args, "accept_short", False)))
                    break
                except RuntimeError as exc:
                    if attempt >= retries:
                        raise
                    print(f"retrying {dest.name}: {exc}", flush=True)
            stats[dest.name] = (duration, rms)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_readme(args, weights, row, scales, stats, rank, alpha, unit_scale=unit_scale)
    written = [dest for dest, _prompt, _scale in jobs]
    missing = [str(path) for path in written if not path.exists()]
    if missing:
        raise RuntimeError(f"listen set incomplete: {missing}")
    return written


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    raw = _glue_negative_option(raw, "--scales")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--plus_label", default=None, help="what +scale should sound like (default: sidecar plus_label)")
    p.add_argument("--minus_label", default=None, help="what -scale should sound like (default: sidecar minus_label)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scales", default="-2,0,2")
    p.add_argument("--row", type=int, default=0, help="which prompt row to render (default 0)")
    p.add_argument("--retry_seeds", type=int, default=2, help="seed bumps when a clip renders short/silent")
    p.add_argument(
        "--accept_silent",
        action="store_true",
        help="keep a below-MIN_RMS render instead of rejecting/retrying it. The "
        "pipeline needs collapse evidence on disk (with --retry_seeds 0, so a "
        "seed bump can never silently unpair a comparison); short/unreadable "
        "clips still fail",
    )
    p.add_argument(
        "--accept_short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep a render that ends before the duration cap (the model sampled "
        "<|audio_end|> early — a natural song ending, not a broken render); "
        "first-draw seed is preserved. Disable with --no-accept_short",
    )
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument(
        "--kind",
        choices=["transformer", "lm"],
        default=None,
        help="default: from sidecar kind (language_model -> lm, else transformer)",
    )
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--force", action="store_true", help="regenerate even if valid wavs exist")
    p.add_argument(
        "--raw_scales",
        action="store_true",
        help="use LoRA multipliers as given; ignore sidecar unit_scale",
    )
    return p.parse_args(raw)


if __name__ == "__main__":
    generate(parse_args())
