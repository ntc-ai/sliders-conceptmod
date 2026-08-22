#!/usr/bin/env python3
"""Score an already-trained transformer slider on the fixed eval probe.

`train_lora_music3.py --eval_every N` writes these numbers during training, but
checkpoints trained before that existed (v3/v4) have none. This scores any
checkpoint after the fact on the identical probe, so old and new runs land on
one scale.

    export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/ml2/music/sliders-conceptmod
    python scripts/eval_slider_probe.py \
      models/triphop-tf-v3/triphop-tf-v3_alpha8.0_rank8_attn_last.safetensors \
      --prompts_file conceptmod/textsliders/data/prompts-triphop-v3-single.yaml \
      --cache_dir cache/triphop-v3

Rank / alpha / targets are read from the checkpoint's `_last.json` sidecar when
present, so normally only the weights path and its prompts/cache are needed.
Pass several checkpoints to get one table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork  # noqa: E402
from conceptmod.textsliders.train_lora_music3 import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    EVAL_TIMESTEPS,
    TARGET_REPLACE_ATTN,
    TARGET_REPLACE_FULL,
    _load_transformer,
    _set_lora_multiplier,
    build_conditions,
    build_eval_probe,
    evaluate_probe,
    load_prompts,
)


def sidecar_for(weights: Path) -> dict:
    # Only the final save writes a sidecar, so `_best` / `_step123` checkpoints
    # have to fall back to the run's `_last.json`. Without this the defaults
    # (rank 8 / targets attn) silently win and the network is built wrong.
    stem = weights.stem
    siblings = [weights.with_suffix(".json"), weights.parent / f"{stem}.json"]
    base = re.sub(r"_(best|step\d+)$", "_last", stem)
    if base != stem:
        siblings.append(weights.parent / f"{base}.json")
    for candidate in siblings:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", nargs="+", type=Path, help="*.safetensors slider checkpoints")
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--cond_seed", type=int, default=7)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--rank", type=int, default=None, help="override the sidecar rank")
    parser.add_argument("--alpha", type=float, default=None, help="override the sidecar alpha")
    parser.add_argument("--targets", choices=["attn", "full"], default=None)
    parser.add_argument("--by_t", action="store_true")
    parser.add_argument(
        "--eval_holdout",
        action="store_true",
        help="score against a clean latent generated fresh for the probe rather than "
        "the run's own cached anchors. The shared bank overstates: an identical "
        "config reads 0.8449 on it and 0.7410 held out.",
    )
    parser.add_argument(
        "--teacher_fp32",
        action="store_true",
        help="build the probe's teacher velocities in fp32. vel_pos - vel_neg is a "
        "2-5%% difference of large bf16 numbers, so the bf16 axis is only good to "
        "cos ~0.89-0.95; above cos ~0.9 a bf16 probe starts scoring fit to the "
        "rounding pattern. Costs ~2x model memory and a slower probe build, but "
        "the probe is 54 forwards, not a per-step cost.",
    )
    args = parser.parse_args(argv)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    prompts, _ = load_prompts(args.prompts_file)
    entries = build_conditions(
        prompts=prompts,
        cache_dir=args.cache_dir,
        duration=args.duration,
        seeds=[args.cond_seed],
        device=device,
        model_dir=args.model_dir,
        skip_ar=True,
        dummy=False,
    )
    if args.teacher_fp32:
        from diffusers import MiniMaxMusic3Transformer1DModel

        transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
            str(args.model_dir), subfolder="transformer",
            torch_dtype=torch.float32, local_files_only=True,
        )
        transformer.requires_grad_(False)
        transformer.eval()
        transformer = transformer.to(device)
        print("teacher: fp32", file=sys.stderr)
    else:
        transformer = _load_transformer(args.model_dir, device)
    amp = device.type == "cuda"

    x0_bank = None
    x0_paths = sorted(args.cache_dir.glob("x0_*.pt"))
    if x0_paths:
        import hashlib

        neutral = entries[0][1]["neutral"]
        digest = hashlib.sha256(
            neutral.to(torch.float32).cpu().contiguous().numpy().tobytes()
        ).hexdigest()[:16]
        matched = [p for p in x0_paths if digest in p.name]
        if matched:
            x0_bank = [
                torch.load(p, map_location="cpu", weights_only=False)["latents"] for p in matched
            ]
    if args.eval_holdout:
        from conceptmod.textsliders.train_lora_music3 import build_x0_banks

        print("generating held-out eval anchor", file=sys.stderr)
        x0_bank = build_x0_banks(
            entries[:1], transformer, cache_dir=args.cache_dir, model_dir=args.model_dir,
            device=device, seed=args.eval_seed, per_row=1, num_steps=30, dummy=False,
        )[0]
    elif x0_bank is None:
        print(
            "warning: no x0 anchor matching this row's neutral condition; probe falls "
            "back to pure-noise x_t, which is off-manifold for anchor-trained sliders",
            file=sys.stderr,
        )

    rows = []
    probe = None
    probe_key = None
    for path in args.weights:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        meta = sidecar_for(path)
        rank = int(args.rank if args.rank is not None else meta.get("rank", 8))
        alpha = float(args.alpha if args.alpha is not None else meta.get("alpha", rank))
        targets = args.targets or str(meta.get("targets") or "attn")
        network = LoRANetwork(
            transformer,
            rank=rank,
            alpha=alpha,
            multiplier=1.0,
            target_replace=(TARGET_REPLACE_FULL if targets == "full" else TARGET_REPLACE_ATTN),
            train_method="full",
            delimiter="-",
        ).to(device)
        _set_lora_multiplier(network, 0.0)
        # The teacher axis vel_pos - vel_neg is a few-percent difference of large
        # bf16 numbers, so it is not robust to changes in the compute path: an
        # attached LoRA wrapper (an extra `+ zeros` that shifts downstream kernel
        # selection) rotates it by ~10%, and so does batching pos/neg together.
        # Build the probe with the wrapper already attached, exactly as the
        # trainer does, or these numbers will not match the in-training ones.
        if probe is not None and probe_key != targets and args.teacher_fp32:
            raise SystemExit(
                "--teacher_fp32 builds the probe once and then drops the model to "
                "bf16 for the student, so it cannot rebuild teachers for a second "
                f"--targets kind (got {probe_key!r} then {targets!r}). Score attn "
                "and full checkpoints in separate invocations."
            )
        if probe is None or probe_key != targets:
            probe = build_eval_probe(
                transformer, entries[0], x0_bank, device=device, seed=args.eval_seed,
                amp=(amp and not args.teacher_fp32),
            )
            if args.teacher_fp32:
                # teachers are cached in the probe; the student runs bf16 like inference
                transformer.to(torch.bfloat16)
                probe.latents[:] = [x.to(torch.bfloat16) for x in probe.latents]
                probe.timesteps[:] = [t.to(torch.bfloat16) for t in probe.timesteps]
                probe.cond_target = probe.cond_target.to(torch.bfloat16)
            probe_key = targets
            print(f"probe: {len(probe.latents)} instances over t={list(EVAL_TIMESTEPS)} "
                  f"(teachers under targets={targets})\n")
        incompatible = network.load_state_dict(load_file(str(path)), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"{path.name}: {len(incompatible.missing_keys)} missing / "
                f"{len(incompatible.unexpected_keys)} unexpected keys — check --rank/--targets",
                file=sys.stderr,
            )
        result = evaluate_probe(transformer, network, probe, amp)
        rows.append((path, f"r{rank}/a{alpha:g}/{targets}", result))
        del network
        torch.cuda.empty_cache()

    if not rows:
        return 1
    rows.sort(key=lambda row: row[2]["cos"], reverse=True)
    width = max(len(path.parent.name) for path, _, _ in rows) + 1
    print(
        f"{'checkpoint':{width}s} {'config':>16s} {'cos':>7s} {'cos_neg':>8s} "
        f"{'collapse':>9s} {'mag':>6s} {'proj_abs':>9s}"
    )
    for path, config, result in rows:
        print(
            f"{path.parent.name:{width}s} {config:>16s} {result['cos']:7.4f} "
            f"{result['cos_neg']:8.4f} {result['collapse']:9.4f} {result['mag']:6.3f} "
            f"{result['proj_abs']:9.4f}"
        )
    if args.by_t:
        for path, _, result in rows:
            cells = "  ".join(f"t={t}:{v:+.3f}" for t, v in (result.get("cos_by_t") or {}).items())
            print(f"\n{path.parent.name}\n  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
