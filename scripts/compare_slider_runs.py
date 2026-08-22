#!/usr/bin/env python3
"""Compare transformer-slider training runs on the fixed eval probe.

The per-step `cos` in the train jsonl is one random (t, eps) draw out of a field
whose target norm spans ~200x across t, so it is far too noisy to rank runs by.
`train_lora_music3.py --eval_every N` writes an `eval` block instead, pinned to
a fixed t grid and a fixed noise seed; this reads those blocks.

    python scripts/compare_slider_runs.py models/*/[a-z]*_train.jsonl

Columns:
  cos       mean cos(delta(+1) - vel_neu, vel_pos - vel_neg) over the probe
  cos_neg   same for the -1 pole against the reversed axis
  collapse  cos(delta(+1), delta(-1)); -1 is a clean two-sided slider
  mag       ||delta(+1)|| / ||guidance * (vel_pos - vel_neg)||; 1.0 = full strength
  proj_abs  sum_i <delta_i, axis_hat_i> / sum_i ||guidance * axis_i||, absolute
            units. cos and mag are per-instance normalized, so both silently
            reweight the timesteps where the edit is largest; proj_abs does not,
            and it is what the Euler solve actually integrates into the render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict | None:
    evals: list[tuple[int, dict]] = []
    steps = 0
    losses: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        steps = max(steps, int(record.get("step", 0)))
        if isinstance(record.get("loss"), (int, float)):
            losses.append(float(record["loss"]))
        if isinstance(record.get("eval"), dict):
            evals.append((int(record["step"]), record["eval"]))
    if not evals:
        return None
    return {
        "path": path,
        "steps": steps,
        "evals": evals,
        "final": evals[-1][1],
        "best": max(evals, key=lambda item: item[1].get("cos", -1.0)),
        "loss_tail": sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else float("nan"),
    }


def sidecar_config(path: Path) -> str:
    meta_path = Path(str(path).replace("_train.jsonl", "_last.json"))
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    bits = [
        f"r{meta.get('rank')}",
        str(meta.get("targets") or ""),
        str(meta.get("loss_kind") or "mse"),
        str(meta.get("xt_mode") or ""),
    ]
    return "/".join(bit for bit in bits if bit and bit != "None")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="*_train.jsonl files")
    parser.add_argument("--by_t", action="store_true", help="also print the per-timestep cos breakdown")
    args = parser.parse_args(argv)

    runs = []
    for path in args.logs:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        run = load_run(path)
        if run is None:
            print(f"no eval blocks in {path.name} (trained before --eval_every, or with 0)", file=sys.stderr)
            continue
        runs.append(run)
    if not runs:
        return 1

    runs.sort(key=lambda run: run["final"].get("cos", -1.0), reverse=True)
    name_width = max(len(run["path"].parent.name) for run in runs) + 1
    print(
        f"{'run':{name_width}s} {'config':>18s} {'steps':>6s} "
        f"{'cos':>7s} {'cos_neg':>8s} {'collapse':>9s} {'mag':>6s} {'proj_abs':>9s} "
        f"{'best cos':>9s} {'@step':>6s}"
    )
    for run in runs:
        final = run["final"]
        best_step, best = run["best"]
        print(
            f"{run['path'].parent.name:{name_width}s} {sidecar_config(run['path']):>18s} "
            f"{run['steps']:6d} {final.get('cos', float('nan')):7.4f} "
            f"{final.get('cos_neg', float('nan')):8.4f} {final.get('collapse', float('nan')):9.4f} "
            f"{final.get('mag', float('nan')):6.3f} {final.get('proj_abs', float('nan')):9.4f} "
            f"{best.get('cos', float('nan')):9.4f} {best_step:6d}"
        )

    if args.by_t:
        for run in runs:
            by_t = run["final"].get("cos_by_t") or {}
            if not by_t:
                continue
            cells = "  ".join(f"t={t}:{value:+.3f}" for t, value in by_t.items())
            print(f"\n{run['path'].parent.name}\n  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
