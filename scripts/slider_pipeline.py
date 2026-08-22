#!/usr/bin/env python3
"""Automated slider-variant pipeline: train -> render -> score -> report.

One sweep = one FIXED caption pair + one frozen base recipe + N variants.
Every seed is pinned in the spec, so variants differ only by the variant: the
ranking is a PAIRED comparison (unpaired, seed sd ~1.0 ln-rms swamps every
recipe effect ever measured on this system; paired, a 0.25 ln effect is
sign-consistent at 3 seeds).

    PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
    $PY scripts/slider_pipeline.py selftest
    $PY scripts/slider_pipeline.py train   slider_pipeline/specs/phase1-loss-triphop.yaml --gpu 0
    $PY scripts/slider_pipeline.py render  slider_pipeline/specs/phase1-loss-triphop.yaml --gpu 0
    $PY scripts/slider_pipeline.py score   slider_pipeline/specs/phase1-loss-triphop.yaml
    $PY scripts/slider_pipeline.py report  slider_pipeline/specs/phase1-loss-triphop.yaml
    $PY scripts/slider_pipeline.py confirm slider_pipeline/specs/phase1-loss-triphop.yaml --winner NAME --gpu 0
    $PY scripts/slider_pipeline.py score-folders --root eval/listen/curve-gp-4s \
        --variants R-final,E-gp05,E-gp2 --intended centroid:+1   # legacy acceptance path

Stages are resumable: finished trainings/renders are adopted after a config
echo check (a checkpoint whose sidecar disagrees with the spec is refused, not
silently used). `score` and `report` are pure CPU and re-runnable at will.
The human sees REPORT.md: gate vectors first, ranking second, evidence linked.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slider_pipeline import features, gates, stats  # noqa: E402
from slider_pipeline.spec import AxisSpec, load_spec  # noqa: E402
from slider_pipeline.stages import (  # noqa: E402
    _folder_to_cells,
    confirm_stage,
    null_e_from_zero_pool,
    render_stage,
    report_stage,
    score_stage,
    train_stage,
    warm_cache,
)


def cmd_selftest(_args) -> int:
    rc = features.selftest()
    rc |= stats.selftest()
    rc |= gates.selftest()
    print("pipeline selftest:", "OK" if rc == 0 else "FAILED")
    return rc


def cmd_score_folders(args) -> int:
    """Score already-rendered ladder folders laid out as <root>/<variant>-seed<k>/.

    This is the acceptance path: the legacy 4 s ladders (curve-gp-4s,
    pairs-4s, seed-sweep-4s) carry human verdicts, so the gates can be checked
    against ground truth with zero GPU time.
    """
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    intended = {}
    for part in args.intended.split(","):
        feat, sign = part.split(":")
        intended[feat] = float(sign)
    axis = AxisSpec(intended=intended, level_axis=args.level_axis)
    results = {}
    all_folders = [f for v in args.variants.split(",") for f in sorted(root.glob(f"{v}-seed*"))]
    warm_cache(all_folders, args.duration)
    for variant in args.variants.split(","):
        folders = sorted(root.glob(f"{variant}-seed*"))
        if len(folders) < 3:
            print(f"{variant}: only {len(folders)} seed folders under {root} — skipped (need >=3)")
            continue
        data = {}
        for folder in folders:
            seed = int(folder.name.rsplit("seed", 1)[1])
            data[seed] = _folder_to_cells(folder, expect_duration=args.duration)
        scales = sorted(set.intersection(*(set(c) for c in data.values())))
        data = {s: {sc: cells[sc] for sc in scales} for s, cells in data.items()}
        null_e95 = args.null_e95
        if null_e95 is None and len(data) >= 4:
            # derive the seed-noise null from this variant's own cross-seed zeros
            pool = {seed: cells[0.0].feats for seed, cells in data.items()}
            null_e95 = null_e_from_zero_pool(pool, axis, sorted(data))["e95"]
            print(f"{variant}: null e95 from own zero pool = {null_e95:.4f}")
        res = gates.evaluate_variant(data, axis, null_e95=null_e95,
                                     allow_single_side=args.allow_single_side)
        results[variant] = res
        failed = [g for g, v in res["gates"].items() if not v["pass"]]
        print(f"{variant:>16s}  score={res['score']:.3f}  "
              f"{'PASS' if res['pass'] else 'VETOED ' + ','.join(failed)}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.ab_csv:
        import csv as _csv

        with open(args.ab_csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["run", "side", "value"])
            for name, r in results.items():
                for side in ("plus", "minus"):
                    w.writerow([name, side, r["sides"][side]["squashed"]])
        print(f"wrote {args.ab_csv} (for scripts/score_ab_session.py --metric)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest")
    for name in ("train", "render", "score", "report"):
        p = sub.add_parser(name)
        p.add_argument("spec", type=Path)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--only", default=None, help="comma-separated variant names")
    p = sub.add_parser("confirm")
    p.add_argument("spec", type=Path)
    p.add_argument("--winner", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p = sub.add_parser("onboard")
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--intended", required=True, help="e.g. centroid:+1")
    p.add_argument("--seeds", default="7,23,77")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--duration", type=float, default=4.0)
    p = sub.add_parser("score-folders")
    p.add_argument("--root", required=True)
    p.add_argument("--variants", required=True)
    p.add_argument("--intended", required=True, help="e.g. centroid:+1[,hi4k:+1]")
    p.add_argument("--level_axis", action="store_true")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--null_e95", type=float, default=None,
                   help="null threshold if known; omitted = G6 reports unscoreable")
    p.add_argument("--out", default=None)
    p.add_argument("--allow_single_side", action="store_true",
                   help="legacy folders only: evaluate a plus-only ladder")
    p.add_argument("--ab_csv", default=None,
                   help="also write a run,side,value metric CSV for score_ab_session.py")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return cmd_selftest(args)
    if args.cmd == "score-folders":
        return cmd_score_folders(args)
    if args.cmd == "onboard":
        intended = {}
        for part in args.intended.split(","):
            feat, sign = part.split(":")
            intended[feat] = float(sign)
        from slider_pipeline.stages import onboard_stage

        res = onboard_stage(
            prompts_file=Path(args.prompts_file), cache_dir=Path(args.cache_dir),
            out_root=Path(args.out_root), axis=AxisSpec(intended=intended),
            seeds=[int(s) for s in args.seeds.split(",")], gpu=args.gpu,
            duration=args.duration)
        return 0 if res["certified"] else 3

    spec = load_spec(args.spec)
    only = args.only.split(",") if getattr(args, "only", None) else None
    if args.cmd == "train":
        train_stage(spec, gpu=args.gpu, only=only)
    elif args.cmd == "render":
        render_stage(spec, gpu=args.gpu, only=only)
    elif args.cmd == "score":
        score_stage(spec)
    elif args.cmd == "report":
        report_stage(spec)
    elif args.cmd == "confirm":
        confirm_stage(spec, winner=args.winner, gpu=args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
