#!/usr/bin/env python3
"""Measure catalog caption entanglement and default-TF leak on energy × tempo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.tf_leak.captions import catalog_geometry, existing_render_numbers
from analysis.tf_leak.field_music import (
    MusicField2D,
    pair_from_catalog,
    score_residual,
    teacher_leak,
    train_music3,
)


DEFAULT_OUT = _REPO / "docs" / "tf-leak"


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def run_field_suite(steps: int = 150, seed: int = 0) -> dict:
    field = MusicField2D()
    specs = [
        ("energy_nmse_axis", "energy", {"kind": "nmse", "target_mode": "axis"}),
        ("energy_nmse_pole", "energy", {"kind": "nmse", "target_mode": "pole"}),
        ("energy_nmse_ortho", "energy", {"kind": "nmse_ortho", "target_mode": "axis"}),
        ("energy_nmse_gain", "energy", {"kind": "nmse", "target_mode": "axis", "gain_weight": 1.0}),
        ("energy_nmse_attrs", "energy", {"kind": "nmse", "target_mode": "axis", "attributes": True}),
        ("cand_energy_nmse_axis", "cand_energy", {"kind": "nmse", "target_mode": "axis"}),
        ("tempo_nmse_axis", "tempo", {"kind": "nmse", "target_mode": "axis"}),
        ("dust_nmse_axis", "dust", {"kind": "nmse", "target_mode": "axis"}),
        ("distortion_nmse_axis", "distortion", {"kind": "nmse", "target_mode": "axis"}),
    ]
    out = {}
    for name, catalog, kwargs in specs:
        attrs = bool(kwargs.pop("attributes", False))
        pair = pair_from_catalog(catalog, attributes=attrs)
        residual = train_music3(field, pair, steps=steps, seed=seed, **kwargs)
        out[name] = {
            "catalog": catalog,
            "teacher": teacher_leak(pair),
            "fit": score_residual(residual),
            **kwargs,
            "attributes": attrs,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "catalog": catalog_geometry(),
        "renders": existing_render_numbers(),
        "field": run_field_suite(steps=args.steps, seed=args.seed),
    }
    (args.out / "metrics.json").write_text(
        json.dumps(_round(blob), indent=2) + "\n", encoding="utf-8"
    )
    geo = blob["catalog"]
    print("BPM deltas:")
    for name, axis in geo["axes"].items():
        print(f"  {name:12s} ΔBPM={axis['bpm_delta']}  attrs={axis['has_attributes']}")
    print("bow cos (energy vs others):")
    for key, pair in geo["pairwise"].items():
        if key.startswith("energy__"):
            print(f"  {key:28s} cos={pair['bow_cos']:+.3f}  shared={pair['shared_any']}")
    print("field leak |tempo|/|energy|:")
    for name, row in blob["field"].items():
        fit = row["fit"]
        print(
            f"  {name:24s} energy={fit['cos_energy_plus']:+.3f} "
            f"tempo={fit['cos_tempo_plus']:+.3f} leak={fit['leak_ratio']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
