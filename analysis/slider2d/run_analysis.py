#!/usr/bin/env python3
"""CPU 2-D slider geometry: train live losses, write plots + report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.field import Field2D
from analysis.slider2d.train import MethodResult, all_specs, run_method


DEFAULT_OUT = _REPO / "docs" / "2d-analysis"


def run_all(steps: int = 250, seed: int = 0) -> list[MethodResult]:
    field = Field2D()
    return [run_method(spec, field, steps=steps, seed=seed) for spec in all_specs()]


def _metrics_blob(results: list[MethodResult], steps: int, seed: int) -> dict:
    return {
        "steps": steps,
        "seed": seed,
        "axes": {"slider": "energetic (+) ↔ calm (−)", "attribute": "male (+) ↔ female (−)"},
        "skipped": [
            "pixels / image sliders (trainscripts/imagesliders)",
            "CLAP / render gates / slider_pipeline scoring",
            "Music 3 GPU train, Hub weights, AR endreg/planreg",
            "uncond_weight, traj_frac, gain_penalty (CPU-expressible but need a zeros-condition / t-compounding teacher)",
        ],
        "methods": {
            r.name: {
                "family": r.family,
                "verdict": r.verdict,
                "reason": r.reason,
                **{k: v for k, v in r.metrics.items() if k != "action"},
            }
            for r in results
        },
    }


def plot_quiver(results: list[MethodResult], field: Field2D, path: Path) -> None:
    names = [r.name for r in results]
    cols = 4
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows), sharex=True, sharey=True)
    axes = axes.ravel()
    xs = torch.linspace(-1.6, 1.6, 7)
    ys = torch.linspace(-1.6, 1.6, 7)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    # Teacher field for ungated "song" at t=0.5
    u = torch.zeros_like(gx)
    v = torch.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            vel = field.velocity(torch.tensor([gx[i, j].item(), gy[i, j].item()]), "song", 0.5)
            u[i, j] = vel[0]
            v[i, j] = vel[1]
    for ax, result in zip(axes, results):
        ax.quiver(gx.numpy(), gy.numpy(), u.numpy(), v.numpy(), color="#bbbbbb", angles="xy", scale_units="xy", scale=8)
        dp = result.residual.delta(1.0).detach()
        dm = result.residual.delta(-1.0).detach()
        ax.arrow(0, 0, float(dp[0]), float(dp[1]), color="#c0392b", width=0.03, length_includes_head=True, head_width=0.12)
        ax.arrow(0, 0, float(dm[0]), float(dm[1]), color="#2471a3", width=0.03, length_includes_head=True, head_width=0.12)
        ax.axhline(0, color="#dddddd", lw=0.6)
        ax.axvline(0, color="#dddddd", lw=0.6)
        color = "#1e8449" if result.verdict == "right" else "#922b21"
        ax.set_title(f"{result.name}\n{result.verdict}", color=color, fontsize=9)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.suptitle("Teacher field (grey) + learned Δ(+1 red, −1 blue)  ·  x=slider  y=attribute", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trajectories(results: list[MethodResult], field: Field2D, path: Path) -> None:
    """Attribute vs slider: start at male/female/ungated song, take 8 Euler steps."""
    starts = [("song", "k"), ("male song", "#1a5276"), ("female song", "#6c3483")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    # left: methods without attrs; right: with attrs / symmetric
    left = [r for r in results if r.name in ("sd_enhance", "m3_nmse_axis", "m3_nmse_pole", "lm_raw", "enc_mse")]
    right = [r for r in results if r.name in ("sd_enhance_attrs", "m3_nmse_axis_attrs", "lm_symmetric", "lm_raw_attrs", "enc_mse_attrs")]
    for ax, group, title in (
        (axes[0], left, "without attributes / raw targets"),
        (axes[1], right, "with attributes / symmetric"),
    ):
        for result in group:
            for pname, color in starts:
                x = field.embed(pname, 0.5).clone()
                xs, ys = [float(x[0])], [float(x[1])]
                for _ in range(8):
                    x = x + 0.15 * result.residual.delta(1.0)
                    xs.append(float(x[0]))
                    ys.append(float(x[1]))
                ax.plot(xs, ys, "-o", color=color, ms=3, lw=1.2, alpha=0.85)
            # one labelled +1 / −1 from origin
            dp = result.residual.delta(1.0)
            ax.annotate(result.name, (float(dp[0]) * 0.3, float(dp[1]) * 0.3 + 0.05), fontsize=7)
        ax.axhline(0, color="#cccccc", lw=0.6)
        ax.axvline(0, color="#cccccc", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("slider  (calm ← → energetic)")
        ax.set_ylabel("attribute  (female ← → male)")
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
    fig.suptitle("Euler walk of +1 residual from song / male song / female song")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_scatter(results: list[MethodResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        leak = abs(result.metrics["leak_ratio"])
        slider = result.metrics["cos_slider_plus"] * result.metrics["want_sign"]
        color = "#1e8449" if result.verdict == "right" else "#c0392b"
        ax.scatter(leak, slider, c=color, s=50)
        ax.annotate(result.name, (leak + 0.01, slider), fontsize=7)
    ax.axvline(0.20, color="#888888", ls="--", lw=0.8)
    ax.axhline(0.90, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("|attr| / |slider| leak ratio")
    ax.set_ylabel("signed slider cosine (erase flipped)")
    ax.set_title("Leak vs slider alignment  (green=right, red=needs help)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(results: list[MethodResult], blob: dict, path: Path) -> None:
    lines = [
        "# 2-D CPU analysis of slider methods",
        "",
        "Synthetic orthonormal field. Slider axis is **energetic ↔ calm**;",
        "attribute axis is **male ↔ female**. Ungated `energetic` leaks male",
        "and both poles sit above a quiet `song` (even / common mode).",
        "No Hub, no GPU, no Music 3 weights.",
        "",
        "## Verdict",
        "",
        "| method | verdict | slider cos | leak | ±1 cos | why |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| `{r.name}` | **{r.verdict}** | {m['cos_slider_plus']:.3f} | "
            f"{m['leak_ratio']:.3f} | {m['cos_plus_minus']:.3f} | {r.reason} |"
        )
    lines += [
        "",
        "![Learned residuals on the teacher field](2d-analysis/quiver.png)",
        "",
        "![Attribute-vs-slider trajectories](2d-analysis/trajectories.png)",
        "",
        "![Leak vs slider cosine](2d-analysis/scatter.png)",
        "",
        "## What this field can see",
        "",
        "- SD enhance/erase and Music 3 TF (`nmse` / `mse` / `pole` / `axis`)",
        "  use the same `neu ± g·(pos − neg)` (or pole) target. On this field",
        "  they agree: the slider axis is right; **without `--attributes` the",
        "  attribute axis leaks**.",
        "- `--attributes male,female` pins gender on every caption. The",
        "  pos−neg difference becomes parallel to the slider — paper",
        "  disentangle is geometrically right here.",
        "- Music 3 `pole` vs `axis`: neu is off the pos−neg chord, so the",
        "  even (common) part differs. An odd LoRA multiplier cannot represent",
        "  that even part. `nmse` + pole also reweights the two edges by",
        "  `1/||edge||²`, so the odd fit is not exactly `(pos-neg)/2`.",
        "- LM `--symmetric` (default) keeps ±1 opposite. The *axis* is still",
        "  whatever pos−neg is, so ungated leak remains. Attribute-prefixed",
        "  raw targets also stay antipodal (pinned gender cancels in pos−neg).",
        "  Raw LM/encoder targets without attributes learn the shared even",
        "  mode → collapse. That matches gender-lm v2 → v3 in MUSIC3.md.",
        "",
        "## What this field cannot see",
        "",
        "- Real multi-row Music 3 captions whose slider axes are *not*",
        "  parallel (MUSIC3.md: attributes averaging destroys TF style",
        "  sliders). Additive prefixes here always cancel.",
        "- Pixels, CLAP, render gates, AR endreg/planreg, `uncond_weight`,",
        "  trajectory training, gain compounding across a 50-step solve.",
        "- `train_lora.py` (SD1) is stale against `prompt_util.py` (wrong",
        "  `PromptEmbedsPair` arity and `unconditional_latents` kwarg).",
        "  Scored is the XL/SD3 formula and the SD1 *intent*.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_analysis.py --out docs/2d-analysis",
        "PYTHONPATH=. pytest tests/test_2d_slider_geometry.py -q",
        "```",
        "",
        "Shared loss extract: `conceptmod/textsliders/slider_targets.py`.",
        "`train_lora_music3.py` imports `music3_slider_loss` from there.",
        "The GPU trainers were not rewritten beyond that import.",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps, CPU.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = run_all(steps=args.steps, seed=args.seed)
    blob = _metrics_blob(results, args.steps, args.seed)
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    field = Field2D()
    plot_quiver(results, field, out / "quiver.png")
    plot_trajectories(results, field, out / "trajectories.png")
    plot_scatter(results, out / "scatter.png")
    write_report(results, blob, out.parent / "2d-analysis.md")
    for r in results:
        print(f"{r.name:22s} {r.verdict:12s} slider={r.metrics['cos_slider_plus']:+.3f} "
              f"leak={r.metrics['leak_ratio']:+.3f} ±1={r.metrics['cos_plus_minus']:+.3f}  {r.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
