"""Pipeline stages: train, render, score, report, confirm.

Each stage is independently runnable and resumable: finished work is detected
and adopted only after its recorded config matches the spec (the config-echo
check exists because a silent argparse default has already cost this project a
week — see MUSIC3.md's retraction header). Failures raise; nothing downgrades
to a warning unless it is explicitly a judgement call, and then it is printed.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

from .features import ln_delta, scan_folder
from .gates import IDENT_KEYS, LadderCell, VariantData, evaluate_variant, intended_proj, offaxis_steps
from .spec import REPO_ROOT, SCORED_FEATURES, SweepSpec, VariantSpec
from .stats import paired_summary, percentile

PY = "/home/mikkel/anaconda3/envs/minimax-music3/bin/python"
TRAINER = REPO_ROOT / "conceptmod/textsliders/train_lora_music3.py"
RENDERER = REPO_ROOT / "conceptmod/textsliders/generate_listen.py"

# sidecar key per trainer arg the pipeline pins or varies; used for the
# config-echo check after every adopted or fresh training run.
ECHO_KEYS = {
    "rank": "rank", "alpha": "alpha", "steps": "steps", "seed": "seed",
    "loss": "loss_kind", "gain_penalty": "gain_penalty", "gain_mode": "gain_mode",
    "gain_tweight": "gain_tweight", "mag_weight": "mag_weight", "lr": "lr",
    "targets": "targets", "xt_mode": "xt_mode", "target_mode": "target_mode",
    "duration": "duration",
}


def _env() -> dict:
    import os

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _run(cmd: list[str], log_path: Path, gpu: int) -> None:
    env = _env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n# {dt.datetime.now().isoformat()} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}\nlog: {log_path}")


def trainer_args(spec: SweepSpec, variant: VariantSpec) -> dict[str, object]:
    merged: dict[str, object] = dict(spec.base_args)
    merged.update(variant.args)
    return merged


def _cli(args: dict[str, object]) -> list[str]:
    out: list[str] = []
    for key, val in sorted(args.items()):
        if isinstance(val, bool):
            if val:
                out.append(f"--{key}")
        else:
            out.append(f"--{key}")
            out.append(str(val))
    return out


def find_weights(save_dir: Path) -> tuple[Path, dict]:
    """The run's _last checkpoint + sidecar. _best is probe-selected and the
    probe does not predict renders, so the pipeline scores _last, always."""
    sidecars = sorted(save_dir.glob("*_last.json"))
    if len(sidecars) != 1:
        raise FileNotFoundError(f"{save_dir}: expected exactly one *_last.json, found {len(sidecars)}")
    meta = json.loads(sidecars[0].read_text(encoding="utf-8"))
    weights = sidecars[0].with_suffix(".safetensors")
    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing next to its sidecar")
    return weights, meta


def check_echo(spec: SweepSpec, variant: VariantSpec, meta: dict) -> None:
    """A run is adopted only if its sidecar echoes the requested config.

    Catches the silent-argparse-default class (a run trained with targets=attn
    while the sweep says full would otherwise poison the comparison unseen).
    """
    want = trainer_args(spec, variant)
    problems = []
    for arg, meta_key in ECHO_KEYS.items():
        if arg not in want:
            continue
        got = meta.get(meta_key)
        expect = want[arg]
        if isinstance(expect, bool) or isinstance(got, bool):
            ok = bool(got) == bool(expect)
        else:
            try:
                ok = math.isclose(float(got), float(expect), rel_tol=1e-9)
            except (TypeError, ValueError):
                ok = str(got) == str(expect)
        if not ok:
            problems.append(f"{arg}: requested {expect!r}, sidecar says {got!r}")
    cs = [int(s) for s in str(want.get("cond_seeds", "")).split(",") if str(s).strip()]
    if cs and meta.get("cond_seeds") != cs:
        problems.append(f"cond_seeds: requested {cs}, sidecar says {meta.get('cond_seeds')}")
    if problems:
        raise RuntimeError(
            f"config echo mismatch for {variant.name} — refusing to adopt this checkpoint:\n  "
            + "\n  ".join(problems)
        )


# --------------------------------------------------------------- train

def train_stage(spec: SweepSpec, gpu: int = 0, only: list[str] | None = None) -> None:
    for variant in spec.variants:
        if only and variant.name not in only:
            continue
        save_dir = spec.models_root / variant.name
        if list(save_dir.glob("*_last.json")):
            _, meta = find_weights(save_dir)
            check_echo(spec, variant, meta)
            print(f"train: adopt {variant.name} (config echo OK)")
            continue
        args = trainer_args(spec, variant)
        cmd = [PY, "-u", str(TRAINER), "--name", variant.name, "--save_dir", str(save_dir),
               "--prompts_file", str(spec.prompts_file), "--cache_dir", str(spec.cache_dir),
               "--device", "0"] + _cli(args)
        print(f"train: {variant.name} on gpu{gpu}")
        _run(cmd, spec.models_root / f"{variant.name}.train.log", gpu)
        _, meta = find_weights(save_dir)
        check_echo(spec, variant, meta)
        manifest = {
            "sweep": spec.sweep, "variant": variant.name, "role": variant.role,
            "requested_args": {k: str(v) for k, v in args.items()},
            "spec": str(spec.spec_path), "finished": dt.datetime.now().isoformat(),
        }
        (save_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --------------------------------------------------------------- render

def _render_scales(spec: SweepSpec) -> list[float]:
    return sorted(set(spec.scales) | {-spec.over_scale, spec.over_scale})


def render_stage(spec: SweepSpec, gpu: int = 0, only: list[str] | None = None,
                 seeds: list[int] | None = None) -> None:
    scales = _render_scales(spec)
    scales_arg = "--scales=" + ",".join(f"{s:g}" for s in scales)
    for variant in spec.variants:
        if only and variant.name not in only:
            continue
        weights, meta = find_weights(spec.models_root / variant.name)
        check_echo(spec, variant, meta)
        for seed in seeds if seeds is not None else spec.compare_seeds:
            out_dir = spec.listen_root / f"{variant.name}-seed{seed}"
            cmd = [PY, "-u", str(RENDERER), "--weights", str(weights),
                   "--prompts_file", str(spec.prompts_file), "--name", variant.name,
                   "--kind", "transformer", "--out_dir", str(out_dir), scales_arg,
                   "--duration", f"{spec.render_duration:g}", "--seed", str(seed),
                   "--device", "0", "--retry_seeds", "0", "--raw_scales", "--accept_silent"]
            print(f"render: {variant.name} seed {seed}")
            _run(cmd, spec.listen_root / "render.log", gpu)
    # zero-only renders for the null distribution (slider off — variant-independent).
    # Rendered only by the invocation that owns the baseline, so two parallel
    # `render --only` shards cannot race on the same files.
    if only is not None and spec.baseline not in only:
        return
    base_w, _ = find_weights(spec.models_root / spec.baseline)
    for seed in spec.null_seeds:
        out_dir = spec.listen_root / f"null-seed{seed}"
        cmd = [PY, "-u", str(RENDERER), "--weights", str(base_w),
               "--prompts_file", str(spec.prompts_file), "--name", "null",
               "--kind", "transformer", "--out_dir", str(out_dir), "--scales=0",
               "--duration", f"{spec.render_duration:g}", "--seed", str(seed),
               "--device", "0", "--retry_seeds", "0", "--raw_scales", "--accept_silent"]
        print(f"render: null seed {seed}")
        _run(cmd, spec.listen_root / "render.log", gpu)


# --------------------------------------------------------------- score

def _folder_to_cells(folder: Path, expect_duration: float) -> dict[float, LadderCell]:
    folder = Path(folder)
    cache_path = folder / ".pipeline_features.json"
    fingerprint = {p.name: p.stat().st_mtime_ns for p in sorted(folder.glob("*.wav"))}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if cached and cached.get("fingerprint") == fingerprint \
                and cached.get("expect_duration") == expect_duration:
            return {float(s): LadderCell(deltas=c["deltas"], feats=c["feats"], ident=c["ident"])
                    for s, c in cached["cells"].items()}
    clips = scan_folder(folder, expect_duration=expect_duration)
    zero = clips[0.0]
    cells: dict[float, LadderCell] = {}
    for scale, clip in clips.items():
        deltas = {f: ln_delta(f, clip.feats[f], zero.feats[f]) for f in SCORED_FEATURES}
        cells[scale] = LadderCell(deltas=deltas, feats=dict(clip.feats), ident=clip.ident)
    payload = {
        "fingerprint": fingerprint, "expect_duration": expect_duration,
        "cells": {str(s): {"deltas": c.deltas, "feats": c.feats, "ident": c.ident}
                  for s, c in cells.items()},
    }
    try:
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # cache is an optimization; measurement correctness never depends on it
    return cells


def warm_cache(folders: list[Path], expect_duration: float, jobs: int | None = None) -> None:
    """Populate per-folder feature caches in parallel (folder-level Pool).

    Scanning is ~10 s/folder single-threaded and embarrassingly parallel; the
    caches are mtime-keyed, so re-scoring after a gate-threshold change is
    seconds. Correctness never depends on this — serial scans produce the
    identical caches."""
    import multiprocessing as mp

    todo = [f for f in folders if f.exists()]
    if not todo:
        return
    jobs = jobs or max(1, min(16, (mp.cpu_count() or 8) - 2))
    if jobs == 1 or len(todo) == 1:
        for f in todo:
            _folder_to_cells(f, expect_duration)
        return
    with mp.Pool(jobs) as pool:
        pool.starmap(_folder_to_cells, [(f, expect_duration) for f in todo])


def load_variant_data(spec: SweepSpec, variant_name: str,
                      seeds: list[int] | None = None) -> dict[int, dict[float, LadderCell]]:
    data: VariantData = {}
    for seed in seeds if seeds is not None else spec.compare_seeds:
        folder = spec.listen_root / f"{variant_name}-seed{seed}"
        data[seed] = _folder_to_cells(folder, spec.render_duration)
    return data


def _restrict(data: VariantData, scales: list[float]) -> VariantData:
    return {seed: {s: c for s, c in cells.items() if s in scales} for seed, cells in data.items()}


def zero_consistency(all_data: dict[str, VariantData]) -> dict:
    """The zero-scale clip is slider-off, so at a given seed it must be the
    same render for every variant. Divergence means the paired design is
    broken (nondeterminism, wrong weights at scale 0, env drift)."""
    report, worst = {}, 0.0
    seeds = set()
    for data in all_data.values():
        seeds.update(data.keys())
    for seed in sorted(seeds):
        vals = {name: d[seed][0.0].feats["rms_pc"] for name, d in all_data.items() if seed in d}
        lo, hi = min(vals.values()), max(vals.values())
        rel = (hi - lo) / max(lo, 1e-9)
        report[str(seed)] = {"rel_spread": round(rel, 5), "n_variants": len(vals)}
        worst = max(worst, rel)
    if worst > 0.05:
        raise RuntimeError(
            f"zero-scale clips diverge across variants (worst rel spread {worst:.3f}) — "
            "the paired design is broken; check GPU/env drift before believing any ranking"
        )
    return {"worst_rel_spread": round(worst, 5), "per_seed": report}


def null_e_from_zero_pool(pool: dict[int, dict[str, float]], axis, compare_seeds: list[int]) -> dict:
    """Score distribution manufactured by seed noise alone: the ranking E for
    pseudo-candidates whose '+1 clip' is a DIFFERENT seed's zero clip."""
    if len(pool) < 4:
        raise RuntimeError(
            f"null pool has only {len(pool)} zero renders; add null seeds "
            "or the 95th percentile is meaningless"
        )
    import itertools
    import random

    es = []
    others = {s: [o for o in pool if o != s] for s in compare_seeds}
    n_total = 1
    for s_ in compare_seeds:
        n_total *= len(others[s_])
    if n_total <= 512:
        combos = list(itertools.product(*(others[s_] for s_ in compare_seeds)))
    else:
        # the full product explodes with many seeds (9^10 for a 10-seed pool);
        # a fixed-seed sample keeps the p95 stable and the run finite.
        # Deterministic by construction — never time-seeded.
        rng = random.Random(1234)
        combos = [tuple(rng.choice(others[s_]) for s_ in compare_seeds) for _ in range(512)]
    for combo in combos:
        per_seed = []
        for seed, sub in zip(compare_seeds, combo):
            deltas = {f: ln_delta(f, pool[sub][f], pool[seed][f]) for f in SCORED_FEATURES}
            cell = LadderCell(deltas=deltas)
            proj = abs(intended_proj(cell, axis))
            atten = 1.0 / (1.0 + max(0.0, offaxis_steps(cell, axis) - 1.0))
            per_seed.append(proj * atten)  # per unit scale at |s|=1
        es.append(st.median(per_seed))
    return {"n": len(es), "e95": percentile(es, 0.95), "median": st.median(es), "max": max(es)}


def null_distribution(spec: SweepSpec, base_data: VariantData) -> dict:
    pool: dict[int, dict[str, float]] = {}
    for seed in spec.compare_seeds:
        pool[seed] = base_data[seed][0.0].feats
    for seed in spec.null_seeds:
        folder = spec.listen_root / f"null-seed{seed}"
        clips = scan_folder(folder, expect_duration=spec.render_duration)
        pool[seed] = clips[0.0].feats
    return null_e_from_zero_pool(pool, spec.axis, spec.compare_seeds)


def overrange_gate(data_over: VariantData, over_scales: list[float]) -> dict:
    """G7 over-range: no silence beyond the fit point (the observed
    silence-above-threshold failure), checked at +-over_scale."""
    offenders = []
    for seed, cells in data_over.items():
        for s in over_scales:
            c = cells[s]
            if math.exp(c.deltas["rms_pc"]) < 0.02 or c.feats.get("rms_pc", 1.0) < 10 ** (-60 / 20):
                offenders.append({"seed": seed, "scale": s})
    return {"pass": not offenders, "offenders": offenders, "limit": "no silence at over-range"}


def paired_vs_baseline(spec: SweepSpec, data: VariantData, base: VariantData) -> dict:
    """Per-seed paired differences on the shared seeds — the measurement that
    makes single-digit recipe effects visible under seed sd ~1.0 ln."""
    hi = max(s for s in spec.scales if s > 0)
    lo = min(s for s in spec.scales if s < 0)
    out = {}
    for label, scale in (("plus", hi), ("minus", lo)):
        for feat_label, fn in (
            ("rms_ln", lambda c: c.deltas["rms_pc"]),
            ("intended_proj", lambda c: intended_proj(c, spec.axis)),
            ("onset_corr", lambda c: c.ident["onset_corr"]),
        ):
            diffs = [fn(data[s][scale]) - fn(base[s][scale]) for s in spec.compare_seeds]
            out[f"{feat_label}_{label}"] = paired_summary(diffs)
    return out


def alpha_collapse_check(spec: SweepSpec, name: str, data: VariantData,
                         base: VariantData, alpha_ratio: float) -> dict:
    """Does alpha only rescale the multiplier axis? Compare variant@s against
    baseline@(s*alpha_ratio) on the same seed: if the trained deltas are
    proportional, matched cells are near-identical and the alpha grid is the
    multiplier axis re-measured at full training cost."""
    rows = []
    for seed in spec.compare_seeds:
        for s in spec.scales:
            if s == 0.0:
                continue
            matched = s * alpha_ratio
            if matched not in spec.scales or matched == 0.0:
                continue
            a, b = data[seed][s], base[seed][matched]
            rows.append({
                "seed": seed, "scale": s, "matched_scale": matched,
                "d_proj": round(intended_proj(a, spec.axis) - intended_proj(b, spec.axis), 4),
                "d_rms_ln": round(a.deltas["rms_pc"] - b.deltas["rms_pc"], 4),
            })
    if not rows:
        raise RuntimeError(f"alpha check for {name}: no matched scales — fix the spec grid")
    d_abs = [abs(r["d_proj"]) for r in rows]
    return {"alpha_ratio": alpha_ratio, "n_matched": len(rows),
            "median_abs_d_proj": round(st.median(d_abs), 4),
            "max_abs_d_proj": round(max(d_abs), 4), "cells": rows}


def score_stage(spec: SweepSpec) -> dict:
    all_data: dict[str, VariantData] = {}
    over = [-spec.over_scale, spec.over_scale]
    folders = [spec.listen_root / f"{v.name}-seed{s}" for v in spec.variants for s in spec.compare_seeds]
    folders += [spec.listen_root / f"null-seed{s}" for s in spec.null_seeds]
    warm_cache(folders, spec.render_duration)
    for variant in spec.variants:
        all_data[variant.name] = load_variant_data(spec, variant.name)
    consistency = zero_consistency(all_data)
    base_all = all_data[spec.baseline]
    null = null_distribution(spec, base_all)

    results: dict[str, dict] = {}
    for variant in spec.variants:
        data_all = all_data[variant.name]
        main = _restrict(data_all, spec.scales)
        res = evaluate_variant(main, spec.axis, null_e95=null["e95"])
        res["gates"]["G7_overrange"] = overrange_gate(_restrict(data_all, over), over)
        res["pass"] = all(g["pass"] for g in res["gates"].values())
        res["role"] = variant.role
        if variant.name != spec.baseline:
            res["paired_vs_baseline"] = paired_vs_baseline(
                spec, _restrict(data_all, spec.scales), _restrict(base_all, spec.scales))
        if variant.role == "alpha":
            args = trainer_args(spec, variant)
            ratio = float(args["alpha"]) / float(spec.base_args["alpha"])
            res["alpha_check"] = alpha_collapse_check(
                spec, variant.name, _restrict(data_all, spec.scales),
                _restrict(base_all, spec.scales), ratio)
        results[variant.name] = res

    # retrain-noise floor: |paired mean| of floor replicates vs baseline. Any
    # candidate whose paired effect sits inside this band is a TIE, not a win.
    floor_vals = []
    for variant in spec.variants:
        if variant.role != "floor":
            continue
        p = results[variant.name]["paired_vs_baseline"]
        floor_vals.append({k: abs(v["mean"]) for k, v in p.items()})
    floor = {}
    if floor_vals:
        for key in floor_vals[0]:
            floor[key] = round(max(fv[key] for fv in floor_vals), 4)

    out = {
        "sweep": spec.sweep, "generated": dt.datetime.now().isoformat(),
        "spec": str(spec.spec_path),
        "zero_consistency": consistency, "null": {k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in null.items()},
        "retrain_floor_abs_mean": floor, "variants": results,
    }
    out_path = spec.listen_root / "scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return out


# --------------------------------------------------------------- report

def report_stage(spec: SweepSpec) -> Path:
    scores_path = spec.listen_root / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} missing — run the score stage first")
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    results = scores["variants"]
    floor = scores.get("retrain_floor_abs_mean") or {}

    def verdict(name: str, r: dict) -> str:
        if not r["pass"]:
            failed = [g for g, v in r["gates"].items() if not v["pass"]]
            return "VETOED " + ",".join(failed)
        if name == spec.baseline:
            return "baseline"
        p = r.get("paired_vs_baseline", {})
        key = "intended_proj_plus"
        if floor and key in p and key in floor and abs(p[key]["mean"]) <= floor[key]:
            return "TIE (inside retrain floor)"
        return "RANKED"

    ranked = sorted(results.items(), key=lambda kv: (-int(kv[1]["pass"]), -kv[1]["score"]))
    lines = [
        f"# {spec.sweep} — automated comparison report",
        "",
        f"Generated {scores['generated']}. Spec: `{scores['spec']}`. "
        f"Null E95 {scores['null']['e95']:.4f} over {scores['null']['n']} seed-noise pseudo-candidates. "
        f"Zero-clip consistency worst spread {scores['zero_consistency']['worst_rel_spread']:.4f}.",
        "",
        "Gates are vetoes; the score ranks gate-passers only. Paired columns are per-seed",
        "differences against the baseline on the frozen compare seeds (mean / sd / sign-agreement);",
        f"retrain-noise floor (|mean| of floor replicates): {floor or 'not measured'}.",
        "",
        "| rank | variant | role | gates | score | Δrms_ln@+1 vs base | Δproj@+1 vs base | verdict |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for i, (name, r) in enumerate(ranked, 1):
        gate_str = " ".join(("+" if v["pass"] else "-") + g.split("_")[0][1:] for g, v in r["gates"].items())
        p = r.get("paired_vs_baseline", {})

        def cell(key: str) -> str:
            if key not in p:
                return "—"
            v = p[key]
            return f"{v['mean']:+.3f}±{v['sd']:.3f} ({v['sign_agree']:.0%})"

        lines.append(
            f"| {i} | {name} | {r['role']} | {gate_str} | {r['score']:.3f} "
            f"| {cell('rms_ln_plus')} | {cell('intended_proj_plus')} | {verdict(name, r)} |")
    lines += ["", "## Evidence per variant", ""]
    for name, r in ranked:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- ladders: " + ", ".join(
            f"[seed {s}]({name}-seed{s}/)" for s in spec.compare_seeds))
        lines.append(f"- median intended curve: {r['median_curve']}")
        lines.append(f"- sides: {r['sides']}")
        for g, v in r["gates"].items():
            mark = "PASS" if v["pass"] else "FAIL"
            detail = {k: val for k, val in v.items() if k != "pass"}
            lines.append(f"- {g}: {mark} {json.dumps(detail, default=str)[:400]}")
        if "alpha_check" in r:
            ac = dict(r["alpha_check"])
            ac.pop("cells", None)
            lines.append(f"- alpha_check: {json.dumps(ac)}")
        lines.append("")
    report = spec.listen_root / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}")

    # metric CSV for the blind AB session scorer (run,side,value)
    ab = spec.listen_root / "ab_metric.csv"
    with open(ab, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "side", "value"])
        for name, r in results.items():
            for side in ("plus", "minus"):
                w.writerow([name, side, r["sides"][side]["squashed"]])
    print(f"wrote {ab} (score with scripts/score_ab_session.py --metric {ab.name})")

    idx = [PY, str(REPO_ROOT / "scripts/build_listen_index.py"), str(spec.listen_root)]
    try:
        subprocess.run(idx, check=True, env=_env(), cwd=REPO_ROOT,
                       stdout=subprocess.DEVNULL)
        print(f"listen index built under {spec.listen_root}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"warning: listen index build failed ({exc}); report and scores are unaffected")
    return report


# --------------------------------------------------------------- onboard

def onboard_stage(prompts_file: Path, cache_dir: Path, out_root: Path, axis,
                  seeds: list[int], gpu: int = 0, us: str = "0,0.5,1",
                  duration: float = 4.0, alpha_level: float = 0.05) -> dict:
    """G0: certify a caption pair's acoustic direction BEFORE spending training
    GPU on it — from condition-interpolation ladders, not caption swaps (the
    swap's magnitude is seed noise; only cond-interp gave monotone, seed-stable
    direction evidence). Sign-flip permutation test over pooled per-seed axis
    spans; a pair with no certifiable direction is a prompt problem, and the
    pipeline refuses it here instead of five checkpoints later.
    """
    from .stats import perm_sign_flip_p

    interp = REPO_ROOT / "scripts/render_cond_interp.py"
    spans: dict[str, dict[int, float]] = {"pos": {}, "neg": {}}
    for pole in ("pos", "neg"):
        for seed in seeds:
            out_dir = out_root / f"{pole}-seed{seed}"
            cmd = [PY, "-u", str(interp), "--prompts_file", str(prompts_file),
                   "--cache_dir", str(cache_dir), "--out_dir", str(out_dir),
                   "--pole", pole, "--us", us, "--duration", f"{duration:g}",
                   "--seed", str(seed), "--device", "0"]
            need = not (out_dir / "condmix_%s_plus1.wav" % pole).exists() or \
                not (out_dir / "condmix_%s_zero.wav" % pole).exists()
            if need:
                _run(cmd, out_root / "onboard.log", gpu)
            from .features import measure_clip

            _, _, f0 = measure_clip(out_dir / f"condmix_{pole}_zero.wav")
            _, _, f1 = measure_clip(out_dir / f"condmix_{pole}_plus1.wav")
            cell = LadderCell(deltas={f: ln_delta(f, f1[f], f0[f]) for f in SCORED_FEATURES})
            spans[pole][seed] = intended_proj(cell, axis)
    # pooled spans: pos pole should project +, neg pole −; pool as one-signed
    pooled = [spans["pos"][s] for s in seeds] + [-spans["neg"][s] for s in seeds]
    p = perm_sign_flip_p(pooled)
    certified = p <= alpha_level and st.mean(pooled) > 0
    out = {
        "prompts_file": str(prompts_file), "seeds": seeds, "us": us,
        "intended": axis.intended,
        "spans_pos": {str(k): round(v, 3) for k, v in spans["pos"].items()},
        "spans_neg": {str(k): round(v, 3) for k, v in spans["neg"].items()},
        "pooled_mean": round(st.mean(pooled), 3), "perm_p": round(p, 4),
        "certified": bool(certified),
        "verdict": ("CERTIFIED" if certified else
                    "REJECTED — no certifiable direction at these poles (G0); "
                    "fix the captions, do not train"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "onboard.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"onboard: {out['verdict']} (perm p={p:.4f}, pooled mean {out['pooled_mean']})")
    return out


# --------------------------------------------------------------- confirm

def confirm_stage(spec: SweepSpec, winner: str, gpu: int = 0) -> dict:
    """Held-out confirmation of one winner: fresh frozen seeds never used for
    ranking, plus the long-duration render. If the paired advantage does not
    survive the holdout seeds, it was seed-fitting, and the report says so."""
    if not spec.holdout_seeds:
        raise RuntimeError("spec has no holdout seeds; confirmation impossible")
    spec.variant(winner)  # loud KeyError for a typo'd winner
    render_stage(spec, gpu=gpu, only=[winner, spec.baseline], seeds=spec.holdout_seeds)
    data = load_variant_data(spec, winner, seeds=spec.holdout_seeds)
    base = load_variant_data(spec, spec.baseline, seeds=spec.holdout_seeds)
    hi = max(s for s in spec.scales if s > 0)
    lo = min(s for s in spec.scales if s < 0)
    out = {"winner": winner, "holdout_seeds": spec.holdout_seeds, "paired": {}}
    for label, scale in (("plus", hi), ("minus", lo)):
        diffs = [intended_proj(data[s][scale], spec.axis) - intended_proj(base[s][scale], spec.axis)
                 for s in spec.holdout_seeds]
        rms = [data[s][scale].deltas["rms_pc"] - base[s][scale].deltas["rms_pc"]
               for s in spec.holdout_seeds]
        out["paired"][label] = {"intended_proj": paired_summary(diffs), "rms_ln": paired_summary(rms)}
    # long-duration windowed check at the extremes (front-loading is measured:
    # 4 s scores overstate level effects ~2x vs the later windows of a 20 s render)
    long_dir = spec.listen_root / f"confirm-{winner}-long"
    weights, _ = find_weights(spec.models_root / winner)
    cmd = [PY, "-u", str(RENDERER), "--weights", str(weights),
           "--prompts_file", str(spec.prompts_file), "--name", winner,
           "--kind", "transformer", "--out_dir", str(long_dir),
           f"--scales={lo:g},0,{hi:g}", "--duration", f"{spec.long_duration:g}",
           "--seed", str(spec.holdout_seeds[0]), "--device", "0",
           "--retry_seeds", "0", "--raw_scales", "--accept_silent"]
    _run(cmd, spec.listen_root / "render.log", gpu)
    clips = scan_folder(long_dir, expect_duration=spec.long_duration)
    zero = clips[0.0]
    long_report = {}
    for s, clip in clips.items():
        if s == 0.0:
            continue
        long_report[str(s)] = {
            "rms_ln": round(ln_delta("rms_pc", clip.feats["rms_pc"], zero.feats["rms_pc"]), 3),
            **{k: round(clip.ident[k], 3) for k in IDENT_KEYS},
        }
    out["long"] = long_report
    out_path = spec.listen_root / f"confirm-{winner}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return out
