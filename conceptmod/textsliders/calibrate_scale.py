"""Calibrate transformer LoRA scale so user ±1 is one trained concept.

The trainer targets vel_neu + guidance * (vel_pos - vel_neg) at multiplier=1.0.
Raw LoRA units are alpha/rank, so we find s* where ||vel(s*)-vel(0)|| on the
neutral condition matches ||guidance*(vel_pos-vel_neg)||.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from conceptmod.textsliders.lora import LoRANetwork

DEFAULT_MODEL_DIR = Path("/ml2/music/models/MiniMax-Music3")
DEFAULT_SCALES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DEFAULT_TIMESTEPS = (0.25, 0.5, 0.75)
DEFAULT_REPLACE = ["MiniMaxMusic3Attention"]
LATENT_CHANNELS = 128
INFERENCE_CFG = 1.7  # ClassifierFreeGuidance in MiniMaxMusic3ChunkDenoiseInner


def sidecar_path(weights: Path) -> Path:
    for candidate in (weights.with_suffix(".json"), Path(str(weights) + ".json")):
        if candidate.exists():
            return candidate
    return weights.with_suffix(".json")


def read_sidecar(weights: Path) -> dict[str, Any]:
    path = sidecar_path(weights)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_sidecar(weights: Path, meta: dict[str, Any]) -> Path:
    path = sidecar_path(weights)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def guess_cache_dir(weights: Path, meta: dict[str, Any] | None = None) -> Path | None:
    repo = Path(__file__).resolve().parents[2]
    meta = meta or {}
    name = str(meta.get("name") or "")
    parent = weights.resolve().parent.name
    candidates: list[Path] = []
    # Prefer the weights folder (energy-slider-v2 -> cache/energy-v2) over sidecar name.
    if parent.endswith("-slider-v2"):
        candidates.append(repo / "cache" / f"{parent[: -len('-slider-v2')]}-v2")
    elif parent.endswith("-slider"):
        candidates.append(repo / "cache" / parent[: -len("-slider")])
    if name:
        candidates.extend((repo / "cache" / f"{name}-v2", repo / "cache" / name))
    candidates.append(repo / "cache" / parent)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        if any(candidate.glob("*.pt")):
            return candidate
    return None


def _align(conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lengths = [int(tensor.shape[1]) for tensor in conditions.values()]
    min_len = min(lengths)
    return {name: tensor[:, :min_len].contiguous() for name, tensor in conditions.items()}


def _norm_prompt(text: str) -> str:
    return " ".join(str(text).split())


def _load_cache_entries(cache_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            if "condition" not in payload:
                # x0_*.pt clean-latent anchors share the cache dir with the
                # condition entries; they carry {"latents", "seed"}, not a
                # condition. Skip instead of KeyError-ing the whole load.
                continue
            condition = payload["condition"]
            prompt = str(payload.get("prompt") or "")
            lm_scale = payload.get("lm_scale")
        else:
            condition = payload
            prompt = path.stem
            lm_scale = None
        entries.append(
            {
                "path": path,
                "condition": condition,
                "prompt": prompt,
                "prompt_key": _norm_prompt(prompt),
                "lm_scale": None if lm_scale is None else float(lm_scale),
            }
        )
    if not entries:
        raise FileNotFoundError(f"no condition cache tensors in {cache_dir}")
    return entries


def _pick_entry(entries: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    key = _norm_prompt(text)
    if not key:
        return None
    for entry in entries:
        if entry["prompt_key"] == key:
            return entry
    for entry in entries:
        if key in entry["prompt_key"] or entry["prompt_key"] in key:
            return entry
    return None


def _keyword_role(prompt: str) -> str | None:
    low = prompt.lower()
    if any(token in low for token in ("pop-punk", "soprano", "woman singing", "adult woman", "distorted", "shouted")):
        return "pos"
    if any(token in low for token in ("lullaby", "baritone", "adult man", "man singing", "whispered", "acoustic")):
        return "neg"
    return None


def _conditions_from_lm_scales(entries: list[dict[str, Any]]) -> dict[str, torch.Tensor] | None:
    scales = [entry["lm_scale"] for entry in entries if entry["lm_scale"] is not None]
    if len(scales) < 3:
        return None
    by_scale = {float(entry["lm_scale"]): entry["condition"] for entry in entries if entry["lm_scale"] is not None}
    pos = next((cond for scale, cond in by_scale.items() if scale > 0), None)
    neg = next((cond for scale, cond in by_scale.items() if scale < 0), None)
    neu = next((cond for scale, cond in by_scale.items() if scale == 0.0), None)
    if pos is None or neg is None or neu is None:
        return None
    return {"positive": pos, "negative": neg, "neutral": neu}


def load_conditions(cache_dir: Path, meta: dict[str, Any] | None = None) -> tuple[dict[str, torch.Tensor], float]:
    meta = meta or {}
    entries = _load_cache_entries(cache_dir)
    prompts = meta.get("prompts") or []
    row = prompts[0] if prompts else {}
    guidance = float(row.get("guidance_scale", meta.get("guidance", 1.0)) or 1.0)

    if row:
        branch: dict[str, torch.Tensor] = {}
        missing = False
        for key in ("positive", "negative", "neutral"):
            text = str(row.get(key) or row.get("target") or "")
            entry = _pick_entry(entries, text)
            if entry is None:
                missing = True
                break
            branch[key] = entry["condition"]
        # LM-slider caches share the neutral caption and differ by lm_scale.
        unique_prompts = {entry["prompt_key"] for entry in entries}
        if missing or (len(unique_prompts) == 1 and len(entries) >= 3):
            lm_branch = _conditions_from_lm_scales(entries)
            if lm_branch is not None:
                return _align(lm_branch), guidance
        if not missing:
            return _align(branch), guidance

    lm_branch = _conditions_from_lm_scales(entries)
    if lm_branch is not None:
        return _align(lm_branch), guidance

    assigned: dict[str, torch.Tensor] = {}
    leftover: list[torch.Tensor] = []
    for entry in entries:
        role = _keyword_role(entry["prompt"])
        if role == "pos" and "positive" not in assigned:
            assigned["positive"] = entry["condition"]
        elif role == "neg" and "negative" not in assigned:
            assigned["negative"] = entry["condition"]
        else:
            leftover.append(entry["condition"])
    if "neutral" not in assigned and leftover:
        assigned["neutral"] = leftover[0]
    if set(assigned) != {"positive", "negative", "neutral"}:
        raise RuntimeError(f"could not resolve pos/neg/neu conditions in {cache_dir}")
    return _align(assigned), guidance


def _l2(delta: torch.Tensor) -> float:
    flat = delta.reshape(delta.shape[0], -1).float()
    return float(flat.norm(dim=1).mean().item())


def _invert_scale(scales: list[float], deltas: list[float], target: float) -> float:
    if target <= 0:
        return 0.0
    xs = [0.0]
    ys = [0.0]
    for scale, delta in zip(scales, deltas):
        if scale <= 0:
            continue
        xs.append(float(scale))
        ys.append(float(delta))
    if len(xs) < 2:
        return 0.0
    for i in range(1, len(xs)):
        if ys[i] >= target:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if y1 <= y0:
                return x1
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    x0, x1 = xs[-2], xs[-1]
    y0, y1 = ys[-2], ys[-1]
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
    if slope <= 1e-12:
        if y1 <= 1e-12:
            return float("inf")
        return x1 * target / y1
    return x1 + (target - y1) / slope


def _round(value: float) -> float:
    if not math.isfinite(value):
        return float(value)
    return float(f"{float(value):.8g}")


def _velocity(
    transformer: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    return transformer(latents, timestep, cond, return_dict=False)[0]


def calibrate_network(
    transformer: torch.nn.Module,
    network: LoRANetwork,
    conditions: dict[str, torch.Tensor],
    guidance: float,
    device: torch.device,
    scales: list[float] | None = None,
    seed: int = 0,
    timesteps: tuple[float, ...] | list[float] = DEFAULT_TIMESTEPS,
    n_latents: int = 2,
) -> dict[str, Any]:
    scales = [float(scale) for scale in (scales or DEFAULT_SCALES) if float(scale) > 0]
    if not scales:
        raise ValueError("need at least one positive LoRA scale to sweep")
    conds = _align({name: conditions[name] for name in ("positive", "negative", "neutral")})
    amp_dtype = next(transformer.parameters()).dtype
    length = int(conds["neutral"].shape[1])
    batch = 1
    cond_pos = conds["positive"].to(device=device, dtype=amp_dtype)
    cond_neg = conds["negative"].to(device=device, dtype=amp_dtype)
    cond_neu = conds["neutral"].to(device=device, dtype=amp_dtype)
    if cond_pos.shape[0] != batch:
        cond_pos = cond_pos[:batch]
        cond_neg = cond_neg[:batch]
        cond_neu = cond_neu[:batch]

    transformer.eval()
    network.eval()
    for lora in network.unet_loras:
        lora.multiplier = 0.0

    acc_posneg = 0.0
    acc_target = 0.0
    acc_neu = 0.0
    acc_delta = {scale: 0.0 for scale in scales}
    acc_proj = {scale: 0.0 for scale in scales}
    acc_cos = {scale: 0.0 for scale in scales}
    acc_delta_cfg = {scale: 0.0 for scale in scales}
    n_samples = 0

    with torch.no_grad():
        for latent_i in range(max(1, int(n_latents))):
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + latent_i)
            latents_cpu = torch.randn(
                batch, LATENT_CHANNELS, length, generator=generator, dtype=torch.float32
            )
            latents = latents_cpu.to(device=device, dtype=amp_dtype)
            cond_zero = torch.zeros_like(cond_neu)
            for t_val in timesteps:
                timestep = torch.full((batch,), float(t_val), device=device, dtype=amp_dtype)
                vel_pos = _velocity(transformer, latents, timestep, cond_pos).float()
                vel_neg = _velocity(transformer, latents, timestep, cond_neg).float()
                vel_0 = _velocity(transformer, latents, timestep, cond_neu).float()
                # Inference applies the LoRA on both CFG branches with a zeros
                # uncond condition; measure that branch too (report-only).
                vel_u0 = _velocity(transformer, latents, timestep, cond_zero).float()
                delta_pn = vel_pos - vel_neg
                dir_hat = delta_pn.flatten(1) / delta_pn.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
                acc_posneg += _l2(delta_pn)
                acc_target += _l2(float(guidance) * delta_pn)
                acc_neu += _l2(vel_0)
                n_samples += 1
                for scale in scales:
                    network.set_lora_slider(float(scale))
                    with network:
                        vel_s = _velocity(transformer, latents, timestep, cond_neu).float()
                        vel_us = _velocity(transformer, latents, timestep, cond_zero).float()
                    delta_c = vel_s - vel_0
                    delta_u = vel_us - vel_u0
                    acc_delta[scale] += _l2(delta_c)
                    flat = delta_c.flatten(1)
                    acc_proj[scale] += float((flat * dir_hat).sum(dim=1).mean().item())
                    acc_cos[scale] += float(
                        torch.nn.functional.cosine_similarity(flat, delta_pn.flatten(1), dim=1)
                        .mean()
                        .item()
                    )
                    acc_delta_cfg[scale] += _l2(delta_u + INFERENCE_CFG * (delta_c - delta_u))

    denom = max(n_samples, 1)
    vel_pos_neg = acc_posneg / denom
    vel_target = acc_target / denom
    vel_neu = acc_neu / denom
    sweep = []
    deltas: list[float] = []
    projs: list[float] = []
    for scale in scales:
        delta = acc_delta[scale] / denom
        proj = acc_proj[scale] / denom
        deltas.append(delta)
        projs.append(proj)
        sweep.append(
            {
                "scale": _round(scale),
                "delta": _round(delta),
                "proj": _round(proj),
                "cos_axis": _round(acc_cos[scale] / denom),
                "delta_cfg": _round(acc_delta_cfg[scale] / denom),
                "rel": _round(delta / (vel_neu + 1e-8)),
                "vs_target": _round(delta / (vel_target + 1e-8)),
                "vs_posneg": _round(delta / (vel_pos_neg + 1e-8)),
            }
        )

    unit_scale = _invert_scale(scales, deltas, vel_target)
    unit_scale_g1 = _invert_scale(scales, deltas, vel_pos_neg)
    # Honest unit: only the component along (vel_pos - vel_neg) counts; the
    # magnitude-matched unit_scale overstates strength by ~1/cos_axis.
    unit_scale_projected = _invert_scale(scales, projs, vel_target)
    small = [acc_cos[scale] / denom for scale in scales if scale <= 2.0]
    cos_axis_mean = sum(small) / max(len(small), 1)
    axis_tracking_low = cos_axis_mean < 0.2
    if axis_tracking_low:
        print(
            f"WARNING: cos_axis mean {cos_axis_mean:.3f} < 0.2 — the LoRA delta barely "
            "tracks the concept direction; unit_scale is not meaningful for this slider",
            flush=True,
        )
    verify_delta = None
    if math.isfinite(unit_scale) and unit_scale > 0:
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            latents = torch.randn(
                batch, LATENT_CHANNELS, length, generator=generator, dtype=torch.float32
            ).to(device=device, dtype=amp_dtype)
            timestep = torch.full((batch,), 0.5, device=device, dtype=amp_dtype)
            vel_0 = _velocity(transformer, latents, timestep, cond_neu).float()
            network.set_lora_slider(float(unit_scale))
            with network:
                vel_s = _velocity(transformer, latents, timestep, cond_neu).float()
            verify_delta = _l2(vel_s - vel_0)
        for lora in network.unet_loras:
            lora.multiplier = 0.0

    result = {
        "unit_scale": _round(unit_scale),
        "unit_scale_projected": _round(unit_scale_projected),
        "cos_axis_mean": _round(cos_axis_mean),
        "axis_tracking_low": axis_tracking_low,
        "unit_scale_guidance": _round(float(guidance)),
        "unit_scale_g1": _round(unit_scale_g1),
        "inference_cfg": INFERENCE_CFG,
        "vel_pos_neg": _round(vel_pos_neg),
        "vel_target": _round(vel_target),
        "vel_neu": _round(vel_neu),
        "guidance": _round(float(guidance)),
        "calibrated": True,
        "calibrate_seed": int(seed),
        "calibrate_timesteps": [float(t) for t in timesteps],
        "calibrate_n_latents": int(n_latents),
        "sweep": sweep,
    }
    if verify_delta is not None:
        result["verify_delta"] = _round(verify_delta)
        result["verify_vs_target"] = _round(verify_delta / (vel_target + 1e-8))
    return result


def print_table(result: dict[str, Any]) -> None:
    print(
        f"{'scale':>7} {'||dv||':>10} {'rel':>8} {'/target':>8} {'/posneg':>8}",
        flush=True,
    )
    for row in result.get("sweep") or []:
        print(
            f"{row['scale']:7g} {row['delta']:10.3f} {row['rel']:8.4f} "
            f"{row['vs_target']:8.3f} {row['vs_posneg']:8.3f}",
            flush=True,
        )
    print(
        f"||vel_pos-vel_neg||              = {result['vel_pos_neg']:.4f}",
        flush=True,
    )
    print(
        f"||guidance*(vel_pos-vel_neg)||   = {result['vel_target']:.4f}  "
        f"(guidance={result['unit_scale_guidance']:g})",
        flush=True,
    )
    print(f"||vel_neu||                      = {result['vel_neu']:.4f}", flush=True)
    print(
        f"unit_scale      (match target)   = {result['unit_scale']:g}",
        flush=True,
    )
    if "unit_scale_projected" in result:
        print(
            f"unit_scale_proj (axis component) = {result['unit_scale_projected']:g}",
            flush=True,
        )
    print(
        f"unit_scale_g1   (match pos-neg)  = {result['unit_scale_g1']:g}",
        flush=True,
    )
    if "verify_delta" in result:
        print(
            f"verify unit_scale ||dv||={result['verify_delta']:.4f}  "
            f"vs_target={result['verify_vs_target']:.3f}",
            flush=True,
        )


def merge_calibration(meta: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    merged = dict(meta)
    for key in (
        "unit_scale",
        "unit_scale_projected",
        "cos_axis_mean",
        "axis_tracking_low",
        "unit_scale_guidance",
        "unit_scale_g1",
        "inference_cfg",
        "vel_pos_neg",
        "vel_target",
        "vel_neu",
        "guidance",
        "calibrated",
        "calibrate_seed",
        "calibrate_timesteps",
        "calibrate_n_latents",
        "sweep",
        "verify_delta",
        "verify_vs_target",
    ):
        if key in result:
            merged[key] = result[key]
    return merged


def attach_lora(
    transformer: torch.nn.Module,
    weights: Path,
    meta: dict[str, Any],
    device: torch.device,
) -> LoRANetwork:
    rank = int(meta.get("rank", 8))
    alpha = float(meta.get("alpha", 8.0))
    replace = list(meta.get("target_replace") or DEFAULT_REPLACE)
    network = LoRANetwork(
        transformer,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=replace,
        train_method=str(meta.get("train_method") or "full"),
        delimiter=str(meta.get("delimiter") or "-"),
        prefix=str(meta.get("prefix") or "lora_unet"),
    )
    network.to(device)
    state = load_file(str(weights), device="cpu")
    missing, unexpected = network.load_state_dict(state, strict=False)
    print(
        f"loaded {weights.name} modules={len(network.unet_loras)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if not network.unet_loras:
        raise RuntimeError(f"LoRA wrapped 0 modules for {replace}")
    if missing:
        raise RuntimeError(f"LoRA load missed {len(missing)} keys (first={missing[0]})")
    for lora in network.unet_loras:
        lora.multiplier = 0.0
    return network


def calibrate_weights(
    weights: Path,
    cache_dir: Path | None = None,
    model_dir: Path | None = None,
    device: torch.device | None = None,
    scales: list[float] | None = None,
    seed: int = 0,
    write: bool = True,
    timesteps: tuple[float, ...] | list[float] = DEFAULT_TIMESTEPS,
    n_latents: int = 2,
    guidance: float | None = None,
) -> dict[str, Any]:
    from conceptmod.textsliders.train_lora_music3 import _load_transformer

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    meta = read_sidecar(weights)
    cache = Path(cache_dir) if cache_dir is not None else guess_cache_dir(weights, meta)
    if cache is None:
        raise FileNotFoundError(f"pass --cache_dir; could not guess cache for {weights}")
    conditions, sidecar_guidance = load_conditions(cache, meta)
    use_guidance = sidecar_guidance if guidance is None else float(guidance)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_dir or DEFAULT_MODEL_DIR)
    print(
        f"calibrate {weights.name} cache={cache} guidance={use_guidance:g} device={device}",
        flush=True,
    )
    transformer = _load_transformer(model_path, device)
    network = attach_lora(transformer, weights, meta, device)
    result = calibrate_network(
        transformer,
        network,
        conditions,
        use_guidance,
        device,
        scales=scales,
        seed=seed,
        timesteps=timesteps,
        n_latents=n_latents,
    )
    print_table(result)
    if write:
        merged = merge_calibration(meta, result)
        path = write_sidecar(weights, merged)
        print(f"wrote {path}", flush=True)
    del network, transformer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
