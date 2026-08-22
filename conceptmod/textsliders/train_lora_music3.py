#!/usr/bin/env python3
"""Train a textual concept slider on MiniMax Music 3's flow transformer.

Does not import conceptmod PromptSettings (pydantic v1). Reuses LoRANetwork
with target MiniMaxMusic3Attention. Dummy mode never loads the 8B LM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork

DEFAULT_MODEL_DIR = Path("/ml2/music/models/MiniMax-Music3")
DEFAULT_SAVE_DIR = Path("/ml2/music/sliders-conceptmod/models/energy-slider")
DEFAULT_CACHE_DIR = Path("/ml2/music/sliders-conceptmod/cache/energy")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-music3.yaml"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-music3.yaml"
TARGET_REPLACE_ATTN = ["MiniMaxMusic3Attention"]
# Attention-only misses proj_in / preprocess_conv, where the condition is
# concatenated onto the latent. Full wraps those plus every block FF.
TARGET_REPLACE_FULL = [
    "MiniMaxMusic3Attention",
    "MiniMaxMusic3TransformerBlock",
    "MiniMaxMusic3Transformer1DModel",
]
TARGET_REPLACE = TARGET_REPLACE_ATTN
DEFAULT_LYRICS = "[verse]\nI keep walking through the night\nlooking for a little light\n[chorus]\nI will not go quietly"
# Match /ml2/music/app/generator.py _LOAD_ORDER, then drop unused modules after cache.
AR_COMPONENTS = (
    "tokenizer",
    "scheduler",
    "vocoder",
    "condition_encoder",
    "rvq_depth_decoder",
    "transformer",
    "language_model",
)
UNLOAD_AFTER_CACHE = ("language_model", "rvq_depth_decoder", "tokenizer", "vocoder", "scheduler", "transformer")
MIN_AR_FRAMES = 8
AR_RETRY_LIMIT = 5


@dataclass
class SliderPrompt:
    target: str
    positive: str
    negative: str
    neutral: str
    lyrics: str
    action: str = "enhance"
    guidance_scale: float = 4.0
    batch_size: int = 1
    unconditional: str = ""


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [-2.0, 2.0])


def _expand_attributes(item: dict) -> list[dict]:
    """Replicate a prompt row once per attribute, prefixing the attribute to all
    four captions (upstream sliders' disentanglement mechanism): the axis delta
    is then averaged over rows where the orthogonal trait is pinned both ways."""
    attributes = item.get("attributes")
    if not attributes:
        return [item]
    rows = []
    for attribute in attributes:
        prefix = str(attribute).strip()
        row = dict(item)
        for key in ("target", "positive", "negative", "neutral"):
            value = item.get(key)
            if value:
                row[key] = f"{prefix} {value}"
        rows.append(row)
    return rows


def load_prompts(path: Path) -> tuple[list[SliderPrompt], PromptsMeta]:
    raw = _load_yaml(path)
    meta = PromptsMeta()
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    prompts: list[SliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "target" not in item:
            raise ValueError(f"each prompt must be a mapping with target: {item!r}")
        for row in _expand_attributes(item):
            target = str(row["target"])
            prompts.append(
                SliderPrompt(
                    target=target,
                    positive=str(row.get("positive") or target),
                    negative=str(row.get("negative") or ""),
                    neutral=str(row.get("neutral") or target),
                    lyrics=str(row.get("lyrics") or DEFAULT_LYRICS),
                    action=str(row.get("action") or "enhance"),
                    guidance_scale=float(row.get("guidance_scale", 4.0)),
                    batch_size=int(row.get("batch_size", 1)),
                    unconditional=str(row.get("unconditional") or ""),
                )
            )
    return prompts, meta


def load_config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    raw = _load_yaml(path) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return raw


def _prompt_hash(prompt: str, lyrics: str, duration: float, seed: int) -> str:
    payload = f"{prompt}\n---\n{lyrics}\n---\n{duration:.3f}\n---\n{seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, prompt: str, lyrics: str, duration: float, seed: int) -> Path:
    return cache_dir / f"{_prompt_hash(prompt, lyrics, duration, seed)}.pt"


def _flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pick_device(index: int, dummy: bool) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free, _total = torch.cuda.mem_get_info(index)
    if dummy and free < 2 * 1024**3:
        print(f"cuda:{index} has {free / 1024**3:.1f} GiB free; dummy falling back to CPU")
        return torch.device("cpu")
    return torch.device(f"cuda:{index}")


def _set_lora_multiplier(network: LoRANetwork, value: float) -> None:
    for lora in network.unet_loras:
        lora.multiplier = value


def _align_conditions(conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lengths = [int(tensor.shape[1]) for tensor in conditions.values()]
    if not lengths:
        raise ValueError("no conditions to align")
    min_len = min(lengths)
    if min_len < 1:
        raise ValueError("condition latent length is zero")
    aligned = {}
    for name, tensor in conditions.items():
        cropped = tensor[:, :min_len].contiguous()
        aligned[name] = cropped
        if tensor.shape[1] != min_len:
            print(f"cropped {name} condition {tensor.shape[1]} -> {min_len}")
    return aligned


def _make_dummy_transformer() -> torch.nn.Module:
    from diffusers import MiniMaxMusic3Transformer1DModel

    return MiniMaxMusic3Transformer1DModel(
        in_channels=128,
        condition_dim=2048,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=32,
        ff_inner_dim=128,
        rotary_dim=16,
        fourier_embedding_dim=64,
    )


def _load_transformer(
    model_dir: Path, device: torch.device, dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    from diffusers import MiniMaxMusic3Transformer1DModel

    transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    return transformer.to(device)


def _load_ar_pipeline(model_dir: Path, device: torch.device):
    from diffusers import ModularPipeline

    local_kwargs = {
        "pretrained_model_name_or_path": str(model_dir),
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    pipe = ModularPipeline.from_pretrained(str(model_dir), local_files_only=True)
    for name in AR_COMPONENTS:
        print(f"loading {name} for AR cache")
        pipe.load_components(names=name, **local_kwargs)
    if device.type == "cuda":
        pipe.to(str(device))
    return pipe


def _unload_ar(pipe: Any) -> None:
    for name in UNLOAD_AFTER_CACHE:
        if hasattr(pipe, name):
            setattr(pipe, name, None)
    _flush()


def _min_ar_frames(duration: float) -> int:
    return max(MIN_AR_FRAMES, int(float(duration) * 12.5))


def _attach_lm_slider(pipe: Any, weights: Path, device: torch.device) -> LoRANetwork:
    from safetensors.torch import load_file

    network = LoRANetwork(
        pipe.language_model,
        rank=8,
        alpha=8.0,
        multiplier=1.0,
        target_replace=["Qwen3Attention"],
        train_method="full",
        delimiter="-",
        prefix="lora_te",
    )
    network.to(device)
    state = load_file(str(weights), device="cpu")
    missing, unexpected = network.load_state_dict(state, strict=False)
    print(
        f"AR LM slider {weights.name} modules={len(network.unet_loras)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing or not network.unet_loras:
        raise RuntimeError(f"failed to wrap LM slider: missing={len(missing)}")
    for lora in network.unet_loras:
        lora.multiplier = 0.0
    return network


def _encode_condition(
    pipe: Any,
    prompt: str,
    lyrics: str,
    duration: float,
    seed: int,
    device: torch.device,
    lm_network: LoRANetwork | None = None,
    lm_scale: float = 0.0,
) -> torch.Tensor:
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    min_frames = _min_ar_frames(duration)
    last_error = "AR produced no frames"
    for attempt in range(AR_RETRY_LIMIT):
        attempt_seed = int(seed) + attempt
        generator = torch.Generator(device=device).manual_seed(attempt_seed)
        state = PipelineState()
        state.set("prompt", prompt)
        state.set("lyrics", lyrics)
        state.set("audio_duration", float(duration))
        state.set("generator", generator)
        blocks = pipe._blocks.sub_blocks
        if lm_network is not None:
            lm_network.set_lora_slider(float(lm_scale))
            with lm_network:
                blocks["text_encoder"](pipe, state)
                blocks["semantic_generator"](pipe, state)
        else:
            blocks["text_encoder"](pipe, state)
            blocks["semantic_generator"](pipe, state)
        frame_hiddens = state.get("frame_hiddens")
        if frame_hiddens is None:
            last_error = "AR stage did not produce frame_hiddens"
            print(f"AR retry {attempt + 1}/{AR_RETRY_LIMIT}: {last_error} seed={attempt_seed}")
            continue
        n_frames = int(frame_hiddens.shape[1])
        print(f"AR frames={n_frames} min={min_frames} seed={attempt_seed} prompt={prompt[:60]!r}")
        if n_frames < min_frames:
            last_error = f"AR early EOS: {n_frames} < {min_frames} frames"
            print(f"AR retry {attempt + 1}/{AR_RETRY_LIMIT}: {last_error}")
            continue
        condition = pipe.condition_encoder(frame_hiddens.to(pipe._execution_device))
        return condition.detach().to(dtype=torch.bfloat16).cpu()
    raise RuntimeError(f"{last_error} after {AR_RETRY_LIMIT} seeds starting at {seed}")


def _cache_or_encode(
    pipe: Any | None,
    cache_dir: Path,
    prompt: str,
    lyrics: str,
    duration: float,
    seed: int,
    device: torch.device,
    skip_ar: bool,
    lm_network: LoRANetwork | None = None,
    lm_scale: float = 0.0,
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prompt = prompt if lm_network is None else f"__lm{lm_scale:+.3f}__\n{prompt}"
    path = _cache_path(cache_dir, cache_prompt, lyrics, duration, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        condition = payload["condition"] if isinstance(payload, dict) else payload
        print(f"cache hit {path.name} shape={tuple(condition.shape)} lm={lm_scale:g}")
        return condition
    if skip_ar:
        raise FileNotFoundError(f"--skip_ar set but cache missing: {path}")
    if pipe is None:
        raise RuntimeError("AR pipeline is not loaded")
    print(f"encoding condition -> {path.name} lm={lm_scale:g} prompt={prompt[:60]!r}")
    condition = _encode_condition(
        pipe, prompt, lyrics, duration, int(seed), device, lm_network=lm_network, lm_scale=lm_scale
    )
    torch.save(
        {
            "condition": condition,
            "prompt": prompt,
            "lyrics": lyrics,
            "duration": float(duration),
            "seed": int(seed),
            "lm_scale": float(lm_scale),
        },
        path,
    )
    return condition


def build_conditions(
    prompts: list[SliderPrompt],
    cache_dir: Path,
    duration: float,
    seeds: list[int],
    device: torch.device,
    model_dir: Path,
    skip_ar: bool,
    dummy: bool,
    lm_weights: Path | None = None,
) -> list[tuple[SliderPrompt, dict[str, torch.Tensor]]]:
    """Return one (prompt, conditions) entry per (prompt row, AR seed)."""
    if dummy:
        latent_len = 32
        entries = []
        for prompt in prompts:
            for _seed in seeds:
                noise = {
                    name: torch.randn(1, latent_len, 2048)
                    for name in ("target", "positive", "negative", "neutral")
                }
                entries.append((prompt, noise))
        return entries

    use_lm = lm_weights is not None
    jobs: list[tuple[str, str, float, int]] = []
    for prompt in prompts:
        base = prompt.neutral or prompt.target
        for seed in seeds:
            if use_lm:
                # Same caption; LM slider writes the axis-shifted AR plan the transformer should match.
                jobs.extend(
                    [
                        (base, prompt.lyrics, 0.0, seed),
                        (base, prompt.lyrics, 1.0, seed),
                        (base, prompt.lyrics, -1.0, seed),
                    ]
                )
            else:
                jobs.extend(
                    [
                        (prompt.target, prompt.lyrics, 0.0, seed),
                        (prompt.positive, prompt.lyrics, 0.0, seed),
                        (prompt.negative, prompt.lyrics, 0.0, seed),
                        (prompt.neutral, prompt.lyrics, 0.0, seed),
                    ]
                )

    def _key(text: str, lyrics: str, scale: float, seed: int) -> str:
        hashed = text if not use_lm else f"__lm{scale:+.3f}__\n{text}"
        return _prompt_hash(hashed, lyrics, duration, seed)

    missing_jobs = []
    for text, lyrics, scale, seed in jobs:
        hashed = text if not use_lm else f"__lm{scale:+.3f}__\n{text}"
        if not _cache_path(cache_dir, hashed, lyrics, duration, seed).exists():
            missing_jobs.append((text, scale, seed))
    need_ar = bool(missing_jobs)

    pipe = None
    lm_network = None
    if need_ar:
        if skip_ar:
            # Name every missing key input: three separate jobs have failed on
            # this error with no way to tell WHICH (prompt, seed, duration)
            # cell the cache lacked. The cache is keyed per caption text, per
            # AR seed, and per duration -- all three must match.
            detail = "; ".join(
                f"prompt={text[:48]!r} seed={seed} dur={duration:g}"
                + (f" lm={scale:+g}" if use_lm else "")
                for text, scale, seed in missing_jobs[:4]
            )
            more = "" if len(missing_jobs) <= 4 else f" (+{len(missing_jobs) - 4} more)"
            raise FileNotFoundError(
                f"--skip_ar set but {len(missing_jobs)} condition(s) missing from "
                f"{cache_dir}: {detail}{more}"
            )
        pipe = _load_ar_pipeline(model_dir, device)
        if use_lm:
            lm_network = _attach_lm_slider(pipe, lm_weights, device)

    encoded: dict[str, torch.Tensor] = {}
    try:
        for text, lyrics, scale, seed in jobs:
            cache_key = _key(text, lyrics, scale, seed)
            if cache_key in encoded:
                continue
            encoded[cache_key] = _cache_or_encode(
                pipe,
                cache_dir,
                text,
                lyrics,
                duration,
                seed,
                device,
                skip_ar,
                lm_network=lm_network,
                lm_scale=scale,
            )
    finally:
        if pipe is not None:
            _unload_ar(pipe)
            del pipe
            _flush()

    entries = []
    for prompt in prompts:
        base = prompt.neutral or prompt.target
        for seed in seeds:
            if use_lm:
                branch = {
                    "target": encoded[_key(base, prompt.lyrics, 0.0, seed)],
                    "positive": encoded[_key(base, prompt.lyrics, 1.0, seed)],
                    "negative": encoded[_key(base, prompt.lyrics, -1.0, seed)],
                    "neutral": encoded[_key(base, prompt.lyrics, 0.0, seed)],
                }
            else:
                branch = {
                    "target": encoded[_key(prompt.target, prompt.lyrics, 0.0, seed)],
                    "positive": encoded[_key(prompt.positive, prompt.lyrics, 0.0, seed)],
                    "negative": encoded[_key(prompt.negative, prompt.lyrics, 0.0, seed)],
                    "neutral": encoded[_key(prompt.neutral, prompt.lyrics, 0.0, seed)],
                }
            entries.append((prompt, _align_conditions(branch)))
    return entries


def _slider_loss(
    vel: torch.Tensor,
    vel_neu: torch.Tensor,
    axis: torch.Tensor,
    kind: str,
    mag_weight: float,
    gain_weight: float = 0.0,
    gain_mode: str = "penalty",
    gain_tw: float = 1.0,
) -> torch.Tensor:
    """`axis` is the signed target delta, guidance * (vel_pos - vel_neg) for +1.

    mse   — plain MSE against vel_neu + axis. Its magnitude tracks ||axis||^2,
            which spans ~200x across t (0.37 of ||vel_neu|| at t=0.05 down to
            0.023 at t=0.97), so a handful of low-t steps dominate the whole run:
            in the shipped triphop log the top 5% of steps carry 73% of total
            MSE mass. Kept for A/B against the v3/v4 checkpoints.
    nmse  — the same MSE divided by the per-step target energy, so every
            timestep contributes equally.
    cos   — optimize the reported metric directly (scale-free), plus a
            magnitude term pulling ||delta|| towards ||axis||.
    nmse_ortho — gain and shape supervised as SEPARATE quantities: nmse fits
            only the components of the edit orthogonal to vel_neu; the parallel
            (pure-gain) component is supervised solely by the gain term, so it
            requires gain_weight > 0 and gain_mode="match" (enforced at parse).
            Motivation: the gain component is the one direction that points the
            same way at every one of the ~50 solve steps, so per-step error in
            it compounds multiplicatively at render while shape error partly
            cancels; a joint MSE prices both identically.

    gain_mode:
    penalty — push the gain component toward ZERO (the original --gain_penalty;
            measured inert on both the dust and trip-hop pairs, kept as a
            negative control). Zero is also the wrong target: the teacher's own
            delta carries a level component.
    match — push the gain component toward the TEACHER's own gain component
            (axis · unit_neu), i.e. exactly as much level change as the concept
            actually asks for, no more.

    gain_tw: timestep-dependent multiplier on the gain term (1.0 = uniform).
            The caller passes ~2*(1-t): a per-step gain bias introduced early in
            the solve (low t, near noise) has the most remaining steps to
            compound through, so it is priced highest.
    """
    delta = vel - vel_neu
    unit = vel_neu.flatten()
    unit = unit / unit.norm().clamp_min(1e-8)
    if gain_weight > 0.0:
        # The component of the delta along vel_neu just rescales the velocity.
        # It is only ~16% of the concept axis, but it is the one component that
        # points the same way at every denoise step, so it compounds through the
        # ~50-step solve while shape components partly cancel: the rendered
        # slider comes out ~5x too much level change and ~10x too little
        # brightness.
        g_delta = delta.flatten() @ unit
        if gain_mode == "match":
            g_target = axis.flatten() @ unit
        elif gain_mode == "penalty":
            g_target = axis.new_zeros(())
        else:
            raise ValueError(f"unknown gain_mode {gain_mode!r}")
        gain = (g_delta - g_target) / axis.norm().clamp_min(1e-8)
        gain_term = gain_weight * float(gain_tw) * gain.pow(2)
    else:
        gain_term = 0.0
    if kind == "mse":
        return torch.nn.functional.mse_loss(vel, vel_neu + axis) + gain_term
    if kind == "nmse":
        scale = axis.pow(2).mean().clamp_min(1e-8)
        return torch.nn.functional.mse_loss(vel, vel_neu + axis) / scale + gain_term
    if kind == "nmse_ortho":
        d_par = (delta.flatten() @ unit)
        a_par = (axis.flatten() @ unit)
        d_perp = delta - (d_par * unit).view_as(delta)
        a_perp = axis - (a_par * unit).view_as(axis)
        scale = a_perp.pow(2).mean().clamp_min(1e-8)
        return (d_perp - a_perp).pow(2).mean() / scale + gain_term
    if kind == "cos":
        cos = torch.nn.functional.cosine_similarity(
            delta.flatten().unsqueeze(0), axis.flatten().unsqueeze(0)
        ).squeeze()
        ratio = delta.norm() / axis.norm().clamp_min(1e-8)
        return (1.0 - cos) + mag_weight * (ratio - 1.0).pow(2) + gain_term
    raise ValueError(f"unknown loss {kind!r}")


@dataclass
class EvalProbe:
    """A fixed, run-independent set of (x_t, t) instances with cached teacher
    velocities. The per-step `cos` in the train log is one random draw out of a
    field whose target norm spans ~200x across t, so it cannot be compared
    between runs; this probe pins the draws so two runs differing only in rank /
    targets / loss are measured on identical inputs."""

    latents: list[torch.Tensor]
    timesteps: list[torch.Tensor]
    directions: list[torch.Tensor]  # vel_pos - vel_neg
    neutrals: list[torch.Tensor]  # vel_neu
    t_values: list[float]
    cond_target: torch.Tensor
    guidance: float
    # Defaulted fields must come last. Neutral does not lie on the pos-neg line
    # (its rms sits below BOTH poles here), so the axis is not what either side
    # of the slider should render; keep each pole's displacement from neutral to
    # score --target_mode pole runs on their own target as well as on the axis.
    edges_plus: list[torch.Tensor] = field(default_factory=list)  # +side pole - vel_neu
    edges_minus: list[torch.Tensor] = field(default_factory=list)  # -side pole - vel_neu
    # Prompt-pair geometry: how far neutral sits off the pos-neg chord. A merged
    # LoRA's delta is ~odd in its multiplier, so only the odd part of the two
    # edges, (edge_plus - edge_minus)/2 = (v_pos - v_neg)/2, is reachable by one
    # bidirectional slider; even_over_odd says how much of the caption swap is not.
    geometry: dict = field(default_factory=dict)


EVAL_TIMESTEPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@torch.no_grad()
def build_eval_probe(
    transformer: torch.nn.Module,
    entry: tuple[SliderPrompt, dict[str, torch.Tensor]],
    x0_bank: list[torch.Tensor] | None,
    device: torch.device,
    seed: int,
    amp: bool,
    n_noise: int = 2,
) -> EvalProbe:
    """The probe's x0 anchors come from `x0_bank`, i.e. the same clean latents the
    run trains on. That is fine when every run shares one bank, but it gives a
    home-field advantage to runs that concentrate on those anchors: a --x0_per_row 8
    run spreads capacity over 8 anchors and is then scored on 2 of them. Pass a
    bank built with a probe-only seed (see --eval_holdout) to measure
    generalization across anchors instead of fit to the training ones."""
    prompt, conds = entry
    length = int(conds["target"].shape[1])
    dtype = torch.bfloat16 if amp else torch.float32

    def _cond(name: str) -> torch.Tensor:
        return conds[name].to(device=device, dtype=dtype)

    cond_pos, cond_neg, cond_neu = _cond("positive"), _cond("negative"), _cond("neutral")
    sign = 1.0 if prompt.action == "enhance" else -1.0
    latents, timesteps, directions, neutrals, t_values = [], [], [], [], []
    edges_plus, edges_minus = [], []
    for t_val in EVAL_TIMESTEPS:
        for k in range(max(1, n_noise)):
            generator = torch.Generator(device=device.type).manual_seed(int(seed) * 1000 + k)
            noise = torch.randn(1, 128, length, device=device, dtype=torch.float32, generator=generator)
            if x0_bank:
                x0 = x0_bank[k % len(x0_bank)].to(device=device, dtype=torch.float32)
                x_t = (1.0 - t_val) * noise + t_val * x0
            else:
                x_t = noise
            x_t = x_t.to(dtype)
            timestep = torch.full((1,), float(t_val), device=device, dtype=dtype)
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    vel_pos = transformer(x_t, timestep, cond_pos, return_dict=False)[0]
                    vel_neg = transformer(x_t, timestep, cond_neg, return_dict=False)[0]
                    vel_neu = transformer(x_t, timestep, cond_neu, return_dict=False)[0]
            else:
                vel_pos = transformer(x_t, timestep, cond_pos, return_dict=False)[0]
                vel_neg = transformer(x_t, timestep, cond_neg, return_dict=False)[0]
                vel_neu = transformer(x_t, timestep, cond_neu, return_dict=False)[0]
            latents.append(x_t)
            timesteps.append(timestep)
            # Signed so an `action: erase` prompt file reports a positive cos for
            # a working slider, same as an `enhance` one -- otherwise the two
            # conventions are not comparable.
            directions.append(sign * (vel_pos.float() - vel_neg.float()))
            neutrals.append(vel_neu.float())
            pole_plus = vel_pos if sign > 0 else vel_neg
            pole_minus = vel_neg if sign > 0 else vel_pos
            edges_plus.append(pole_plus.float() - vel_neu.float())
            edges_minus.append(pole_minus.float() - vel_neu.float())
            t_values.append(float(t_val))
    cos_anti, even_over_odd = [], []
    for edge_p, edge_m in zip(edges_plus, edges_minus):
        cos_anti.append(
            torch.nn.functional.cosine_similarity(
                edge_p.flatten().unsqueeze(0), -edge_m.flatten().unsqueeze(0)
            ).item()
        )
        odd = 0.5 * (edge_p - edge_m)
        even = 0.5 * (edge_p + edge_m)
        even_over_odd.append((even.norm() / odd.norm().clamp_min(1e-8)).item())
    geometry = {
        "cos_edge_antipodal": sum(cos_anti) / max(len(cos_anti), 1),
        "even_over_odd": sum(even_over_odd) / max(len(even_over_odd), 1),
    }
    print(
        f"probe prompt geometry: cos(edge+, -edge-)={geometry['cos_edge_antipodal']:.4f} "
        f"||even||/||odd||={geometry['even_over_odd']:.3f} "
        "(one bidirectional LoRA reaches only the odd part of the caption swap)",
        flush=True,
    )
    return EvalProbe(
        latents=latents,
        timesteps=timesteps,
        directions=directions,
        neutrals=neutrals,
        edges_plus=edges_plus,
        edges_minus=edges_minus,
        geometry=geometry,
        t_values=t_values,
        cond_target=_cond("target"),
        guidance=float(prompt.guidance_scale),
    )


@torch.no_grad()
def evaluate_probe(
    transformer: torch.nn.Module,
    network: LoRANetwork,
    probe: EvalProbe,
    amp: bool,
) -> dict[str, float]:
    """Mean cos / magnitude of the LoRA delta against the concept axis, over the
    fixed probe. `mag` is ||delta(+1)|| / ||guidance * (vel_pos - vel_neg)||:
    cos says whether the axis is tracked, mag whether it is reached."""

    def _forward(x, t, slider: float) -> torch.Tensor:
        network.set_lora_slider(slider)
        _set_lora_multiplier(network, 1.0)
        with network:
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = transformer(x, t, probe.cond_target, return_dict=False)[0]
            else:
                out = transformer(x, t, probe.cond_target, return_dict=False)[0]
        _set_lora_multiplier(network, 0.0)
        return out.float()

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()

    was_training = network.training
    network.eval()
    cos_pos, cos_neg, collapse, mag, proj, by_t = [], [], [], [], [], {}
    cos_edge, cos_edge_neg = [], []
    proj_num = 0.0  # sum_i <delta_i, axis_hat_i>, absolute units
    proj_den = 0.0  # sum_i ||guidance * axis_i||
    edges_plus = probe.edges_plus or [None] * len(probe.latents)
    edges_minus = probe.edges_minus or [None] * len(probe.latents)
    for x, t, direction, neutral, t_val, edge_p, edge_m in zip(
        probe.latents, probe.timesteps, probe.directions, probe.neutrals, probe.t_values,
        edges_plus, edges_minus,
    ):
        delta_plus = _forward(x, t, 1.0) - neutral
        delta_minus = _forward(x, t, -1.0) - neutral
        c = _cos(delta_plus, direction)
        cos_pos.append(c)
        cos_neg.append(_cos(delta_minus, -direction))
        collapse.append(_cos(delta_plus, delta_minus))
        if edge_p is not None:
            cos_edge.append(_cos(delta_plus, edge_p))
            cos_edge_neg.append(_cos(delta_minus, edge_m))
        axis_norm = (probe.guidance * direction).norm().clamp_min(1e-8)
        mag.append((delta_plus.norm() / axis_norm).item())
        proj.append(c * mag[-1])
        # Absolute-units axis displacement. To first order the Euler solve gives
        # delta_x_final = integral of delta_v dt, so what reaches the render is the
        # *unnormalized* axis component summed over t. cos and mag are both
        # per-instance normalized, which silently reweights the timesteps where
        # the edit is largest; this does not.
        proj_num += float(
            (delta_plus.flatten() @ (direction.flatten() / direction.norm().clamp_min(1e-8))).item()
        )
        proj_den += float(axis_norm.item())
        by_t.setdefault(t_val, []).append(c)
    if was_training:
        network.train()

    def _mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    result = {
        "cos": _mean(cos_pos),
        "cos_neg": _mean(cos_neg),
        "collapse": _mean(collapse),
        "mag": _mean(mag),
        "proj": _mean(proj),
        "proj_abs": proj_num / max(proj_den, 1e-8),
        "n": len(cos_pos),
    }
    if cos_edge:
        result["cos_edge"] = _mean(cos_edge)
        result["cos_edge_neg"] = _mean(cos_edge_neg)
    result["cos_by_t"] = {f"{t:g}": round(_mean(v), 4) for t, v in sorted(by_t.items())}
    return result


def _condition_hash(condition: torch.Tensor) -> str:
    data = condition.to(torch.float32).cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


INFERENCE_CFG = 1.7  # ClassifierFreeGuidance default in MiniMaxMusic3ChunkDenoiseInner


@torch.no_grad()
def _generate_x0(
    transformer: torch.nn.Module,
    condition: torch.Tensor,
    model_dir: Path,
    device: torch.device,
    seed: int,
    num_steps: int = 30,
    amp: bool = True,
) -> torch.Tensor:
    """Flow-integrate a clean latent for one condition, replicating inference:
    sigmas linspace(1, 1/N), CFG 1.7 with a zeros condition on the uncond branch.
    Single window at the condition's full length (training forwards use the same
    length, so the anchor stays self-consistent even past the 200-frame chunk)."""
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler", local_files_only=True
    )
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device)

    dtype = torch.bfloat16 if amp else torch.float32
    cond = condition.to(device=device, dtype=dtype)
    zeros = torch.zeros_like(cond)
    generator = torch.Generator(device=device.type).manual_seed(int(seed))
    latents = torch.randn(
        (1, 128, int(cond.shape[1])), device=device, dtype=dtype, generator=generator
    )
    for t in scheduler.timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
                vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
        else:
            vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
            vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
        velocity = vel_uncond + INFERENCE_CFG * (vel_cond - vel_uncond)
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
    return latents.to(torch.float32).cpu()


def build_x0_banks(
    entries: list[tuple[SliderPrompt, dict[str, torch.Tensor]]],
    transformer: torch.nn.Module,
    cache_dir: Path,
    model_dir: Path,
    device: torch.device,
    seed: int,
    per_row: int,
    num_steps: int,
    dummy: bool,
    source: str = "neutral",
) -> list[list[torch.Tensor]]:
    """One bank of clean latents per (row, cond-seed) entry, cached on disk.
    Generated from the `source` condition (default neutral) with LoRA
    multipliers already at 0. `source` exists for the anchor-dependence
    diagnostic: probe geometry anchored to neutral-generated latents reads the
    same for unrelated caption pairs, so measuring it on positive- or
    negative-anchored latents separates "the pair's geometry" from "distance
    off the anchor's own trajectory"."""
    banks: list[list[torch.Tensor]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, (_prompt, conds) in enumerate(entries):
        neutral = conds[source]
        bank: list[torch.Tensor] = []
        for i in range(max(1, per_row)):
            x0_seed = int(seed) * 100 + i
            if dummy:
                bank.append(torch.randn(1, 128, int(neutral.shape[1])))
                continue
            path = cache_dir / f"x0_{_condition_hash(neutral)}_s{x0_seed}.pt"
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                bank.append(payload["latents"])
                print(f"x0 cache hit {path.name}", flush=True)
                continue
            print(f"generating x0 anchor {index + 1}/{len(entries)} seed={x0_seed}", flush=True)
            latents = _generate_x0(
                transformer, neutral, model_dir, device, x0_seed, num_steps=num_steps
            )
            torch.save({"latents": latents, "seed": x0_seed}, path)
            bank.append(latents)
        banks.append(bank)
    return banks


@torch.no_grad()
def rollout_states(
    transformer: torch.nn.Module,
    network: LoRANetwork,
    condition: torch.Tensor,
    model_dir: Path,
    device: torch.device,
    seed: int,
    scales: list[float],
    num_steps: int,
    amp: bool,
    t_lo: float = 0.02,
    t_hi: float = 0.98,
) -> list[tuple[torch.Tensor, float, float]]:
    """States the slider actually visits, harvested from a slider-on rollout.

    Training x_t is normally (1-t)*eps + t*x0 with x0 from an *unperturbed*
    trajectory, so the adapter only ever sees states the base model would have
    reached without it -- and the eval probe scores it at those same states.
    At inference the adapter produces its own trajectory and the deviation
    can compound (the DAgger argument). NOTE the teacher-4s renders weaken the
    drift story for the observed level collapse: the teacher itself craters rms
    at fractional strengths (axis at post-CFG net 1.0 renders -54% rms), and the
    probe's proj_abs ~0.57 means the LoRA under-delivers magnitude, so much of
    the "collapse at 1.7x" is the teacher's own curve sampled at the effective
    (not nominal) strength. Rolling the *current*
    adapter out and training on the states it lands in is the DAgger fix. The
    target at those states is still the closed-loop teacher, so the adapter
    learns a restoring force where it operates instead of only on the base
    manifold.

    Replicates the inference sampler: sigmas linspace(1, 1/N), CFG 1.7 against
    a zeros condition, adapter live on BOTH branches -- a merged LoRA cannot be
    excluded from the unconditional one.
    """
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler", local_files_only=True
    )
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device)

    dtype = torch.bfloat16 if amp else torch.float32
    cond = condition.to(device=device, dtype=dtype)
    zeros = torch.zeros_like(cond)
    states: list[tuple[torch.Tensor, float, float]] = []
    for scale in scales:
        # set_timesteps also rewinds _step_index; the scheduler is stateful and
        # is exhausted after one pass.
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        generator = torch.Generator(device=device.type).manual_seed(int(seed))
        latents = torch.randn(
            (1, 128, int(cond.shape[1])), device=device, dtype=dtype, generator=generator
        )
        network.set_lora_slider(float(scale))
        with network:
            for t in scheduler.timesteps:
                t_val = float(t)
                if t_lo <= t_val <= t_hi:
                    states.append((latents.detach().float().clone(), t_val, float(scale)))
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                if amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
                        vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
                else:
                    vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
                    vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
                velocity = vel_uncond + INFERENCE_CFG * (vel_cond - vel_uncond)
                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
    network.set_lora_slider(1.0)
    return states


def train(args: argparse.Namespace) -> Path:
    if args.loss == "nmse_ortho" and not (args.gain_penalty > 0.0 and args.gain_mode == "match"):
        raise SystemExit(
            "--loss nmse_ortho leaves the pure-gain component of the delta out of the "
            "shape loss entirely, so it must be supervised by the gain term: pass "
            "--gain_penalty > 0 --gain_mode match. Without that the gain component is "
            "unconstrained — and gain is the component that compounds into level collapse."
        )
    config = load_config_defaults(Path(args.config_file) if args.config_file else None)
    prompts_path = Path(args.prompts_file or config.get("prompts_file") or DEFAULT_PROMPTS)
    if not prompts_path.is_absolute():
        candidate = _REPO_ROOT / prompts_path
        prompts_path = candidate if candidate.exists() else Path(__file__).resolve().parent / prompts_path
    prompts, prompts_meta = load_prompts(prompts_path)
    if args.plus_label:
        prompts_meta.plus_label = args.plus_label
    if args.minus_label:
        prompts_meta.minus_label = args.minus_label

    train_cfg = config.get("train") or {}
    save_cfg = config.get("save") or {}
    network_cfg = config.get("network") or {}
    pretrained = config.get("pretrained_model") or {}

    name = args.name or save_cfg.get("name") or "energy"
    rank = int(args.rank if args.rank is not None else network_cfg.get("rank", 4))
    alpha = float(args.alpha if args.alpha is not None else network_cfg.get("alpha", 1.0))
    steps = int(args.steps if args.steps is not None else train_cfg.get("iterations", 80))
    duration = float(args.duration if args.duration is not None else 4.0)
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-4))
    if args.guidance is not None:
        for prompt in prompts:
            prompt.guidance_scale = float(args.guidance)
    target_replace = TARGET_REPLACE_FULL if args.targets == "full" else TARGET_REPLACE_ATTN
    if args.target_mode == "pole" and args.bidirectional:
        raise SystemExit(
            "--target_mode pole with --bidirectional is ill-posed: a LoRA's delta is "
            "(to first order) an odd function of its multiplier, but the two pole "
            "displacements are not antipodal (neutral sits off the pos-neg chord), so "
            "the pair of targets demands a large even component. Measured: P-pole / "
            "P-pole-traj50 ended with collapse +0.62/+0.73, mag 9.4/5.1, cos_edge "
            "0.06/0.11 -- the optimizer inflates the weights chasing second-order "
            "terms and destroys the direction. Train one side per LoRA instead: "
            "--target_mode pole --no-bidirectional (swap the prompt file's "
            "positive/negative to get the other side)."
        )
    lr_name = str(args.lr_scheduler or train_cfg.get("lr_scheduler", "cosine"))
    model_dir = Path(args.model_dir or pretrained.get("name_or_path") or DEFAULT_MODEL_DIR)
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    save_dir = Path(args.save_dir or save_cfg.get("path") or DEFAULT_SAVE_DIR)
    if args.dummy:
        steps = min(steps, 2)
    device = _pick_device(int(args.device), dummy=bool(args.dummy))
    seed = int(args.seed)
    torch.manual_seed(seed)
    if args.cond_seeds:
        cond_seeds = [int(part) for part in str(args.cond_seeds).split(",") if part.strip()]
    else:
        cond_seeds = [seed]

    print(
        f"train music3 slider name={name} rank={rank} alpha={alpha} steps={steps} "
        f"duration={duration} device={device} dummy={bool(args.dummy)} "
        f"targets={args.targets} lm_condition={bool(args.lm_weights)} "
        f"xt_mode={args.xt_mode} traj_frac={args.traj_frac} "
        f"target_mode={args.target_mode} "
        f"bidirectional={bool(args.bidirectional)} cond_seeds={cond_seeds}"
    )

    entries = build_conditions(
        prompts=prompts,
        cache_dir=cache_dir,
        duration=duration,
        seeds=cond_seeds,
        device=device,
        model_dir=model_dir,
        skip_ar=bool(args.skip_ar),
        dummy=bool(args.dummy),
        lm_weights=Path(args.lm_weights) if args.lm_weights else None,
    )

    if args.dummy:
        transformer = _make_dummy_transformer().to(device)
        transformer.requires_grad_(False)
        transformer.train()
    else:
        transformer = _load_transformer(
            model_dir, device, torch.float32 if args.teacher_fp32 else torch.bfloat16
        )

    network = LoRANetwork(
        transformer,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=target_replace,
        train_method="full",
        delimiter="-",
    )
    network.to(device)
    if len(network.unet_loras) == 0:
        raise RuntimeError(f"LoRANetwork found 0 modules for {target_replace}")
    print(f"LoRA modules: {len(network.unet_loras)}")
    _set_lora_multiplier(network, 0.0)

    for param in transformer.parameters():
        param.requires_grad_(False)
    for param in network.parameters():
        param.requires_grad_(True)

    if args.xt_mode == "noise":
        x0_banks: list[list[torch.Tensor]] = []
    else:
        x0_banks = build_x0_banks(
            entries,
            transformer,
            cache_dir=cache_dir,
            model_dir=model_dir,
            device=device,
            seed=seed,
            per_row=int(args.x0_per_row),
            num_steps=int(args.x0_steps),
            dummy=bool(args.dummy),
            source=str(args.x0_source),
        )

    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    if lr_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    amp_enabled = device.type == "cuda" and not args.dummy
    amp_dtype = torch.bfloat16
    data_rng = torch.Generator(device=device.type).manual_seed(seed)

    probe = None
    eval_every = max(0, int(args.eval_every))
    probe_bank = x0_banks[0] if x0_banks else None
    if eval_every and entries and args.eval_holdout and args.xt_mode != "noise":
        # A clean latent the run never trains on, so the probe scores generalization
        # across anchors rather than fit to the two it was fed.
        print("generating held-out eval anchor", flush=True)
        probe_bank = build_x0_banks(
            entries[:1], transformer, cache_dir=cache_dir, model_dir=model_dir,
            device=device, seed=int(args.eval_seed), per_row=1,
            num_steps=int(args.x0_steps), dummy=bool(args.dummy),
        )[0]
    if eval_every and entries:
        print("building fixed eval probe", flush=True)
        probe = build_eval_probe(
            transformer,
            entries[0],
            probe_bank,
            device=device,
            seed=int(args.eval_seed),
            amp=amp_enabled,
        )
        print(f"eval probe: {len(probe.latents)} instances over t={list(EVAL_TIMESTEPS)}", flush=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_alpha{alpha}_rank{rank}_{args.targets}"
    if args.dummy:
        stem = f"{stem}_dummy"
    log_path = save_dir / f"{stem}_train.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    last_loss = None
    last_eval: dict[str, float] | None = None
    # Long runs can degrade: a 2000-step fp32-teacher run peaked at cos 0.8529
    # (step 1100) and finished at 0.7563, with the +1 pole drifting while -1 held.
    # Keep the best-scoring weights so `_last` is not the only thing on disk.
    best_eval: dict[str, float] | None = None
    best_step = 0
    mid_step = max(1, steps // 2)
    traj_frac = float(args.traj_frac)
    traj_scales = [float(x) for x in str(args.traj_scales).split(",") if x.strip()]
    traj_refresh = max(1, int(args.traj_refresh))
    traj_bank: list[tuple[torch.Tensor, float, float]] = []
    # A private stream: the traj draw must not perturb data_rng, or a traj run
    # and a traj_frac=0 run would no longer see the same (t, eps) sequence.
    traj_rng = random.Random(seed * 7919 + 13)

    progress = tqdm(range(steps), desc="train")
    for step in progress:
        prompt, conds = entries[step % len(entries)]
        x0_bank = x0_banks[step % len(entries)] if x0_banks else None
        batch = max(1, int(prompt.batch_size))
        latent_len = int(conds["target"].shape[1])
        # Draw x_t/t from a dedicated generator, not the global RNG: LoRA init
        # consumes global RNG in proportion to rank and module count, so two runs
        # that differ only in --rank or --targets used to see different (t, eps)
        # sequences and could not be compared step for step.
        noise = torch.randn(
            batch, 128, latent_len, device=device, dtype=torch.float32, generator=data_rng
        )
        # Model convention (denoise.py): flow time t in [0,1], 0 = noise, 1 = clean;
        # x_t = (1-t)·ε + t·x0. Pure noise is only on-manifold near t=0, so anchor
        # x_t to a generated clean latent except in the explicit noise/mix modes.
        if traj_frac > 0.0 and not args.dummy and step % traj_refresh == 0:
            traj_bank = rollout_states(
                transformer, network, conds["target"], model_dir=model_dir,
                device=device, seed=seed * 1000 + step, scales=traj_scales,
                num_steps=int(args.traj_steps), amp=amp_enabled,
            )
            _set_lora_multiplier(network, 0.0)
            print(
                f"traj bank refreshed at step {step}: {len(traj_bank)} states "
                f"over scales {traj_scales}",
                flush=True,
            )
        use_traj = bool(traj_bank) and traj_rng.random() < traj_frac
        state_scale = 1.0
        use_noise = (not use_traj) and (
            x0_bank is None or (args.xt_mode == "mix" and step % 10 < 3)
        )
        if use_traj:
            latents, t_state, state_scale = traj_bank[traj_rng.randrange(len(traj_bank))]
            latents = latents.to(device=device, dtype=torch.float32)
            if latents.shape[0] != batch:
                latents = latents.expand(batch, -1, -1)
            timestep = torch.full(
                (batch,), float(t_state), device=device, dtype=torch.float32
            )
        elif use_noise:
            hi = 0.35 if x0_bank is not None else 0.95
            timestep = torch.empty(batch, device=device, dtype=torch.float32).uniform_(
                0.02, hi, generator=data_rng
            )
            latents = noise
        else:
            timestep = torch.empty(batch, device=device, dtype=torch.float32).uniform_(
                0.02, 0.98, generator=data_rng
            )
            x0 = x0_bank[step % len(x0_bank)].to(device=device, dtype=torch.float32)
            if x0.shape[0] != batch:
                x0 = x0.expand(batch, -1, -1)
            t_mix = timestep.view(batch, 1, 1)
            latents = (1.0 - t_mix) * noise + t_mix * x0

        def _cond(name: str) -> torch.Tensor:
            tensor = conds[name].to(device=device, dtype=torch.float32)
            if tensor.shape[0] != batch:
                tensor = tensor.expand(batch, -1, -1).contiguous()
            return tensor

        cond_pos = _cond("positive")
        cond_neg = _cond("negative")
        cond_neu = _cond("neutral")
        cond_tgt = _cond("target")
        if amp_enabled and not args.teacher_fp32:
            latents = latents.to(amp_dtype)
            timestep = timestep.to(amp_dtype)
            cond_pos = cond_pos.to(amp_dtype)
            cond_neg = cond_neg.to(amp_dtype)
            cond_neu = cond_neu.to(amp_dtype)
            cond_tgt = cond_tgt.to(amp_dtype)

        _set_lora_multiplier(network, 0.0)
        with torch.no_grad():
            # vel_pos - vel_neg is a 2-5% difference of large numbers. In bf16 that
            # axis is only good to cos ~0.89-0.95, which is label noise the slider
            # is then fit against: scoring the same checkpoints with an fp32 teacher
            # reads ~0.04 higher cos across the board.
            if amp_enabled and not args.teacher_fp32:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    vel_pos = transformer(latents, timestep, cond_pos, return_dict=False)[0]
                    vel_neg = transformer(latents, timestep, cond_neg, return_dict=False)[0]
                    vel_neu = transformer(latents, timestep, cond_neu, return_dict=False)[0]
            else:
                vel_pos = transformer(latents, timestep, cond_pos, return_dict=False)[0]
                vel_neg = transformer(latents, timestep, cond_neg, return_dict=False)[0]
                vel_neu = transformer(latents, timestep, cond_neu, return_dict=False)[0]

        def _lora_forward(slider: float) -> torch.Tensor:
            network.set_lora_slider(slider)
            with network:
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        return transformer(latents, timestep, cond_tgt, return_dict=False)[0]
                return transformer(latents, timestep, cond_tgt, return_dict=False)[0]

        if prompt.action not in ("enhance", "erase"):
            raise ValueError(f"action must be enhance or erase, got {prompt.action!r}")
        sign = 1.0 if prompt.action == "enhance" else -1.0
        guidance_scale = float(prompt.guidance_scale)
        axis = sign * guidance_scale * (vel_pos.float() - vel_neg.float())

        def _target_delta(direction: float) -> torch.Tensor:
            """The delta the slider should add at `direction`.

            axis: direction * g * (vel_pos - vel_neg). This assumes neutral sits
            on the pos-neg line. It does not: at 4s the neutral caption renders
            quieter than BOTH poles, so the - side of the axis points somewhere
            neither caption occupies, and the teacher composed at net -1 renders
            rms +197% against the - caption's +14%.

            pole: direction * g * (nearer pole - vel_neu), i.e. each side aims at
            its own caption's actual displacement from neutral. Composing that
            live at net 1.0 reproduces the caption swap almost exactly
            (-15.4% rms / +110.8% centroid vs ground truth -15.5% / +110.6%),
            where the axis at the same net strength gives -2.0% / +85.4%.
            """
            if args.target_mode != "pole":
                return direction * axis
            toward_pos = (direction * sign) >= 0.0
            pole = vel_pos if toward_pos else vel_neg
            return abs(direction) * guidance_scale * (pole.float() - vel_neu.float())
        uncond_term = None
        if args.uncond_weight > 0.0:
            # CFG runs the uncond branch on a zeros condition, and a merged LoRA
            # cannot be excluded from it. Note the delta does NOT cancel there:
            # with v = v_u + w(v_c - v_u), adding d to both branches nets 1.0*d
            # while adding it to the conditional branch alone nets w*d (w=1.7).
            # So a merged slider runs at 1/1.7 of the strength its cond-branch
            # calibration implies. This penalty makes the adapter inert on zeros,
            # which recovers the missing 1.7x; it is NOT the cause of the level
            # collapse (that is open-loop drift -- see MUSIC3.md).
            cond_zero = torch.zeros_like(cond_tgt)
            with torch.no_grad():
                if amp_enabled and not args.teacher_fp32:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        vel_zero = transformer(latents, timestep, cond_zero, return_dict=False)[0]
                else:
                    vel_zero = transformer(latents, timestep, cond_zero, return_dict=False)[0]
            network.set_lora_slider(1.0)
            with network:
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        vel_zero_lora = transformer(latents, timestep, cond_zero, return_dict=False)[0]
                else:
                    vel_zero_lora = transformer(latents, timestep, cond_zero, return_dict=False)[0]
            uncond_term = args.uncond_weight * (
                (vel_zero_lora.float() - vel_zero.float()).pow(2).mean()
                / axis.pow(2).mean().clamp_min(1e-8)
            )

        # A trajectory state belongs to the slider setting that produced it, so
        # that setting is the one whose restoring force is learned there: forward
        # at the state's own scale, target scaled to match. (Training slider 1.0
        # on a state visited at 0.33 supervises a strength the deployed slider
        # never uses at that state, and silently contradicted --traj_scales'
        # "each state is trained at its own setting".)
        primary = float(state_scale) if use_traj and abs(state_scale) > 1e-6 else 1.0
        # A per-step gain bias introduced early in the solve (low t) has the most
        # remaining steps to compound through; ~2*(1-t) prices that. E[1-t] over
        # the training draw is ~0.5, so the factor keeps the mean weight near 1.
        gain_tw = 2.0 * (1.0 - float(timestep.float().mean())) if args.gain_tweight else 1.0
        vel_primary = _lora_forward(primary)
        loss = _slider_loss(
            vel_primary.float(), vel_neu.float(), _target_delta(primary), args.loss,
            args.mag_weight, args.gain_penalty, args.gain_mode, gain_tw,
        )
        vel_plus = vel_primary if primary > 0 else None
        vel_minus = vel_primary if primary < 0 else None
        if args.bidirectional:
            # -s is not the linear inverse of +s through 36 layers; train it explicitly.
            other = -primary
            vel_other = _lora_forward(other)
            loss = 0.5 * (
                loss
                + _slider_loss(
                    vel_other.float(), vel_neu.float(), _target_delta(other), args.loss,
                    args.mag_weight, args.gain_penalty, args.gain_mode, gain_tw,
                )
            )
            if other > 0:
                vel_plus = vel_other
            else:
                vel_minus = vel_other
        if uncond_term is not None:
            loss = loss + uncond_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_value_ clamps each element to +-1, so an outlier step (the
        # shipped triphop log peaks at 250x the median loss) degenerates into a
        # sign-like update. Norm clipping keeps the direction of those steps.
        if args.grad_clip == "value":
            torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())
        with torch.no_grad():
            def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
                return torch.nn.functional.cosine_similarity(
                    a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
                ).item()

            true_dir = (vel_pos.float() - vel_neg.float())
            delta_plus = None if vel_plus is None else vel_plus.float() - vel_neu.float()
            cos_pos = float("nan") if delta_plus is None else _cos(delta_plus, true_dir)
            cos_neg = None
            collapse = None
            if vel_minus is not None:
                delta_minus = vel_minus.float() - vel_neu.float()
                cos_neg = _cos(delta_minus, -true_dir)
                if delta_plus is not None:
                    collapse = _cos(delta_plus, delta_minus)
        postfix = {"loss": f"{last_loss:.5f}", "cos": f"{cos_pos:.3f}"}
        if collapse is not None:
            postfix["col"] = f"{collapse:.2f}"
        progress.set_postfix(**postfix)
        print(
            f"step {step + 1}/{steps} loss={last_loss:.6f} cos={cos_pos:.4f}"
            + (
                f" cos_neg={cos_neg:.4f} collapse={collapse:.4f}"
                if cos_neg is not None
                else ""
            ),
            flush=True,
        )
        record = {"step": step + 1, "loss": last_loss, "cos": cos_pos}
        if cos_neg is not None:
            record["cos_neg"] = cos_neg
            record["collapse"] = collapse
        if probe is not None and ((step + 1) % eval_every == 0 or (step + 1) == steps):
            last_eval = evaluate_probe(transformer, network, probe, amp_enabled)
            record["eval"] = last_eval
            # A pole run is aimed at its own edge, not the pos-neg axis; select
            # its best checkpoint on the metric it optimizes.
            best_key = "cos_edge" if (args.target_mode == "pole" and "cos_edge" in last_eval) else "cos"
            if best_eval is None or last_eval[best_key] > best_eval.get(best_key, float("-inf")):
                best_eval, best_step = last_eval, step + 1
                best_path = save_dir / f"{stem}_best.safetensors"
                network.save_weights(str(best_path), dtype=torch.float32)
                record["saved_best"] = True
            edge_txt = (
                f" cos_edge={last_eval['cos_edge']:.4f} cos_edge_neg={last_eval['cos_edge_neg']:.4f}"
                if "cos_edge" in last_eval
                else ""
            )
            print(
                f"eval step {step + 1}/{steps} cos={last_eval['cos']:.4f} "
                f"cos_neg={last_eval['cos_neg']:.4f} collapse={last_eval['collapse']:.4f} "
                f"mag={last_eval['mag']:.3f} proj_abs={last_eval['proj_abs']:.4f}{edge_txt}",
                flush=True,
            )
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        if (step + 1) == mid_step:
            mid_path = save_dir / f"{stem}_step{step + 1}.safetensors"
            network.save_weights(str(mid_path), dtype=torch.float32)
            print(f"saved mid checkpoint {mid_path}", flush=True)

    log_handle.close()
    weights_path = save_dir / f"{stem}_last.safetensors"
    network.save_weights(str(weights_path), dtype=torch.float32)
    meta = {
        "schema": 3,
        "name": name,
        "kind": "transformer",
        "prefix": "lora_unet",
        "rank": rank,
        "alpha": alpha,
        "steps": steps,
        "duration": duration,
        "dummy": bool(args.dummy),
        "target_replace": target_replace,
        "lm_weights": args.lm_weights,
        "delimiter": "-",
        "train_method": "full",
        "targets": args.targets,
        "xt_mode": args.xt_mode,
        "bidirectional": bool(args.bidirectional),
        "cond_seeds": cond_seeds,
        "x0_per_row": int(args.x0_per_row),
        "target_mode": args.target_mode,
        "traj_frac": float(args.traj_frac),
        "traj_refresh": int(args.traj_refresh),
        "traj_steps": int(args.traj_steps),
        "traj_scales": str(args.traj_scales),
        "plus_label": prompts_meta.plus_label,
        "minus_label": prompts_meta.minus_label,
        "recommended_range": prompts_meta.recommended_range,
        "loss": last_loss,
        "loss_kind": args.loss,
        "gain_penalty": float(args.gain_penalty),
        "gain_mode": args.gain_mode,
        "gain_tweight": bool(args.gain_tweight),
        "mag_weight": float(args.mag_weight),
        "uncond_weight": float(args.uncond_weight),
        "grad_clip": args.grad_clip,
        # Reproducibility: every random source this run consumed. `seed` drives
        # LoRA init (global RNG), the (t, eps) data stream, x0 anchors and
        # rollouts; cond_seeds the AR/condition encoding; eval_seed the probe.
        # A run that cannot state its seeds cannot be paired against another.
        "seed": int(args.seed),
        "lr": lr,
        "lr_scheduler": lr_name,
        "prompts_file": str(prompts_path),
        "cache_dir": str(cache_dir),
        "eval": last_eval,
        "eval_best": best_eval,
        "eval_best_step": best_step,
        "eval_timesteps": list(EVAL_TIMESTEPS),
        "prompt_geometry": (probe.geometry if probe is not None else None),
        "eval_seed": int(args.eval_seed),
        "eval_holdout": bool(args.eval_holdout),
        "prompts": [asdict(prompt) for prompt in prompts],
        "weights": str(weights_path),
    }
    meta_path = save_dir / f"{stem}_last.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {weights_path}")
    print(f"saved {meta_path}")
    if not args.dummy and not args.no_calibrate and entries:
        from conceptmod.textsliders.calibrate_scale import (
            calibrate_network,
            merge_calibration,
            print_table,
        )

        print("calibrating unit_scale on trained weights", flush=True)
        result = calibrate_network(
            transformer,
            network,
            entries[0][1],
            guidance=float(entries[0][0].guidance_scale),
            device=device,
            seed=seed,
        )
        print_table(result)
        meta = merge_calibration(meta, result)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"updated {meta_path} unit_scale={result['unit_scale']:g}", flush=True)
    return weights_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="energy")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="matched to the default rank 8. Larger ranks want less: the "
                        "8-128 ladder is flat only when lr is matched to rank")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_ar", action="store_true", help="use existing condition cache only")
    parser.add_argument(
        "--targets",
        choices=["attn", "full"],
        default="full",
        help="attn=MiniMaxMusic3Attention only; full also LoRAs proj_in/FF/convs",
    )
    parser.add_argument(
        "--lm_weights",
        type=str,
        default=None,
        help="if set, build pos/neg conditions by running AR with this LM slider ±1 on the neutral caption",
    )
    parser.add_argument("--guidance", type=float, default=None, help="override prompt guidance_scale")
    parser.add_argument(
        "--xt_mode",
        choices=["anchor", "mix", "noise"],
        default="anchor",
        help="anchor=x_t from generated clean latents; mix=30%% pure-noise steps; noise=legacy pure randn",
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also train the -1 direction against vel_neu - g*(vel_pos - vel_neg)",
    )
    parser.add_argument(
        "--cond_seeds",
        type=str,
        default=None,
        help="comma-separated AR seeds; each prompt row gets one condition set per seed (default: --seed)",
    )
    parser.add_argument("--x0_per_row", type=int, default=8,
                        help="clean-latent anchors per condition set. 8 is the measured knee: "
                        "2->8 is worth +0.040 held-out cos, 8->16 is within noise")
    parser.add_argument(
        "--x0_source",
        choices=["neutral", "positive", "negative"],
        default="neutral",
        help="which caption's condition generates the x0 anchors. Non-default is a "
        "DIAGNOSTIC (probe anchor-dependence); training should stay on neutral",
    )
    parser.add_argument("--x0_steps", type=int, default=30, help="Euler steps when generating x0 anchors")
    parser.add_argument(
        "--target_mode",
        choices=["axis", "pole"],
        default="axis",
        help="axis=v_neu + s*g*(v_pos - v_neg) (assumes neutral is on the pos-neg line, "
        "which it is not); pole=each side aims at its own caption's displacement from "
        "neutral, which is what the caption swap actually is",
    )
    parser.add_argument(
        "--traj_frac",
        type=float,
        default=0.0,
        help="fraction of steps whose x_t comes from a rollout of the CURRENT adapter "
        "instead of the base-model anchor. Closes the open-loop gap: the adapter is "
        "otherwise only ever trained (and scored) on states the base model would have "
        "reached without it. 0 disables",
    )
    parser.add_argument(
        "--traj_refresh",
        type=int,
        default=50,
        help="training steps between rollouts; the bank goes stale as the adapter moves",
    )
    parser.add_argument("--traj_steps", type=int, default=20, help="Euler steps per rollout")
    parser.add_argument(
        "--traj_scales",
        type=str,
        default="1,-1",
        help="slider settings to roll out at; each state is trained at its own setting",
    )
    parser.add_argument(
        "--uncond_weight",
        type=float,
        default=0.0,
        help="penalise the LoRA's velocity delta on the all-zeros (CFG unconditional) "
        "condition, normalised by the axis energy. A merged LoRA perturbs that branch "
        "too, which measured -53 points of rms and half the brightness on a render; "
        "0 disables",
    )
    parser.add_argument(
        "--gain_penalty",
        type=float,
        default=0.0,
        help="penalise the delta's component along vel_neu (a pure rescaling of "
        "the velocity). That component is only ~16%% of the concept axis but is "
        "coherent across denoise steps, so it compounds into the render while "
        "shape components cancel; 0 disables. --gain_mode picks the target the "
        "component is pushed toward",
    )
    parser.add_argument(
        "--gain_mode",
        choices=["penalty", "match"],
        default="penalty",
        help="penalty=push the vel_neu-parallel (pure gain) component of the delta "
        "toward zero (legacy --gain_penalty; measured inert on the dust and "
        "trip-hop pairs). match=push it toward the teacher's OWN gain component "
        "(axis . unit_neu) — the teacher carries some level change, so zero is "
        "the wrong target",
    )
    parser.add_argument(
        "--gain_tweight",
        action="store_true",
        help="weight the gain term by ~2*(1-t): a gain bias at low t (early in "
        "the solve) has the most remaining steps to compound through",
    )
    parser.add_argument(
        "--teacher_fp32",
        action="store_true",
        help="compute vel_pos/vel_neg/vel_neu in fp32 (the student still trains under "
        "bf16 autocast, as inference runs it). Removes ~0.04 of target noise at the "
        "cost of ~2x model memory and slower teacher forwards.",
    )
    parser.add_argument(
        "--lr_scheduler",
        choices=["cosine", "constant"],
        default=None,
        help="default: the config's value (cosine). Cosine anneals to eta_min 1e-6 "
        "by --steps, which measurably costs cos: an otherwise identical 1000-step "
        "run reads 0.8089 at step 500 where the 500-step run finishes at 0.7928",
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "nmse", "cos", "nmse_ortho"],
        default="nmse",
        help="mse=legacy (target energy varies ~200x across t, so low-t steps "
        "dominate); nmse=per-step energy-normalized; cos=optimize the reported "
        "axis metric plus a magnitude term; nmse_ortho=fit only the components "
        "orthogonal to vel_neu, with the pure-gain component supervised solely "
        "by the gain term (requires --gain_penalty>0 --gain_mode match)",
    )
    parser.add_argument(
        "--mag_weight",
        type=float,
        default=0.25,
        help="--loss cos only: weight on (||delta||/||axis|| - 1)^2",
    )
    parser.add_argument(
        "--grad_clip",
        choices=["norm", "value"],
        default="norm",
        help="value=legacy elementwise clamp to +-1 (turns outlier steps into "
        "sign updates); norm=global norm clip, keeps their direction",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=50,
        help="steps between fixed-probe evals (0 disables). The probe is pinned "
        "to --eval_seed and a fixed t grid so runs are comparable",
    )
    parser.add_argument("--eval_seed", type=int, default=1234, help="fixed eval probe seed")
    parser.add_argument(
        "--eval_holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score the probe on a clean latent the run never trains on. Off by "
        "default so numbers stay comparable with runs scored on the shared bank; "
        "required to compare runs that differ in --x0_per_row, since the shared "
        "bank favours runs that concentrate on those anchors",
    )
    parser.add_argument("--plus_label", type=str, default=None, help="override sidecar plus_label")
    parser.add_argument("--minus_label", type=str, default=None, help="override sidecar minus_label")
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="skip writing unit_scale into the sidecar after the last step",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="random conditions + tiny transformer, 2 steps, never loads the 8B LM",
    )
    return parser.parse_args(argv)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
