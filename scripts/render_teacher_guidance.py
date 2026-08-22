#!/usr/bin/env python3
"""Render the slider's TEACHER directly, with no LoRA and no fit error.

A slider is trained to distil, at every denoise step,

    v_target = v_neutral + s * g * (v_pos - v_neg)

This renders exactly that object by composing the velocity live inside the
sampler: three transformer forwards per step instead of one. It is the ceiling
the LoRA is chasing, so it partitions the blame for a disappointing render:

  * if the teacher render is also wrong, the OBJECTIVE is wrong and no training
    recipe on this target can fix it;
  * if the teacher render is right, the objective is fine and the gap is fit,
    capacity or calibration.

It also gives the honest yardstick for evaluating a trained slider: the caption
swap is not reachable by a transformer LoRA over a fixed AR plan (the poles here
differ in BPM and arrangement, which the plan owns), but the teacher render is.

Modes:
  axis      delta = s*g*(v_pos - v_neg)   -- what the trainer currently targets
  posedge   delta = s*g*(v_pole - v_neu)  -- displacement of --pole from neutral
  orth      axis, with the v_neu-parallel component removed
  parcap    axis, with the v_neu-parallel component held at 1x instead of g

NET-STRENGTH BOOKKEEPING: composing on the conditional branch only is amplified
by CFG, so the delta that reaches the velocity is 1.7*s*g; with --both_branches
it is s*g. The printed "post-CFG net" is the honest number -- the 2026-08-21
posedge/negpole rows were mislabeled by exactly this 1.7x.

CAUTION on posedge at s*g=1 cond-only: v_u + w*(v_neu_c + 1.0*(v_pole_c -
v_neu_c) - v_u) == v_u + w*(v_pole_c - v_u). It is algebraically THE caption
swap, same trajectory step for step, so matching REF_pole there verifies the
plumbing, not the objective. The informative settings are the fractional ones.

    python scripts/render_teacher_guidance.py --mode axis --scales 0.333,1 \
      --out_dir eval/listen/teacher-4s
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import soundfile as sf
import torch

_ROOT = Path("/ml2/music/sliders-conceptmod")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptmod.textsliders.infer_music3 import _load_pipeline, _to_wav_array  # noqa: E402
from conceptmod.textsliders.train_lora_music3 import (  # noqa: E402
    build_conditions,
    load_prompts,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts_file", type=Path,
                    default=_ROOT / "conceptmod/textsliders/data/prompts-triphop-v3-single.yaml")
    ap.add_argument("--cache_dir", type=Path, default=_ROOT / "cache/triphop-v3")
    ap.add_argument("--encode_ar", action="store_true",
                    help="run the AR/LM stage to populate an empty --cache_dir. Needed for a duration\n"
                    "the cache was not built at: conditions are per-duration, and without this the\n"
                    "script fails with '--skip_ar set but missing LM/AR cache'.")
    ap.add_argument("--model_dir", type=Path, default=Path("/ml2/music/models/MiniMax-Music3"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["axis", "posedge", "orth", "parcap"], default="axis")
    ap.add_argument("--pole", choices=["pos", "neg"], default="pos",
                    help="which pole posedge displaces toward. The axis modes assume neutral "
                    "lies on the pos-neg line; it does not (neutral rms sits below BOTH poles "
                    "here), which is why the axis minus side renders nothing like the - caption. "
                    "posedge --pole neg is the honest - side target.")
    ap.add_argument("--scales", default="-1,-0.333,0,0.333,1")
    ap.add_argument("--duration", type=float, default=4.0,
                    help="keep this equal to the cached conditions' duration")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--latent_seed", type=int, default=None,
                    help="denoise-noise seed (default: --seed). Lets the crater be checked "
                    "on a different trajectory without re-encoding AR conditions")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=None, help="override the prompt's g")
    ap.add_argument("--both_branches", action="store_true",
                    help="apply the delta to the unconditional CFG branch as well. A merged LoRA "
                    "necessarily does this; composing on the conditional branch only does not.")
    ap.add_argument("--ref", action="store_true",
                    help="also render the three captions plain (no composition), for ground truth "
                    "at this duration -- levels are not comparable across durations")
    args = ap.parse_args(argv)

    device = f"cuda:{int(args.device)}"
    prompts, _ = load_prompts(args.prompts_file)
    prompt = prompts[0]
    g = float(args.guidance if args.guidance is not None else prompt.guidance_scale)

    entries = build_conditions(
        prompts=[prompt], cache_dir=args.cache_dir, duration=args.duration,
        seeds=[args.seed], device=torch.device(device), model_dir=args.model_dir,
        skip_ar=not args.encode_ar, dummy=False,
    )
    conds = entries[0][1]
    pipe = _load_pipeline(args.model_dir, device)

    dtype = next(pipe.transformer.parameters()).dtype
    cond_pos = conds["positive"].to(device=device, dtype=dtype)
    cond_neg = conds["negative"].to(device=device, dtype=dtype)
    cond_neu = conds["neutral"].to(device=device, dtype=dtype)

    original_forward = pipe.transformer.forward
    state = {"scale": 0.0, "calls": 0}

    def composed_forward(hidden_states, timestep, encoder_hidden_states, return_dict=True):
        if encoder_hidden_states.abs().any() and encoder_hidden_states.shape[1] < cond_pos.shape[1]:
            # Chunked inference: the incoming cond is one WINDOW's slice, and
            # composing against cached conds aligned from position 0 corrupts
            # every window after the first. The 12s cells rendered before this
            # guard existed are invalid for exactly that reason.
            raise SystemExit(
                f"chunked denoise detected (window {encoder_hidden_states.shape[1]} < "
                f"cached cond {cond_pos.shape[1]}): composition would misalign windows "
                f">1. Use a duration that fits one window, or extend this script with "
                f"per-window offsets."
            )
        base = original_forward(hidden_states, timestep, encoder_hidden_states, return_dict=False)[0]
        s = state["scale"]
        # Compose on the CONDITIONAL branch only by default. With v = v_u + w(v_c - v_u),
        # a delta on the conditional branch alone nets w*d (w=1.7) while the same delta
        # on both nets 1.0*d -- it does NOT cancel, it just loses the 1.7x. A merged
        # LoRA necessarily takes the both-branches case; --both_branches renders that.
        if s == 0.0 or (not args.both_branches and not encoder_hidden_states.abs().any()):
            state["skipped"] = state.get("skipped", 0) + 1
            return (base,)
        n = encoder_hidden_states.shape[1]
        if not encoder_hidden_states.abs().any():
            n = min(n, cond_pos.shape[1])

        def c(t: torch.Tensor) -> torch.Tensor:
            t = t[:, :n]
            if t.shape[0] != hidden_states.shape[0]:
                t = t.expand(hidden_states.shape[0], -1, -1)
            return t

        with torch.no_grad():
            v_pos = original_forward(hidden_states, timestep, c(cond_pos), return_dict=False)[0]
            if args.mode == "posedge":
                v_ref = original_forward(hidden_states, timestep, c(cond_neu), return_dict=False)[0]
                v_pole = v_pos
                if args.pole == "neg":
                    v_pole = original_forward(
                        hidden_states, timestep, c(cond_neg), return_dict=False
                    )[0]
                delta = v_pole.float() - v_ref.float()
            else:
                v_neg = original_forward(hidden_states, timestep, c(cond_neg), return_dict=False)[0]
                delta = v_pos.float() - v_neg.float()
            if args.mode in ("orth", "parcap"):
                v_neu = original_forward(hidden_states, timestep, c(cond_neu), return_dict=False)[0]
                u = v_neu.float().flatten()
                u = u / u.norm().clamp_min(1e-8)
                par = (delta.flatten() @ u)
                par_vec = (par * u).view_as(delta)
                if args.mode == "orth":
                    delta = delta - par_vec
                else:  # parcap: orthogonal part amplified by g, parallel left at 1x
                    delta = (delta - par_vec) + par_vec / max(g, 1e-6)
        out = base.float() + s * g * delta
        out = out.to(base.dtype)
        state["calls"] += 1
        return (out,) if return_dict is False else (out,)

    pipe.transformer.forward = composed_forward
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sr = int(pipe.sampling_rate)

    latent_seed = int(args.seed if args.latent_seed is None else args.latent_seed)
    seed_tag = "" if latent_seed == int(args.seed) else f"_ls{latent_seed}"
    for scale in [float(x) for x in args.scales.split(",") if x.strip()]:
        state["scale"], state["calls"], state["skipped"] = scale, 0, 0
        gen = torch.Generator(device).manual_seed(latent_seed)
        cfg_mult = 1.0 if args.both_branches else 1.7
        print(
            f"rendering mode={args.mode} s={scale:g} g*s={g*scale:g} "
            f"post-CFG net={cfg_mult * g * scale:g} "
            f"({'both branches' if args.both_branches else 'cond only, x1.7 through CFG'})",
            flush=True,
        )
        audio = pipe(prompt=prompt.neutral, lyrics=prompt.lyrics,
                     audio_duration=float(args.duration), generator=gen, output="audios")[0]
        wav = _to_wav_array(audio)
        tag = "zero" if scale == 0 else (f"plus{scale:g}" if scale > 0 else f"minus{abs(scale):g}")
        suffix = ("_both" if args.both_branches else "") + (
            "_negpole" if (args.mode == "posedge" and args.pole == "neg") else ""
        )
        dest = args.out_dir / f"teacher_{args.mode}{suffix}{seed_tag}_{tag}.wav"
        sf.write(str(dest), wav, sr)
        # Mono-downmix std, matching score_render_curve's feats(); the raw
        # (frames, channels) array's std reads high (decorrelated channels) and
        # once mislabeled the net-1.7 teacher cell by +28 rms points.
        arr = np.asarray(wav, dtype=np.float64)
        rms = float((arr.mean(axis=1) if arr.ndim == 2 else arr).std())
        print(f"  wrote {dest.name}  rms={rms:.4f}  composed={state['calls']} "
              f"uncond_passthrough={state.get('skipped', 0)}", flush=True)

    if args.ref:
        pipe.transformer.forward = original_forward
        for label, text in (("neu", prompt.neutral), ("pos", prompt.positive), ("neg", prompt.negative)):
            gen = torch.Generator(device).manual_seed(latent_seed)
            print(f"rendering REF {label}", flush=True)
            audio = pipe(prompt=text, lyrics=prompt.lyrics, audio_duration=float(args.duration),
                         generator=gen, output="audios")[0]
            wav = _to_wav_array(audio)
            dest = args.out_dir / f"REF_{label}{seed_tag}.wav"
            sf.write(str(dest), wav, sr)
            print(f"  wrote {dest.name}  rms={float(np.asarray(wav, dtype=np.float64).std()):.4f}", flush=True)

    pipe.transformer.forward = original_forward
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
