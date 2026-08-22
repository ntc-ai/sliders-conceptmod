#!/usr/bin/env python3
"""Apply a trained MiniMax Music 3 energy slider and write short wavs."""

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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import soundfile as sf
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork

DEFAULT_MODEL_DIR = Path("/ml2/music/models/MiniMax-Music3")
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-music3.yaml"
DEFAULT_OUT_DIR = Path("/ml2/music/sliders-conceptmod/models/energy-slider/infer")
TARGET_REPLACE = ["MiniMaxMusic3Attention"]
_LOAD_ORDER = (
    "tokenizer",
    "scheduler",
    "vocoder",
    "condition_encoder",
    "rvq_depth_decoder",
    "transformer",
    "language_model",
)
DEFAULT_LYRICS = "[verse]\nI keep walking through the night\nlooking for a little light\n[chorus]\nI will not go quietly"
DEFAULT_PROMPT = (
    "Mid-energy pop song, moderate tempo, balanced drums and electric guitar, "
    "clean sung vocals, radio mix."
)


def _sidecar_meta(weights: Path) -> dict:
    for candidate in (weights.with_suffix(".json"), Path(str(weights) + ".json")):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _load_default_prompt(prompts_file: Path) -> tuple[str, str]:
    if not prompts_file.exists():
        return DEFAULT_PROMPT, DEFAULT_LYRICS
    raw = yaml.safe_load(prompts_file.read_text(encoding="utf-8"))
    if not raw:
        return DEFAULT_PROMPT, DEFAULT_LYRICS
    item = raw[0]
    return str(item.get("target") or DEFAULT_PROMPT), str(item.get("lyrics") or DEFAULT_LYRICS)


def _load_pipeline(model_dir: Path, device: str):
    from diffusers import ModularPipeline

    if not (model_dir / "modular_model_index.json").exists():
        raise FileNotFoundError(f"Model not found at {model_dir}")

    idx = int(device.split(":", 1)[1])
    free, _total = torch.cuda.mem_get_info(idx)
    free_gib = free / 1024**3
    local_kwargs = {
        "pretrained_model_name_or_path": str(model_dir),
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    if free_gib < 28.0:
        raise RuntimeError(
            f"{device} has {free_gib:.1f} GiB free; listen/infer needs ~28 GiB on GPU. "
            "Refusing CPU offload. Pick an empty GPU: this script defaults "
            "CUDA_VISIBLE_DEVICES to \"0\" unless the caller already set it, and --device "
            "indexes the visible devices, so run with e.g. CUDA_VISIBLE_DEVICES=1 in the "
            "environment to use physical GPU 1."
        )
    pipe = ModularPipeline.from_pretrained(str(model_dir), local_files_only=True)

    for name in _LOAD_ORDER:
        print(f"loading {name}")
        pipe.load_components(names=name, **local_kwargs)

    pipe.to(device)
    return pipe


def _to_wav_array(audio) -> np.ndarray:
    if torch.is_tensor(audio):
        arr = audio.detach().T.float().cpu().numpy()
    else:
        arr = np.asarray(audio)
        if arr.ndim > 1:
            arr = arr.T
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def _parse_scales(raw: str) -> list[float]:
    scales = [float(part.strip()) for part in raw.split(",") if part.strip() != ""]
    if not scales:
        raise ValueError("no scales given")
    return scales


def _listen_name(scale: float) -> str:
    labels = {
        -2.0: "01_energy_very_quiet_scale-2",
        -1.0: "02_energy_quiet_scale-1",
        0.0: "03_energy_neutral_scale0",
        1.0: "04_energy_loud_scale+1",
        2.0: "05_energy_very_loud_scale+2",
    }
    if scale in labels:
        return f"{labels[scale]}.wav"
    sign = "+" if scale > 0 else ""
    return f"energy_scale{sign}{scale:g}.wav"


def infer(args: argparse.Namespace) -> list[Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax Music 3 inference requires CUDA")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    meta = _sidecar_meta(weights)
    rank = int(args.rank if args.rank is not None else meta.get("rank", 4))
    alpha = float(args.alpha if args.alpha is not None else meta.get("alpha", 1.0))
    prompt, lyrics = args.prompt, args.lyrics
    if not prompt or not lyrics:
        file_prompt, file_lyrics = _load_default_prompt(Path(args.prompts_file))
        prompt = prompt or file_prompt
        lyrics = lyrics or file_lyrics

    device = f"cuda:{int(args.device)}"
    model_dir = Path(args.model_dir)
    pipe = _load_pipeline(model_dir, device)

    network = LoRANetwork(
        pipe.transformer,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=TARGET_REPLACE,
        train_method="full",
        delimiter="-",
    )
    network.to(device)
    state = {}
    try:
        from safetensors.torch import load_file

        state = load_file(str(weights), device="cpu")
    except Exception:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = network.load_state_dict(state, strict=False)
    if len(network.unet_loras) == 0:
        raise RuntimeError(f"LoRA wrapped 0 modules for {TARGET_REPLACE}")
    if missing:
        raise RuntimeError(f"LoRA load missed {len(missing)} keys (first={missing[0]})")
    if unexpected:
        print(f"unexpected LoRA keys: {len(unexpected)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = _parse_scales(args.scales)
    written: list[Path] = []
    sample_rate = int(pipe.sampling_rate)
    seed = int(args.seed)

    for scale in scales:
        network.set_lora_slider(float(scale))
        generator = torch.Generator(device).manual_seed(seed)
        print(f"generating scale={scale} duration={args.duration}")
        with network:
            audio = pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=float(args.duration),
                generator=generator,
                output="audios",
            )[0]
        wav = _to_wav_array(audio)
        dest = out_dir / _listen_name(scale)
        sf.write(str(dest), wav, sample_rate)
        written.append(dest)
        print(f"wrote {dest}")
    return written


def _glue_negative_option(argv: list[str], flag: str) -> list[str]:
    """argparse treats values like -2,-1 as extra options; attach them with '='."""
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    raw = _glue_negative_option(raw, "--scales")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--scales", type=str, default="-2,-1,0,1,2")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--lyrics", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(raw)


def main() -> None:
    infer(parse_args())


if __name__ == "__main__":
    main()
