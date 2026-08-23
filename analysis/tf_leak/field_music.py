"""2-D teacher from *real* catalog wording: energy (loud/quiet) × tempo (BPM).

Coordinates are read off the caption text. Energy is a small loudness
lexicon taken from the distinctive tokens of ``prompts-energy.yaml`` and
``prompts-cand-energy-v1.yaml`` (no tempo words). Tempo is the numeric
BPM, scaled so 110 is 0 and a ±70 swing is ±1, plus the tempo yaml's
own speed words. Genre names are ignored.

That is enough to see the shipped energy pair sit on the diagonal
(loud *and* BPM 168) while ``cand-energy`` (fixed BPM 110) sits on x.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from analysis.tf_leak.captions import extract_bpm, load_row, tokenize
from conceptmod.textsliders.slider_targets import (
    expand_attributes_music3,
    music3_axis_delta,
    music3_pole_delta,
    music3_slider_loss,
)


# Distinctive loudness words from the energy / cand-energy yamls.
# "aggressive" is the one adjective energy shares with distortion; it
# stays on energy because the shipped energy caption uses it as intensity.
ENERGY_POS = {
    "loud", "energy", "high", "shouted", "yelling", "pounding", "slammed",
    "dense", "aggressive", "powerful", "driving", "busy", "layered",
    "thick", "doubled",
}
ENERGY_NEG = {
    "quiet", "silent", "whispered", "calm", "sparse", "ambient", "lullaby",
    "gentle", "restrained", "breathy", "minimal", "solo",
}

# Distinctive speed words from prompts-tempo.yaml. BPM numbers are separate.
TEMPO_POS = {"fast", "frantic", "racing", "rushed", "speed", "hyper", "double-time"}
TEMPO_NEG = {"slow", "dragged", "molasses", "half-time", "spacious", "gaps"}

BPM_CENTER = 110.0
BPM_SCALE = 70.0  # 180 → +1, 40 → −1

E_ENERGY = torch.tensor([1.0, 0.0])
E_TEMPO = torch.tensor([0.0, 1.0])


def _count(tokens: list[str], vocab: set[str]) -> int:
    return sum(1 for tok in tokens if tok in vocab)


def energy_score(text: str) -> float:
    toks = tokenize(text)
    raw = float(_count(toks, ENERGY_POS) - _count(toks, ENERGY_NEG))
    return raw / 4.0


def tempo_score(text: str) -> float:
    toks = tokenize(text)
    words = float(_count(toks, TEMPO_POS) - _count(toks, TEMPO_NEG)) / 3.0
    bpm = extract_bpm(text)
    bpm_axis = 0.0 if bpm is None else (bpm - BPM_CENTER) / BPM_SCALE
    return bpm_axis + 0.25 * words


def embed_text(text: str) -> torch.Tensor:
    return torch.tensor([energy_score(text), tempo_score(text)], dtype=torch.float32)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
    )


def project(delta: torch.Tensor) -> dict[str, float]:
    ds = float(delta.flatten() @ E_ENERGY)
    da = float(delta.flatten() @ E_TEMPO)
    return {
        "proj_energy": ds,
        "proj_tempo": da,
        "cos_energy": cosine(delta, E_ENERGY),
        "cos_tempo": cosine(delta, E_TEMPO),
        "leak_ratio": da / (abs(ds) + 1e-8),
        "norm": float(delta.norm()),
    }


@dataclass(frozen=True)
class Caption:
    name: str
    text: str


class MusicField2D:
    """Flow-matching teacher toward the 2-D reading of a caption."""

    def embed(self, caption: Caption | str, t: float = 0.5) -> torch.Tensor:
        text = caption.text if isinstance(caption, Caption) else caption
        return embed_text(text)

    def velocity(self, x: torch.Tensor, caption: Caption | str, t: float = 0.5) -> torch.Tensor:
        mag = 1.55 - 1.15 * float(t)
        return mag * (self.embed(caption, t) - x)

    def train_points(self) -> list[torch.Tensor]:
        return [
            torch.tensor([fx, fy], dtype=torch.float32)
            for fx in (-0.6, 0.0, 0.6)
            for fy in (-0.6, 0.0, 0.6)
        ]

    def timesteps(self) -> list[float]:
        return [0.1, 0.3, 0.5, 0.7, 0.9]


@dataclass
class Residual:
    w_odd: torch.Tensor

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w_odd

    def snapshot(self) -> "Residual":
        return Residual(self.w_odd.detach().clone())


@dataclass
class Pair:
    target: Caption
    positive: Caption
    negative: Caption
    neutral: Caption
    action: str = "enhance"
    guidance: float = 3.0


def pair_from_catalog(name: str, *, attributes: bool = False) -> Pair:
    row = dict(load_row(name))
    if attributes:
        row["attributes"] = ["A man is singing.", "A woman is singing."]
        row = expand_attributes_music3(row)[0]
    guidance = float(row.get("guidance_scale") or row.get("guidance") or 3.0)
    return Pair(
        target=Caption("target", row.get("target") or row["neutral"]),
        positive=Caption("positive", row["positive"]),
        negative=Caption("negative", row["negative"]),
        neutral=Caption("neutral", row.get("neutral") or row["target"]),
        guidance=guidance,
    )


def train_music3(
    field: MusicField2D,
    pair: Pair,
    *,
    kind: str = "nmse",
    target_mode: str = "axis",
    bidirectional: bool = True,
    steps: int = 150,
    lr: float = 0.08,
    seed: int = 0,
    mag_weight: float = 0.25,
    gain_weight: float = 0.0,
) -> Residual:
    torch.manual_seed(seed)
    residual = Residual(torch.zeros(2, requires_grad=True))
    opt = torch.optim.Adam([residual.w_odd], lr=lr)
    xs = field.train_points()
    ts = field.timesteps()
    scales = (1.0, -1.0) if bidirectional else (1.0,)

    def _axis(x, t, direction: float) -> torch.Tensor:
        vel_pos = field.velocity(x, pair.positive, t)
        vel_neg = field.velocity(x, pair.negative, t)
        vel_neu = field.velocity(x, pair.neutral, t)
        if target_mode == "pole":
            return music3_pole_delta(
                direction, vel_pos, vel_neg, vel_neu, pair.guidance, pair.action
            )
        return music3_axis_delta(direction, vel_pos, vel_neg, pair.guidance, pair.action)

    for _ in range(steps):
        total = residual.w_odd.new_zeros(())
        n = 0
        for x in xs:
            for t in ts:
                vel_neu = field.velocity(x, pair.neutral, t)
                for scale in scales:
                    vel = field.velocity(x, pair.target, t) + residual.delta(scale)
                    axis = _axis(x, t, scale)
                    total = total + music3_slider_loss(
                        vel, vel_neu, axis, kind, mag_weight, gain_weight=gain_weight
                    )
                    n += 1
        loss = total / n
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def score_residual(residual: Residual) -> dict:
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    plus = project(d_plus)
    return {
        "delta_plus": [float(d_plus[0]), float(d_plus[1])],
        "delta_minus": [float(d_minus[0]), float(d_minus[1])],
        "cos_energy_plus": plus["cos_energy"],
        "cos_tempo_plus": plus["cos_tempo"],
        "leak_ratio": plus["leak_ratio"],
        "proj_energy_plus": plus["proj_energy"],
        "proj_tempo_plus": plus["proj_tempo"],
        "cos_plus_minus": cosine(d_plus, d_minus),
        "norm_plus": plus["norm"],
    }


def teacher_leak(pair: Pair) -> dict:
    """pos−neg of the live captions, no training. The thing the default loss fits."""
    axis = embed_text(pair.positive.text) - embed_text(pair.negative.text)
    return project(axis)
