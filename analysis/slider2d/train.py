"""Fit tiny residuals with the live slider losses on the 2-D field."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from analysis.slider2d.field import Field2D, Prompt, PROMPTS, cosine, project
from conceptmod.textsliders.slider_targets import (
    encoder_mse_loss,
    expand_attributes_music3,
    expand_attributes_sd,
    lm_hidden_targets,
    music3_axis_delta,
    music3_pole_delta,
    music3_slider_loss,
    sd_noise_target,
)


@dataclass
class Residual:
    """LoRA-like residual in R².

    ``odd``: Δ(s) = s · w  — a LoRA multiplier, used for SD / Music 3 TF.
    ``odd_even``: Δ(s) = s · w_odd + |s| · w_even — enough capacity for the
    LM/encoder collapse the live trainers measured (raw pos/neg share an
    even component that a multiplier cannot cancel).
    """

    kind: str
    w_odd: torch.Tensor
    w_even: torch.Tensor | None = None

    def delta(self, scale: float) -> torch.Tensor:
        out = float(scale) * self.w_odd
        if self.w_even is not None:
            out = out + abs(float(scale)) * self.w_even
        return out

    def parameters(self) -> list[torch.Tensor]:
        params = [self.w_odd]
        if self.w_even is not None:
            params.append(self.w_even)
        return params

    @classmethod
    def create(cls, kind: str) -> "Residual":
        if kind not in ("odd", "odd_even"):
            raise ValueError(kind)
        w_odd = torch.zeros(2, requires_grad=True)
        w_even = torch.zeros(2, requires_grad=True) if kind == "odd_even" else None
        return cls(kind, w_odd, w_even)

    def snapshot(self) -> "Residual":
        even = None if self.w_even is None else self.w_even.detach().clone()
        return Residual(self.kind, self.w_odd.detach().clone(), even)


@dataclass
class Pair:
    target: Prompt
    positive: Prompt
    negative: Prompt
    neutral: Prompt
    action: str = "enhance"
    guidance: float = 4.0


@dataclass
class MethodResult:
    name: str
    residual: Residual
    metrics: dict
    verdict: str
    reason: str
    family: str


def _prompt(name: str) -> Prompt:
    if name in PROMPTS:
        return PROMPTS[name]
    raise KeyError(f"unknown prompt {name!r}")


def base_row() -> dict:
    return {
        "target": "song",
        "positive": "energetic",
        "negative": "calm",
        "neutral": "song",
        "unconditional": "calm",
        "action": "enhance",
        "guidance_scale": 4.0,
    }


def pairs_from_rows(rows: list[dict], action: str, guidance: float) -> list[Pair]:
    out = []
    for row in rows:
        out.append(
            Pair(
                target=_prompt(row["target"]),
                positive=_prompt(row["positive"]),
                negative=_prompt(row.get("negative") or row.get("unconditional") or "calm"),
                neutral=_prompt(row["neutral"]),
                action=action,
                guidance=guidance,
            )
        )
    return out


def sd_pairs(with_attrs: bool, action: str = "enhance", guidance: float = 4.0) -> list[Pair]:
    attrs = ("male", "female") if with_attrs else ()
    rows = expand_attributes_sd(base_row(), attrs)
    return pairs_from_rows(rows, action, guidance)


def music3_pairs(with_attrs: bool, action: str = "enhance", guidance: float = 4.0) -> list[Pair]:
    row = base_row()
    if with_attrs:
        row["attributes"] = ["male", "female"]
    return pairs_from_rows(expand_attributes_music3(row), action, guidance)


def _optimize(residual: Residual, loss_fn, steps: int, lr: float, seed: int) -> Residual:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(residual.parameters(), lr=lr)
    for _ in range(steps):
        loss = loss_fn(residual)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def train_sd(
    field: Field2D,
    pairs: list[Pair],
    *,
    steps: int = 250,
    lr: float = 0.08,
    seed: int = 0,
    bidirectional: bool = False,
) -> Residual:
    """SD / XL noise-prediction slider. Live trainer fits only +1.

    ``train_lora.py`` (SD1) passes ``unconditional_latents`` into a signature
    that now expects ``negative_latents`` and omits ``negative`` in
    ``PromptEmbedsPair`` — that trainer is currently inconsistent with
    ``prompt_util``. Geometry here matches XL / SD3 / Flux / Cascade, and the
    SD1 *intent* (uncond as the opposite pole).
    """
    residual = Residual.create("odd")
    xs = field.train_points()
    ts = field.timesteps()

    def loss_fn(res: Residual) -> torch.Tensor:
        total = xs[0].new_zeros(())
        n = 0
        scales = (1.0, -1.0) if bidirectional else (1.0,)
        for pair in pairs:
            for x in xs:
                for t in ts:
                    for scale in scales:
                        student = field.noise(x, pair.target, t) + res.delta(scale)
                        if scale < 0 and bidirectional:
                            flipped = "erase" if pair.action == "enhance" else "enhance"
                            target = sd_noise_target(
                                field.noise(x, pair.neutral, t),
                                field.noise(x, pair.positive, t),
                                field.noise(x, pair.negative, t),
                                pair.guidance,
                                flipped,
                            )
                        else:
                            target = sd_noise_target(
                                field.noise(x, pair.neutral, t),
                                field.noise(x, pair.positive, t),
                                field.noise(x, pair.negative, t),
                                pair.guidance,
                                pair.action,
                            )
                        total = total + F.mse_loss(student, target)
                        n += 1
        return total / n

    return _optimize(residual, loss_fn, steps, lr, seed)


def train_music3(
    field: Field2D,
    pairs: list[Pair],
    *,
    kind: str = "nmse",
    target_mode: str = "axis",
    bidirectional: bool = True,
    steps: int = 250,
    lr: float = 0.08,
    seed: int = 0,
    mag_weight: float = 0.25,
) -> Residual:
    residual = Residual.create("odd")
    xs = field.train_points()
    ts = field.timesteps()

    def _delta(pair: Pair, x, t, direction: float) -> torch.Tensor:
        vel_pos = field.velocity(x, pair.positive, t)
        vel_neg = field.velocity(x, pair.negative, t)
        vel_neu = field.velocity(x, pair.neutral, t)
        if target_mode == "pole":
            return music3_pole_delta(
                direction, vel_pos, vel_neg, vel_neu, pair.guidance, pair.action
            )
        return music3_axis_delta(direction, vel_pos, vel_neg, pair.guidance, pair.action)

    def loss_fn(res: Residual) -> torch.Tensor:
        total = xs[0].new_zeros(())
        n = 0
        scales = (1.0, -1.0) if bidirectional else (1.0,)
        for pair in pairs:
            for x in xs:
                for t in ts:
                    vel_neu = field.velocity(x, pair.neutral, t)
                    for scale in scales:
                        vel = field.velocity(x, pair.target, t) + res.delta(scale)
                        axis = _delta(pair, x, t, scale)
                        total = total + music3_slider_loss(
                            vel, vel_neu, axis, kind, mag_weight
                        )
                        n += 1
        return total / n

    return _optimize(residual, loss_fn, steps, lr, seed)


def train_lm(
    field: Field2D,
    pairs: list[Pair],
    *,
    symmetric: bool = True,
    steps: int = 250,
    lr: float = 0.08,
    seed: int = 0,
    common_beta: float = 0.0,
) -> Residual:
    residual = Residual.create("odd_even")
    t = 0.5

    def loss_fn(res: Residual) -> torch.Tensor:
        total = residual.w_odd.new_zeros(())
        n = 0
        for pair in pairs:
            pos = field.embed(pair.positive, t)
            neg = field.embed(pair.negative, t)
            neu = field.embed(pair.neutral, t)
            tgt_plus, tgt_minus = lm_hidden_targets(
                pos, neg, neu, symmetric=symmetric, common_beta=common_beta
            )
            pred_plus = neu + res.delta(1.0)
            pred_minus = neu + res.delta(-1.0)
            total = total + F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_minus, tgt_minus)
            n += 1
        return total / n

    return _optimize(residual, loss_fn, steps, lr, seed)


def train_encoder(
    field: Field2D,
    pairs: list[Pair],
    *,
    steps: int = 250,
    lr: float = 0.08,
    seed: int = 0,
) -> Residual:
    residual = Residual.create("odd_even")
    t = 0.5

    def loss_fn(res: Residual) -> torch.Tensor:
        total = residual.w_odd.new_zeros(())
        n = 0
        for pair in pairs:
            pos = field.embed(pair.positive, t)
            neg = field.embed(pair.negative, t)
            neu = field.embed(pair.neutral, t)
            pred_plus = neu + res.delta(1.0)
            pred_minus = neu + res.delta(-1.0)
            total = total + encoder_mse_loss(pred_plus, pred_minus, pos, neg)
            n += 1
        return total / n

    return _optimize(residual, loss_fn, steps, lr, seed)


def score_residual(residual: Residual, action: str = "enhance") -> dict:
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    plus = project(d_plus)
    minus = project(d_minus)
    want = 1.0 if action == "enhance" else -1.0
    return {
        "delta_plus": [float(d_plus[0]), float(d_plus[1])],
        "delta_minus": [float(d_minus[0]), float(d_minus[1])],
        "cos_slider_plus": plus["cos_slider"],
        "cos_slider_minus": minus["cos_slider"],
        "cos_attr_plus": plus["cos_attr"],
        "leak_ratio": plus["leak_ratio"],
        "proj_slider_plus": plus["proj_slider"],
        "proj_attr_plus": plus["proj_attr"],
        "cos_plus_minus": cosine(d_plus, d_minus),
        "norm_plus": plus["norm"],
        "norm_minus": minus["norm"],
        "action": action,
        "want_sign": want,
    }


def decide(name: str, metrics: dict, claim: str) -> tuple[str, str]:
    """Geometric verdict. ``claim`` is what the method is *trying* to do."""
    leak = abs(metrics["leak_ratio"])
    slider = metrics["cos_slider_plus"] * metrics["want_sign"]
    bipolar = metrics["cos_plus_minus"]
    if claim == "track_and_disentangle":
        if slider >= 0.90 and leak <= 0.20 and bipolar <= -0.70:
            return "right", "tracks the slider, attribute leak is small, ±1 are opposite"
        why = []
        if slider < 0.90:
            why.append(f"slider cos {slider:.2f} < 0.90")
        if leak > 0.20:
            why.append(f"leak {leak:.2f} > 0.20")
        if bipolar > -0.70:
            why.append(f"±1 cos {bipolar:.2f} (not antipodal)")
        return "needs_help", "; ".join(why)
    if claim == "track_allow_leak":
        if slider >= 0.90:
            extra = f"leak {leak:.2f} (expected without attributes)"
            return ("right" if leak <= 0.20 else "needs_help"), (
                f"slider axis is right; {extra}" if leak > 0.20 else "tracks slider, little leak"
            )
        return "needs_help", f"slider cos {slider:.2f} < 0.90"
    if claim == "bipolar_hidden":
        if bipolar <= -0.85 and slider >= 0.90 and leak <= 0.25:
            return "right", "symmetric / odd targets keep ±1 opposite on the slider"
        why = []
        if bipolar > -0.85:
            why.append(f"collapse {bipolar:.2f} (need ≤ -0.85)")
        if slider < 0.90:
            why.append(f"slider cos {slider:.2f}")
        if leak > 0.25:
            why.append(f"leak {leak:.2f}")
        return "needs_help", "; ".join(why)
    if claim == "bipolar_allow_leak":
        if bipolar <= -0.85 and slider >= 0.90:
            extra = (
                f"leak {leak:.2f} remains without attributes"
                if leak > 0.20
                else "clean axis"
            )
            return "right", f"±1 opposite on the slider; {extra}"
        why = []
        if bipolar > -0.85:
            why.append(f"collapse {bipolar:.2f} (need ≤ -0.85)")
        if slider < 0.90:
            why.append(f"slider cos {slider:.2f}")
        return "needs_help", "; ".join(why)
    if claim == "raw_hidden":
        if bipolar > -0.50:
            return "needs_help", f"raw pos/neg share an even mode; ±1 cos={bipolar:.2f}"
        return "right", "raw targets happened to stay antipodal on this field"
    raise ValueError(claim)


@dataclass
class MethodSpec:
    name: str
    family: str
    claim: str
    action: str = "enhance"
    builder: str = ""
    kwargs: dict = field(default_factory=dict)


def all_specs() -> list[MethodSpec]:
    return [
        MethodSpec("sd_enhance", "sd", "track_allow_leak", builder="sd", kwargs={"with_attrs": False}),
        MethodSpec("sd_erase", "sd", "track_allow_leak", action="erase", builder="sd", kwargs={"with_attrs": False, "action": "erase"}),
        MethodSpec("sd_enhance_attrs", "sd", "track_and_disentangle", builder="sd", kwargs={"with_attrs": True}),
        MethodSpec("m3_nmse_axis", "music3", "track_allow_leak", builder="m3", kwargs={"kind": "nmse", "target_mode": "axis", "with_attrs": False}),
        MethodSpec("m3_nmse_axis_attrs", "music3", "track_and_disentangle", builder="m3", kwargs={"kind": "nmse", "target_mode": "axis", "with_attrs": True}),
        MethodSpec("m3_nmse_pole", "music3", "track_allow_leak", builder="m3", kwargs={"kind": "nmse", "target_mode": "pole", "with_attrs": False}),
        MethodSpec("m3_mse_axis", "music3", "track_allow_leak", builder="m3", kwargs={"kind": "mse", "target_mode": "axis", "with_attrs": False}),
        MethodSpec("m3_erase", "music3", "track_allow_leak", action="erase", builder="m3", kwargs={"kind": "nmse", "target_mode": "axis", "with_attrs": False, "action": "erase"}),
        MethodSpec("lm_raw", "lm", "raw_hidden", builder="lm", kwargs={"symmetric": False, "with_attrs": False}),
        MethodSpec("lm_symmetric", "lm", "bipolar_allow_leak", builder="lm", kwargs={"symmetric": True, "with_attrs": False}),
        MethodSpec("lm_raw_attrs", "lm", "bipolar_hidden", builder="lm", kwargs={"symmetric": False, "with_attrs": True}),
        MethodSpec("enc_mse", "encoder", "raw_hidden", builder="enc", kwargs={"with_attrs": False}),
        MethodSpec("enc_mse_attrs", "encoder", "bipolar_hidden", builder="enc", kwargs={"with_attrs": True}),
    ]


def run_method(spec: MethodSpec, field: Field2D, *, steps: int = 250, seed: int = 0) -> MethodResult:
    action = spec.kwargs.get("action", spec.action)
    with_attrs = spec.kwargs.get("with_attrs", False)
    if spec.builder == "sd":
        pairs = sd_pairs(with_attrs, action=action)
        residual = train_sd(field, pairs, steps=steps, seed=seed)
    elif spec.builder == "m3":
        pairs = music3_pairs(with_attrs, action=action)
        residual = train_music3(
            field,
            pairs,
            kind=spec.kwargs.get("kind", "nmse"),
            target_mode=spec.kwargs.get("target_mode", "axis"),
            bidirectional=spec.kwargs.get("bidirectional", True),
            steps=steps,
            seed=seed,
        )
    elif spec.builder == "lm":
        pairs = music3_pairs(with_attrs, action=action)
        residual = train_lm(
            field, pairs, symmetric=spec.kwargs["symmetric"], steps=steps, seed=seed
        )
    elif spec.builder == "enc":
        pairs = music3_pairs(with_attrs, action=action)
        residual = train_encoder(field, pairs, steps=steps, seed=seed)
    else:
        raise ValueError(spec.builder)
    # SD trains a noise predictor (ε = −v). Score and plot the implied
    # sample-space residual so enhance points at +slider like Music 3 velocity.
    if spec.family == "sd":
        residual = Residual("odd", -residual.w_odd.detach().clone())
    metrics = score_residual(residual, action=action)
    verdict, reason = decide(spec.name, metrics, spec.claim)
    return MethodResult(spec.name, residual, metrics, verdict, reason, spec.family)
