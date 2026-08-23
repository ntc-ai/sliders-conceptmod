"""CPU-pure slider targets extracted from the live trainers.

The GPU trainers can import these later; the 2-D fixture already does.
Formulas are copied from:

- ``PromptEmbedsPair._enhance`` / ``_erase`` in ``prompt_util.py``
  (SD / XL / SD3 / Flux / Cascade noise-prediction sliders)
- ``_slider_loss`` and ``_target_delta`` in ``train_lora_music3.py``
- LM raw vs ``--symmetric`` targets in ``train_lm_slider_music3.py``
- Encoder MSE in ``train_encoder_music3.py``

No Hub, no GPU, no model weights.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def sd_noise_target(
    neutral: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    guidance: float,
    action: str = "enhance",
) -> torch.Tensor:
    """Teacher noise (or any same-shaped score) the SD slider fits.

    ``prompt_util.PromptEmbedsPair._enhance`` / ``_erase``:

        enhance:  neu + g * (pos - neg)
        erase:    neu - g * (pos - neg)
    """
    if action == "erase":
        return neutral - guidance * (positive - negative)
    if action == "enhance":
        return neutral + guidance * (positive - negative)
    raise ValueError("action must be erase or enhance")


def music3_axis_delta(
    direction: float,
    vel_pos: torch.Tensor,
    vel_neg: torch.Tensor,
    guidance: float,
    action: str = "enhance",
) -> torch.Tensor:
    """``_target_delta`` with ``--target_mode axis`` (Music 3 default).

    ``axis = sign * g * (vel_pos - vel_neg)``, then ``direction * axis``.
    ``sign`` is +1 for enhance, -1 for erase.
    """
    if action not in ("enhance", "erase"):
        raise ValueError(f"action must be enhance or erase, got {action!r}")
    sign = 1.0 if action == "enhance" else -1.0
    axis = sign * guidance * (vel_pos - vel_neg)
    return direction * axis


def music3_pole_delta(
    direction: float,
    vel_pos: torch.Tensor,
    vel_neg: torch.Tensor,
    vel_neu: torch.Tensor,
    guidance: float,
    action: str = "enhance",
) -> torch.Tensor:
    """``_target_delta`` with ``--target_mode pole``.

    Each slider sign aims at its own caption's displacement from neutral:
    ``abs(direction) * g * (nearer_pole - vel_neu)``.
    """
    if action not in ("enhance", "erase"):
        raise ValueError(f"action must be enhance or erase, got {action!r}")
    sign = 1.0 if action == "enhance" else -1.0
    toward_pos = (direction * sign) >= 0.0
    pole = vel_pos if toward_pos else vel_neg
    return abs(direction) * guidance * (pole - vel_neu)


def music3_slider_loss(
    vel: torch.Tensor,
    vel_neu: torch.Tensor,
    axis: torch.Tensor,
    kind: str,
    mag_weight: float,
    gain_weight: float = 0.0,
    gain_mode: str = "penalty",
    gain_tw: float = 1.0,
) -> torch.Tensor:
    """Verbatim extract of ``train_lora_music3._slider_loss``.

    ``axis`` is the signed target delta, ``guidance * (vel_pos - vel_neg)`` for +1.
    """
    delta = vel - vel_neu
    unit = vel_neu.flatten()
    unit = unit / unit.norm().clamp_min(1e-8)
    if gain_weight > 0.0:
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
        return F.mse_loss(vel, vel_neu + axis) + gain_term
    if kind == "nmse":
        scale = axis.pow(2).mean().clamp_min(1e-8)
        return F.mse_loss(vel, vel_neu + axis) / scale + gain_term
    if kind == "nmse_ortho":
        d_par = delta.flatten() @ unit
        a_par = axis.flatten() @ unit
        d_perp = delta - (d_par * unit).view_as(delta)
        a_perp = axis - (a_par * unit).view_as(axis)
        scale = a_perp.pow(2).mean().clamp_min(1e-8)
        return (d_perp - a_perp).pow(2).mean() / scale + gain_term
    if kind == "cos":
        cos = F.cosine_similarity(
            delta.flatten().unsqueeze(0), axis.flatten().unsqueeze(0)
        ).squeeze()
        ratio = delta.norm() / axis.norm().clamp_min(1e-8)
        return (1.0 - cos) + mag_weight * (ratio - 1.0).pow(2) + gain_term
    raise ValueError(f"unknown loss {kind!r}")


def lm_hidden_targets(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    *,
    symmetric: bool = True,
    target_scale: float = 1.0,
    common_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LM slider targets from ``train_lm_slider_music3.py``.

    symmetric (default): ``tgt(±1) = neu ± (pos-neg)/2 * target_scale + β * common``
    raw: ``tgt(+1), tgt(-1) = pos, neg``
    """
    if symmetric:
        axis = (pos - neg) / 2.0 * float(target_scale)
        common = (pos + neg) / 2.0 - neu
        return neu + axis + float(common_beta) * common, neu - axis + float(common_beta) * common
    if float(target_scale) != 1.0:
        raise ValueError("--target_scale is defined for symmetric targets only")
    return pos, neg


def encoder_mse_loss(
    pred_pos: torch.Tensor,
    pred_neg: torch.Tensor,
    pos_tgt: torch.Tensor,
    neg_tgt: torch.Tensor,
) -> torch.Tensor:
    """Encoder-only slider: MSE each pole to the frozen teacher encoding."""
    return F.mse_loss(pred_pos, pos_tgt) + F.mse_loss(pred_neg, neg_tgt)


def expand_attributes_sd(row: dict, attributes: Iterable[str]) -> list[dict]:
    """``prompt_util.load_prompts_from_yaml`` prefixing (all five caption fields)."""
    attrs = [a.strip() for a in attributes if str(a).strip()]
    if not attrs:
        return [dict(row)]
    out = []
    for att in attrs:
        copy_ = dict(row)
        for key in ("target", "positive", "neutral", "negative", "unconditional"):
            if key in copy_ and copy_[key] is not None:
                copy_[key] = f"{att} {copy_[key]}"
        out.append(copy_)
    return out


def expand_attributes_music3(row: dict) -> list[dict]:
    """``train_lora_music3._expand_attributes`` (target/pos/neg/neu only)."""
    attributes = row.get("attributes")
    if not attributes:
        return [dict(row)]
    rows = []
    for attribute in attributes:
        prefix = str(attribute).strip()
        item = dict(row)
        for key in ("target", "positive", "negative", "neutral"):
            value = row.get(key)
            if value:
                item[key] = f"{prefix} {value}"
        rows.append(item)
    return rows
