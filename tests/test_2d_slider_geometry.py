"""CPU gates on slider geometry. Fail if a claimed-correct method regresses."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from analysis.slider2d.field import E_ATTR, E_SLIDER, Field2D
from analysis.slider2d.train import music3_pairs, sd_pairs
from conceptmod.textsliders.slider_targets import (
    expand_attributes_music3,
    expand_attributes_sd,
    lm_hidden_targets,
    music3_axis_delta,
    music3_slider_loss,
    sd_noise_target,
)


# Shared across the module so the suite stays well under 30s.
_RESULTS = None


def results():
    global _RESULTS
    if _RESULTS is None:
        from analysis.slider2d.run_analysis import run_all

        _RESULTS = {r.name: r for r in run_all(steps=200, seed=0)}
    return _RESULTS


def test_axes_orthonormal():
    assert float(E_SLIDER @ E_ATTR) == pytest.approx(0.0)
    assert float(E_SLIDER.norm()) == pytest.approx(1.0)
    assert float(E_ATTR.norm()) == pytest.approx(1.0)


def test_ungated_pos_neg_leaks_attribute():
    field = Field2D()
    axis = field.embed("energetic", 0.5) - field.embed("calm", 0.5)
    leak = abs(float(axis @ E_ATTR) / float(axis @ E_SLIDER))
    assert leak > 0.20


def test_gated_pos_neg_is_pure_slider():
    field = Field2D()
    axis = field.embed("male energetic") - field.embed("male calm")
    assert float(axis @ E_ATTR) == pytest.approx(0.0)
    assert float(axis @ E_SLIDER) == pytest.approx(2.0)


def test_sd_target_matches_prompt_util_formula():
    neu, pos, neg = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.2]), torch.tensor([-1.0, 0.1])
    g = 4.0
    got = sd_noise_target(neu, pos, neg, g, "enhance")
    assert torch.allclose(got, neu + g * (pos - neg))
    got_e = sd_noise_target(neu, pos, neg, g, "erase")
    assert torch.allclose(got_e, neu - g * (pos - neg))


def test_music3_axis_is_signed_pos_minus_neg():
    vel_pos, vel_neg = torch.tensor([1.0, 0.4]), torch.tensor([-0.5, 0.1])
    d = music3_axis_delta(1.0, vel_pos, vel_neg, 3.0, "enhance")
    assert torch.allclose(d, 3.0 * (vel_pos - vel_neg))
    d_erase = music3_axis_delta(1.0, vel_pos, vel_neg, 3.0, "erase")
    assert torch.allclose(d_erase, -3.0 * (vel_pos - vel_neg))


def test_music3_nmse_zero_at_teacher():
    vel_neu = torch.tensor([0.2, -0.1])
    axis = torch.tensor([1.5, 0.0])
    vel = vel_neu + axis
    loss = music3_slider_loss(vel, vel_neu, axis, "nmse", 0.25)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_lm_symmetric_is_odd_around_neu():
    pos, neg, neu = torch.tensor([1.0, 0.9]), torch.tensor([-1.0, 0.4]), torch.tensor([0.0, -0.7])
    plus, minus = lm_hidden_targets(pos, neg, neu, symmetric=True)
    assert torch.allclose((plus + minus) / 2, neu)
    assert torch.allclose(plus - neu, neu - minus)


def test_attribute_expansion_prefixes():
    row = {"target": "song", "positive": "energetic", "negative": "calm", "neutral": "song", "unconditional": ""}
    sd = expand_attributes_sd(row, ["male", "female"])
    assert [r["target"] for r in sd] == ["male song", "female song"]
    assert sd[0]["positive"] == "male energetic"
    m3 = expand_attributes_music3({**row, "attributes": ["male", "female"]})
    assert m3[0]["target"] == "male song"
    assert "unconditional" in m3[0]


def test_sd_pairs_and_music3_pairs_match_names():
    assert [p.target.name for p in sd_pairs(True)] == ["male song", "female song"]
    assert [p.target.name for p in music3_pairs(True)] == ["male song", "female song"]
    assert [p.target.name for p in sd_pairs(False)] == ["song"]


def test_claimed_correct_sd_attrs_tracks_and_disentangles():
    r = results()["sd_enhance_attrs"]
    assert r.verdict == "right"
    assert r.metrics["cos_slider_plus"] > 0.90
    assert abs(r.metrics["leak_ratio"]) < 0.20


def test_claimed_correct_m3_attrs_tracks_and_disentangles():
    r = results()["m3_nmse_axis_attrs"]
    assert r.verdict == "right"
    assert r.metrics["cos_slider_plus"] > 0.90
    assert abs(r.metrics["leak_ratio"]) < 0.20


def test_noattrs_leaks_attribute():
    for name in ("sd_enhance", "m3_nmse_axis"):
        r = results()[name]
        assert abs(r.metrics["leak_ratio"]) > 0.20
        assert r.verdict == "needs_help"


def test_erase_flips_slider_axis():
    enhance = results()["sd_enhance"].metrics["cos_slider_plus"]
    erase = results()["sd_erase"].metrics["cos_slider_plus"]
    assert enhance > 0.85
    assert erase < -0.85
    m3e = results()["m3_erase"].metrics["cos_slider_plus"]
    assert m3e < -0.85


def test_odd_lora_is_antipodal():
    for name in ("sd_enhance", "m3_nmse_axis", "m3_nmse_pole"):
        assert results()[name].metrics["cos_plus_minus"] == pytest.approx(-1.0, abs=1e-4)


def test_lm_symmetric_antipodal_and_on_axis():
    r = results()["lm_symmetric"]
    assert r.verdict == "right"
    assert r.metrics["cos_plus_minus"] < -0.85
    assert r.metrics["cos_slider_plus"] > 0.90


def test_lm_raw_collapses():
    r = results()["lm_raw"]
    assert r.metrics["cos_plus_minus"] > -0.50
    assert r.verdict == "needs_help"


def test_encoder_raw_matches_lm_raw_geometry():
    enc = results()["enc_mse"]
    lm = results()["lm_raw"]
    assert enc.verdict == lm.verdict == "needs_help"
    assert enc.metrics["cos_plus_minus"] > -0.50


def test_sd_and_music3_same_direction_without_attrs():
    sd = torch.tensor(results()["sd_enhance"].metrics["delta_plus"])
    m3 = torch.tensor(results()["m3_nmse_axis"].metrics["delta_plus"])
    cos = F.cosine_similarity(sd.unsqueeze(0), m3.unsqueeze(0)).item()
    assert cos > 0.95


def test_sd1_intent_uncond_is_negative():
    """SD1 yaml uses unconditional as the opposite pole; same enhance target."""
    neu = torch.tensor([0.0, -0.2])
    pos = torch.tensor([1.0, 0.3])
    uncond = torch.tensor([-1.0, 0.1])
    t = sd_noise_target(neu, pos, uncond, 4.0, "enhance")
    assert torch.allclose(t, neu + 4.0 * (pos - uncond))
