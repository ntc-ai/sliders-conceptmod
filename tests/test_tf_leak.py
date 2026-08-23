"""Gates on catalog caption geometry and the energy × tempo field."""

from __future__ import annotations

import pytest

from analysis.tf_leak.captions import (
    catalog_geometry,
    existing_render_numbers,
    extract_bpm,
    load_row,
)
from analysis.tf_leak.field_music import (
    MusicField2D,
    embed_text,
    energy_score,
    pair_from_catalog,
    project,
    score_residual,
    teacher_leak,
    tempo_score,
    train_music3,
)


_GEO = None
_FIELD = None


def geo():
    global _GEO
    if _GEO is None:
        _GEO = catalog_geometry()
    return _GEO


def field_fits():
    """Train the four load-bearing cells once for the module."""
    global _FIELD
    if _FIELD is None:
        field = MusicField2D()
        _FIELD = {}
        for name, catalog, kwargs in (
            ("energy", "energy", {"kind": "nmse", "target_mode": "axis"}),
            ("energy_pole", "energy", {"kind": "nmse", "target_mode": "pole"}),
            ("energy_ortho", "energy", {"kind": "nmse_ortho", "target_mode": "axis"}),
            ("cand_energy", "cand_energy", {"kind": "nmse", "target_mode": "axis"}),
            ("tempo", "tempo", {"kind": "nmse", "target_mode": "axis"}),
        ):
            attrs = name.endswith("_attrs")
            pair = pair_from_catalog(catalog, attributes=attrs)
            residual = train_music3(field, pair, steps=120, seed=0, **kwargs)
            _FIELD[name] = {"pair": pair, "fit": score_residual(residual), "teacher": teacher_leak(pair)}
    return _FIELD


def test_shipped_tf_yamls_have_no_gender_attributes():
    axes = geo()["axes"]
    for name in ("energy", "tempo", "distortion", "space", "dust"):
        assert axes[name]["has_attributes"] is False


def test_energy_tf_moves_bpm_like_tempo():
    energy = geo()["axes"]["energy"]
    tempo = geo()["axes"]["tempo"]
    assert energy["bpm_pos"] == 168.0
    assert energy["bpm_neg"] == 52.0
    assert energy["bpm_delta"] == 116.0
    assert tempo["bpm_delta"] == 132.0
    assert energy["bpm_delta"] > 100.0


def test_dust_live_cand_energy_pin_bpm():
    for name in ("dust", "live", "cand_energy", "grit"):
        axis = geo()["axes"][name]
        assert axis["bpm_delta"] == 0.0
        assert axis["bpm_pos"] == axis["bpm_neg"] == axis["bpm_neu"]


def test_energy_tempo_bow_cosine_is_zero():
    """The 2-D gender leak is not this: energy and tempo share no adjectives."""
    pair = geo()["pairwise"]["energy__tempo"]
    assert pair["bow_cos"] == pytest.approx(0.0, abs=1e-9)
    assert pair["shared_any"] == []


def test_energy_grit_bow_cosine_is_zero():
    pair = geo()["pairwise"]["energy__grit"]
    assert pair["bow_cos"] == pytest.approx(0.0, abs=1e-9)
    assert pair["shared_any"] == []


def test_energy_distortion_share_only_aggressive():
    pair = geo()["pairwise"]["energy__distortion"]
    assert pair["shared_pos"] == ["aggressive"]
    assert pair["bow_cos"] < 0.10


def test_listen_rms_quoted_from_files():
    renders = existing_render_numbers()
    assert renders["energy-20s"]["minus2"] == pytest.approx(0.0891)
    assert renders["energy-20s"]["zero"] == pytest.approx(0.1119)
    assert renders["energy-20s"]["plus2"] == pytest.approx(0.1492)
    assert renders["tempo-20s"]["minus2"] == pytest.approx(0.0540)
    assert renders["tempo-20s"]["zero"] == pytest.approx(0.0544)
    assert renders["tempo-20s"]["plus2"] == pytest.approx(0.0538)
    assert renders["distortion-20s"]["minus2"] == pytest.approx(0.0503)
    assert renders["distortion-20s"]["zero"] == pytest.approx(0.0672)
    assert renders["distortion-20s"]["plus2"] == pytest.approx(0.0934)
    assert renders["triphop-v3-tf-raw-20s"]["plus2"] == pytest.approx(0.0035)
    assert renders["dust-tf-v1"]["at_plus1_rms_pct"] == pytest.approx(-7.5)
    assert renders["dust-tf-v1"]["at_plus1_centroid_pct"] == pytest.approx(41.5)


def test_tempo_tf_rms_stays_flat():
    rms = existing_render_numbers()["tempo-20s"]
    assert abs(rms["plus2"] - rms["zero"]) < 0.002
    assert abs(rms["minus2"] - rms["zero"]) < 0.002


def test_distortion_tf_moves_loudness():
    rms = existing_render_numbers()["distortion-20s"]
    assert rms["plus2"] > rms["zero"] > rms["minus2"]


def test_energy_caption_sits_on_energy_and_tempo():
    row = load_row("energy")
    pos = embed_text(row["positive"])
    neg = embed_text(row["negative"])
    axis = pos - neg
    leak = project(axis)["leak_ratio"]
    assert energy_score(row["positive"]) > 0.5
    assert energy_score(row["negative"]) < -0.5
    assert tempo_score(row["positive"]) > 0.5
    assert tempo_score(row["negative"]) < -0.5
    assert leak > 0.20


def test_cand_energy_is_pure_energy():
    row = load_row("cand_energy")
    axis = embed_text(row["positive"]) - embed_text(row["negative"])
    scored = project(axis)
    assert extract_bpm(row["positive"]) == extract_bpm(row["negative"]) == 110.0
    assert abs(scored["leak_ratio"]) < 0.15
    assert scored["cos_energy"] > 0.95


def test_tempo_caption_is_pure_tempo():
    row = load_row("tempo")
    axis = embed_text(row["positive"]) - embed_text(row["negative"])
    scored = project(axis)
    assert abs(energy_score(row["positive"])) < 0.15
    assert abs(energy_score(row["negative"])) < 0.15
    assert scored["cos_tempo"] > 0.95
    assert abs(scored["leak_ratio"]) > 5.0  # almost no energy component


def test_default_tf_on_energy_still_leaks_tempo():
    fit = field_fits()["energy"]["fit"]
    assert fit["cos_energy_plus"] > 0.70
    assert abs(fit["leak_ratio"]) > 0.20


def test_pole_and_ortho_do_not_kill_energy_tempo_leak():
    fits = field_fits()
    for name in ("energy_pole", "energy_ortho"):
        assert abs(fits[name]["fit"]["leak_ratio"]) > 0.20


def test_gender_prefix_does_not_unpin_bpm():
    raw = pair_from_catalog("energy", attributes=False)
    pinned = pair_from_catalog("energy", attributes=True)
    assert pinned.positive.text.startswith("A man is singing.")
    assert extract_bpm(raw.positive.text) == extract_bpm(pinned.positive.text)
    assert abs(teacher_leak(raw)["leak_ratio"] - teacher_leak(pinned)["leak_ratio"]) < 1e-6


def test_cand_energy_default_tf_does_not_need_help_on_tempo():
    fit = field_fits()["cand_energy"]["fit"]
    assert fit["cos_energy_plus"] > 0.90
    assert abs(fit["leak_ratio"]) < 0.20


def test_tempo_default_tf_does_not_leak_energy():
    fit = field_fits()["tempo"]["fit"]
    assert fit["cos_tempo_plus"] > 0.90
    assert abs(fit["cos_energy_plus"]) < 0.20
