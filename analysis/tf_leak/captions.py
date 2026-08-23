"""Load catalog prompt yamls and score teacher-direction entanglement.

No model. Teacher direction is ``count(pos) − count(neg)`` over tokens,
plus the numeric BPM delta. Shared adjectives are the leak the 2-D gender
toy actually tested; BPM is the leak MUSIC3.md already blamed on pairs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml


_REPO = Path(__file__).resolve().parents[2]
DATA = _REPO / "conceptmod" / "textsliders" / "data"
LISTEN = _REPO / "eval" / "listen"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
BPM_RE = re.compile(r"bpm[:\s]+(\d+)", re.I)

# Boilerplate that appears on every pole of the catalog. Dropped from
# "distinctive" reports so "genre" / "bpm" / "song" do not look like leak.
STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "a", "song", "genre", "bpm", "mid", "radio", "ordinary",
    "is", "that", "this", "not", "nor", "neither",
}

# Shipped / catalog TF axes the user named, plus the fixed-BPM controls
# MUSIC3.md already wrote (dust, grit, cand-energy) and the trip-hop
# single-row pair whose loudness collapse is documented.
CATALOG = {
    "energy": DATA / "prompts-energy.yaml",
    "tempo": DATA / "prompts-tempo.yaml",
    "distortion": DATA / "prompts-distortion.yaml",
    "space": DATA / "prompts-space.yaml",
    "dust": DATA / "prompts-cand-dust-v1.yaml",
    "grit": DATA / "prompts-cand-grit-v1.yaml",
    "cand_energy": DATA / "prompts-cand-energy-v1.yaml",
    "live": DATA / "prompts-live-v3.yaml",
    "rhyme": DATA / "prompts-rhyme-v4.yaml",
    "triphop": DATA / "prompts-triphop-v3-single.yaml",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower().replace(":", " "))


def extract_bpm(text: str) -> float | None:
    found = BPM_RE.findall(text or "")
    return float(found[0]) if found else None


def load_yaml_rows(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"empty prompts: {path}")
    return raw


def load_row(name: str, row: int = 0) -> dict:
    return load_yaml_rows(CATALOG[name])[row]


def distinctive_tokens(positive: str, negative: str) -> tuple[set[str], set[str]]:
    pos, neg = Counter(tokenize(positive)), Counter(tokenize(negative))
    pos_only = {t for t in pos if pos[t] > neg[t] and t not in STOP}
    neg_only = {t for t in neg if neg[t] > pos[t] and t not in STOP}
    return pos_only, neg_only


def bow_direction(positive: str, negative: str, vocab: list[str]) -> list[float]:
    pos, neg = Counter(tokenize(positive)), Counter(tokenize(negative))
    return [float(pos[t] - neg[t]) for t in vocab]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def catalog_geometry(names: tuple[str, ...] | None = None) -> dict:
    names = tuple(names) if names is not None else tuple(CATALOG)
    rows = {name: load_row(name) for name in names}
    vocab: list[str] = sorted(
        {
            tok
            for row in rows.values()
            for tok in tokenize(row["positive"]) + tokenize(row["negative"])
        }
    )
    axes = {}
    for name, row in rows.items():
        pos, neg = row["positive"], row["negative"]
        neu = row.get("neutral") or row.get("target") or ""
        pos_only, neg_only = distinctive_tokens(pos, neg)
        bpm_pos, bpm_neg, bpm_neu = extract_bpm(pos), extract_bpm(neg), extract_bpm(neu)
        axes[name] = {
            "file": str(CATALOG[name].relative_to(_REPO)),
            "has_attributes": bool(row.get("attributes")),
            "bpm_pos": bpm_pos,
            "bpm_neg": bpm_neg,
            "bpm_neu": bpm_neu,
            "bpm_delta": (None if bpm_pos is None or bpm_neg is None else bpm_pos - bpm_neg),
            "pos_only": sorted(pos_only),
            "neg_only": sorted(neg_only),
            "bow": bow_direction(pos, neg, vocab),
            "guidance": float(row.get("guidance_scale") or row.get("guidance") or 3.0),
        }

    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, na = set(axes[a]["pos_only"]), set(axes[a]["neg_only"])
            pb, nb = set(axes[b]["pos_only"]), set(axes[b]["neg_only"])
            pairwise[f"{a}__{b}"] = {
                "bow_cos": cosine(axes[a]["bow"], axes[b]["bow"]),
                "shared_pos": sorted(pa & pb),
                "shared_neg": sorted(na & nb),
                "shared_any": sorted((pa | na) & (pb | nb)),
            }

    # Drop the raw bow vectors from the public blob (verbose, reconstructable).
    public_axes = {
        name: {k: v for k, v in axis.items() if k != "bow"} for name, axis in axes.items()
    }
    return {"vocab_size": len(vocab), "axes": public_axes, "pairwise": pairwise}


def parse_listen_rms(folder: str) -> dict[str, float]:
    """Read slider −2 / 0 / +2 rms from a LISTEN.md table. Quotes only file numbers."""
    text = (LISTEN / folder / "LISTEN.md").read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        name = cells[1]
        try:
            rms = float(cells[3])
        except ValueError:
            continue
        if "minus2" in name or name.endswith("_minus2.wav"):
            rows["minus2"] = rms
        elif "zero" in name or "base" in name:
            rows["zero"] = rms
        elif "plus2" in name:
            rows["plus2"] = rms
    if set(rows) != {"minus2", "zero", "plus2"}:
        raise ValueError(f"{folder}: could not parse −2/0/+2 rms, got {rows}")
    return rows


def existing_render_numbers() -> dict:
    """In-repo listen / sidecar numbers only. Nothing invented."""
    energy = parse_listen_rms("energy-20s")
    tempo = parse_listen_rms("tempo-20s")
    distortion = parse_listen_rms("distortion-20s")
    space = parse_listen_rms("space-20s")
    triphop = parse_listen_rms("triphop-v3-tf-raw-20s")
    dust_side = json.loads(
        (_REPO / "models" / "dust-tf-v1" / "dust-tf-v1_last.json").read_text(encoding="utf-8")
    )
    ev = dust_side["evidence"]
    return {
        "energy-20s": energy,
        "tempo-20s": tempo,
        "distortion-20s": distortion,
        "space-20s": space,
        "triphop-v3-tf-raw-20s": triphop,
        "dust-tf-v1": {
            "at_plus1_rms_pct": ev["at_plus1"]["rms_pct"],
            "at_plus1_centroid_pct": ev["at_plus1"]["centroid_pct"],
            "at_minus0p5_rms_pct": ev["at_minus0p5"]["rms_pct"],
            "at_minus0p5_centroid_pct": ev["at_minus0p5"]["centroid_pct"],
            "song_identity_onset_corr": ev["song_identity_onset_corr"],
        },
    }
