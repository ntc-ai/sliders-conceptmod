"""Sweep spec: the frozen contract for one comparison sweep.

Everything the pipeline randomizes or measures is pinned here, so a result can
be reproduced from the spec alone and a seed change can never masquerade as an
approach effect. Validation is deliberately strict and loud: a spec that would
produce an unpaired or unscoreable comparison refuses to load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Features the scorer knows. ln-ratio deltas against the same-seed zero clip
# except width_db, which is measured as an absolute dB difference.
SCORED_FEATURES = ("rms_pc", "centroid", "hi4k", "flatness", "crest", "width_db", "flux_p90_med")

# One "clearly audible step" per feature, in ln units (width_db in dB).
# Judgement calls, not measurements (SCORING.md says so out loud); they are
# frozen here so the optimizer cannot own them.
AUDIBLE_STEP = {
    "rms_pc": 0.345,        # ~3 dB
    "centroid": 0.182,      # ~20 %
    "hi4k": 0.405,          # ~50 %
    "flatness": 0.405,      # ~50 %
    "crest": 0.182,         # ~20 %
    "width_db": 2.0,        # 2 dB of side/mid
    "flux_p90_med": 0.223,  # ~25 %
}


@dataclass
class AxisSpec:
    """The human-fixed measurement contract for the concept."""

    intended: dict[str, float]           # feature -> sign (+1 / -1)
    level_axis: bool = False             # loudness axes get a wider G2 band
    must_not_move: dict[str, float] = field(default_factory=dict)  # feature -> max |ln delta|

    def __post_init__(self) -> None:
        if not self.intended:
            raise ValueError("axis.intended must name at least one feature")
        for feat in list(self.intended) + list(self.must_not_move):
            if feat not in SCORED_FEATURES:
                raise ValueError(f"axis references unknown feature {feat!r}; known: {SCORED_FEATURES}")
        for feat, sign in self.intended.items():
            if float(sign) not in (-1.0, 1.0):
                raise ValueError(f"axis.intended[{feat!r}] must be +-1, got {sign}")


@dataclass
class VariantSpec:
    name: str
    args: dict[str, object] = field(default_factory=dict)   # trainer CLI overrides
    role: str = "candidate"   # candidate | baseline | floor | alpha | control

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError(f"bad variant name {self.name!r}")
        if self.role not in ("candidate", "baseline", "floor", "alpha", "control"):
            raise ValueError(f"unknown role {self.role!r} for variant {self.name}")


@dataclass
class SweepSpec:
    sweep: str
    prompts_file: Path
    cache_dir: Path
    base_args: dict[str, object]
    compare_seeds: list[int]
    holdout_seeds: list[int]
    null_seeds: list[int]
    scales: list[float]
    axis: AxisSpec
    baseline: str
    variants: list[VariantSpec]
    render_duration: float = 4.0
    over_scale: float = 1.5
    long_duration: float = 20.0
    spec_path: Path | None = None

    # ---- derived paths -------------------------------------------------
    @property
    def models_root(self) -> Path:
        return REPO_ROOT / "models" / "pipeline" / self.sweep

    @property
    def listen_root(self) -> Path:
        return REPO_ROOT / "eval" / "listen" / "pipeline" / self.sweep

    def variant(self, name: str) -> VariantSpec:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"no variant {name!r} in sweep {self.sweep}")

    # ---- validation ----------------------------------------------------
    def __post_init__(self) -> None:
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"prompts_file missing: {self.prompts_file}")
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"cache_dir missing: {self.cache_dir} — build the condition cache first "
                "(train once without --skip_ar), the pipeline never runs the AR stage implicitly"
            )
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variant names in {self.sweep}")
        if self.baseline not in names:
            raise ValueError(f"baseline {self.baseline!r} is not a variant")
        if 0.0 not in self.scales:
            raise ValueError("scales must include 0.0 — every delta is paired against the same-seed zero clip")
        if not any(s > 0 for s in self.scales) or not any(s < 0 for s in self.scales):
            raise ValueError("scales must cover both signs — a dead pole must be measurable")
        if len(self.compare_seeds) < 3:
            raise ValueError("need >= 3 compare seeds for medians and sign tests")
        overlap = set(self.compare_seeds) & set(self.holdout_seeds)
        if overlap:
            raise ValueError(f"holdout seeds leak into compare set: {sorted(overlap)}")
        for key in ("prompts_file", "cache_dir", "save_dir", "name", "device"):
            if key in self.base_args or any(key in v.args for v in self.variants):
                raise ValueError(f"{key!r} is pipeline-owned; remove it from base/variant args")
        for key in ("seed", "cond_seeds", "eval_seed"):
            if key not in self.base_args:
                raise ValueError(
                    f"base args must pin {key!r} explicitly — unpinned seeds make the "
                    "comparison unpaired and unreproducible"
                )
        if abs(self.over_scale) <= max(abs(s) for s in self.scales):
            raise ValueError("over_scale must lie beyond the ladder (G7 over-range)")


def _as_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_spec(path: Path) -> SweepSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: spec must be a mapping")
    try:
        pair = raw["pair"]
        seeds = raw["seeds"]
        spec = SweepSpec(
            sweep=str(raw["sweep"]),
            prompts_file=_as_path(REPO_ROOT, pair["prompts_file"]),
            cache_dir=_as_path(REPO_ROOT, pair["cache_dir"]),
            base_args=dict(raw.get("base", {})),
            compare_seeds=[int(s) for s in seeds["compare"]],
            holdout_seeds=[int(s) for s in seeds.get("holdout", [])],
            # YAML parses a bare `null:` key as the None key (not the string
            # "null"), so accept both spellings; an empty pool fails loudly in
            # null_distribution, but catch the spec mistake here first.
            null_seeds=[int(s) for s in (seeds.get("null") or seeds.get(None) or seeds.get("null_seeds") or [])],
            scales=[float(s) for s in raw["scales"]],
            axis=AxisSpec(
                intended={k: float(v) for k, v in raw["axis"]["intended"].items()},
                level_axis=bool(raw["axis"].get("level_axis", False)),
                must_not_move={k: float(v) for k, v in raw["axis"].get("must_not_move", {}).items()},
            ),
            baseline=str(raw["baseline"]),
            variants=[
                VariantSpec(name=str(v["name"]), args=dict(v.get("args", {})), role=str(v.get("role", "candidate")))
                for v in raw["variants"]
            ],
            render_duration=float(raw.get("render", {}).get("duration", 4.0)),
            over_scale=float(raw.get("render", {}).get("over_scale", 1.5)),
            long_duration=float(raw.get("render", {}).get("long_duration", 20.0)),
            spec_path=Path(path).resolve(),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing required spec key {exc}") from exc
    return spec
