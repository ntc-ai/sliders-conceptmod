"""Prompt-gated 2-D teacher: slider axis vs attribute axis.

Axes are orthonormal. Ungated slider tokens carry a gender leak (energetic
people read male) and a shared even component (both poles sit above a quiet
neutral). Prefixing ``male`` / ``female`` pins the attribute, so leakage is
visible without attributes and cancelled with them.

This is a synthetic field. It does not load Music 3 or SD weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


E_SLIDER = torch.tensor([1.0, 0.0])  # energetic (+) ↔ calm (−)
E_ATTR = torch.tensor([0.0, 1.0])  # male (+) ↔ female (−)

# Ungated concept locations. Chosen so:
# - pos − neg is *not* parallel to e_slider (leak)
# - (pos + neg)/2 is far from neu (even / common mode, LM collapse)
# - neu is off the pos−neg chord (axis vs pole diverge)
UNGATED = {
    +1.0: torch.tensor([1.00, 0.95]),  # energetic, leaks male
    -1.0: torch.tensor([-1.00, 0.35]),  # calm, still slightly male
    0.0: torch.tensor([0.00, -0.70]),  # generic "song", quieter than both poles
}

# Extra leak at low t so mse (low-t heavy) and nmse (equal t) can disagree.
LEAK_T = 0.28


@dataclass(frozen=True)
class Prompt:
    name: str
    slider: float  # +1 energetic, −1 calm, 0 unspecified
    attr: float  # +1 male, −1 female, 0 ungated


# Canonical prompts. Attribute expansion prefixes these names.
PROMPTS = {
    "song": Prompt("song", 0.0, 0.0),
    "energetic": Prompt("energetic", 1.0, 0.0),
    "calm": Prompt("calm", -1.0, 0.0),
    "male song": Prompt("male song", 0.0, 1.0),
    "male energetic": Prompt("male energetic", 1.0, 1.0),
    "male calm": Prompt("male calm", -1.0, 1.0),
    "female song": Prompt("female song", 0.0, -1.0),
    "female energetic": Prompt("female energetic", 1.0, -1.0),
    "female calm": Prompt("female calm", -1.0, -1.0),
}


class Field2D:
    """Teacher that maps (x, prompt, t) → velocity in R²."""

    def embed(self, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        if isinstance(prompt, str):
            prompt = PROMPTS[prompt]
        if prompt.attr != 0.0:
            loc = torch.stack(
                [
                    torch.tensor(prompt.slider),
                    torch.tensor(prompt.attr),
                ]
            )
        else:
            loc = UNGATED[float(prompt.slider)].clone()
            # Entanglement is stronger near noise (low t).
            loc = loc + torch.tensor([0.0, (1.0 - t) * LEAK_T * (0.7 + 0.3 * prompt.slider)])
        return loc

    def velocity(self, x: torch.Tensor, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        """Flow-matching velocity toward the prompt location.

        Magnitude is larger at low t, matching the Music 3 observation that
        ``||vel_pos - vel_neg||`` spans a wide range across the solve.
        """
        mag = 1.55 - 1.15 * float(t)
        return mag * (self.embed(prompt, t) - x)

    def noise(self, x: torch.Tensor, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        """SD-style noise prediction: ε = −v on this field."""
        return -self.velocity(x, prompt, t)

    def train_points(self) -> list[torch.Tensor]:
        return [
            torch.tensor([fx, fy])
            for fx in (-0.6, 0.0, 0.6)
            for fy in (-0.6, 0.0, 0.6)
        ]

    def timesteps(self) -> list[float]:
        return [0.1, 0.3, 0.5, 0.7, 0.9]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
    )


def project(delta: torch.Tensor) -> dict[str, float]:
    """Project a residual onto the two orthonormal axes."""
    ds = float(delta.flatten() @ E_SLIDER)
    da = float(delta.flatten() @ E_ATTR)
    leak = da / (abs(ds) + 1e-8)
    return {
        "proj_slider": ds,
        "proj_attr": da,
        "cos_slider": cosine(delta, E_SLIDER),
        "cos_attr": cosine(delta, E_ATTR),
        "leak_ratio": leak,
        "norm": float(delta.norm()),
    }
