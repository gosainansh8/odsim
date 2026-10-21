"""Simulation configuration: a validated, immutable description of one model run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

SITE_KINDS = ("treatment", "ssp")

# Parameters that must be probabilities in [0, 1].
_PROBABILITY_FIELDS = (
    "frac_housed",
    "p_seek",
    "p_overdose",
    "p_fatal",
    "naloxone_effect",
    "p_enroll",
    "p_dropout",
    "bystander_effect",
)

# Parameters that must be non-negative (multipliers and radii).
_NON_NEGATIVE_FIELDS = (
    "naloxone_radius",
    "enroll_radius",
    "unhoused_enroll_mult",
    "unhoused_od_mult",
    "treatment_od_mult",
)


@dataclass(frozen=True)
class Site:
    """A fixed service location on the grid."""

    x: int
    y: int
    kind: str  # "treatment" (enrollment point) or "ssp" (syringe service / naloxone)

    def __post_init__(self) -> None:
        if self.kind not in SITE_KINDS:
            raise ValueError(f"site kind must be one of {SITE_KINDS}, got {self.kind!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Site:
        return cls(x=int(d["x"]), y=int(d["y"]), kind=str(d["kind"]))


@dataclass(frozen=True)
class SimConfig:
    """All model parameters. Rates are per agent per step (one step = one day)."""

    width: int = 50
    height: int = 50
    n_agents: int = 1000
    n_steps: int = 365
    sites: tuple[Site, ...] = ()

    frac_housed: float = 0.5
    p_seek: float = 0.2  # chance an agent walks toward its nearest site instead of randomly

    p_overdose: float = 0.002
    p_fatal: float = 0.1
    naloxone_radius: float = 3.0  # overdoses within this distance of an SSP get naloxone
    naloxone_effect: float = 0.6  # relative reduction in fatality when naloxone is nearby
    # Relative reduction in fatality when another living agent is in the same or an
    # adjacent cell (a witnessed overdose). 0 disables agent-agent interaction.
    bystander_effect: float = 0.0

    enroll_radius: float = 2.0
    p_enroll: float = 0.05
    p_dropout: float = 0.01

    unhoused_enroll_mult: float = 0.67
    unhoused_od_mult: float = 1.5
    treatment_od_mult: float = 0.3

    def __post_init__(self) -> None:
        for name in ("width", "height", "n_agents", "n_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in _PROBABILITY_FIELDS:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        for name in _NON_NEGATIVE_FIELDS:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")
        # Accept lists of Site or dicts, store as an immutable tuple of Site.
        sites = tuple(s if isinstance(s, Site) else Site.from_dict(s) for s in self.sites)
        for s in sites:
            if not (0 <= s.x < self.width and 0 <= s.y < self.height):
                raise ValueError(f"site {s} lies outside the {self.width}x{self.height} grid")
        object.__setattr__(self, "sites", sites)

    # ---- construction helpers -------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> SimConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sites"] = [asdict(s) for s in self.sites]
        return d

    def with_overrides(self, **overrides: Any) -> SimConfig:
        """Return a copy with some parameters changed.

        The special key ``extra_sites`` appends sites instead of replacing them,
        which is how scenarios such as "open two new treatment sites" are expressed.
        """
        overrides = dict(overrides)
        extra = overrides.pop("extra_sites", None)
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if extra:
            base_sites = overrides.get("sites", self.sites)
            overrides["sites"] = tuple(base_sites) + tuple(extra)
        return replace(self, **overrides)

    def sites_of_kind(self, kind: str) -> list[Site]:
        return [s for s in self.sites if s.kind == kind]
