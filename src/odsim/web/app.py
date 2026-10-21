"""Web API and dashboard.

    pip install 'odsim[web]'
    odsim serve            # or: uvicorn odsim.web.app:app

Endpoints
    GET  /                 interactive dashboard
    GET  /api/health       liveness check
    GET  /api/parameters   adjustable parameters, their ranges and base values
    POST /api/simulate     run baseline vs a user-defined intervention

Every request is validated against fixed ranges and capped in size, so a public
deployment cannot be asked for an arbitrarily expensive simulation. Identical
requests are served from an LRU cache because simulations are deterministic
given their seed.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import SimConfig
from ..experiments import compare_scenarios
from ..model import Simulation

HERE = Path(__file__).parent
MAX_AGENTS = int(os.environ.get("ODSIM_MAX_AGENTS", 5000))
MAX_REPS = int(os.environ.get("ODSIM_MAX_REPS", 8))

# name -> (label, min, max, step, help). Only these may be changed through the API.
PARAMETERS: dict[str, tuple[str, float, float, float, str]] = {
    "frac_housed": ("Share housed", 0.0, 1.0, 0.05, "Fraction of people with stable housing"),
    "p_enroll": ("Treatment enrollment rate", 0.0, 0.02, 0.001,
                 "Daily chance of enrolling when near a treatment site"),  # fmt: skip
    "unhoused_enroll_mult": ("Enrollment access if unhoused", 0.0, 1.0, 0.05,
                             "Enrollment rate for unhoused people, relative to housed"),  # fmt: skip
    "p_dropout": ("Treatment dropout rate", 0.0, 0.05, 0.001, "Daily chance of leaving treatment"),
    "naloxone_radius": ("Naloxone reach", 0.0, 20.0, 0.5,
                        "Distance from a syringe service site within which naloxone is on hand"),  # fmt: skip
    "naloxone_effect": ("Naloxone effectiveness", 0.0, 1.0, 0.05,
                        "Reduction in fatality when naloxone is nearby"),  # fmt: skip
    "bystander_effect": ("Bystander response", 0.0, 1.0, 0.05,
                         "Reduction in fatality when someone else is nearby"),  # fmt: skip
}


def load_base_config() -> SimConfig:
    path = os.environ.get("ODSIM_CONFIG", HERE / "default_config.json")
    return SimConfig.from_json(path)


BASE = load_base_config()


class SimulateRequest(BaseModel):
    intervention: dict[str, float] = Field(default_factory=dict)
    n_agents: int = Field(2000, ge=100, le=MAX_AGENTS)
    reps: int = Field(4, ge=1, le=MAX_REPS)
    seed: int = Field(0, ge=0, le=10_000)


app = FastAPI(title="odsim", version="0.2.0")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/parameters")
def parameters() -> dict:
    return {
        "parameters": [
            {"name": k, "label": label, "min": lo, "max": hi, "step": step, "help": help_,
             "base": getattr(BASE, k)}  # fmt: skip
            for k, (label, lo, hi, step, help_) in PARAMETERS.items()
        ],
        "limits": {"max_agents": MAX_AGENTS, "max_reps": MAX_REPS},
        "grid": {"width": BASE.width, "height": BASE.height},
        "n_steps": BASE.n_steps,
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest) -> dict:
    for name, value in req.intervention.items():
        if name not in PARAMETERS:
            raise HTTPException(422, f"parameter {name!r} cannot be changed")
        _, lo, hi, _, _ = PARAMETERS[name]
        if not lo <= value <= hi:
            raise HTTPException(422, f"{name} must be between {lo} and {hi}")
    key = json.dumps(
        {"i": dict(sorted(req.intervention.items())), "n": req.n_agents, "r": req.reps, "s": req.seed}
    )
    return _simulate_cached(key)


@lru_cache(maxsize=256)
def _simulate_cached(key: str) -> dict:
    spec = json.loads(key)
    base = BASE.with_overrides(n_agents=spec["n"])
    comparison = compare_scenarios(
        base, {"baseline": {}, "intervention": spec["i"]}, reps=spec["r"], seed=spec["s"]
    )
    table = comparison.table.replace({np.nan: None}).to_dict(orient="records")
    curves = {
        name: {
            "time": [0, *km.times.tolist(), base.n_steps],
            "survival": [1.0, *km.survival.tolist(), float(km.survival[-1]) if km.survival.size else 1.0],
        }
        for name, km in comparison.curves.items()
    }
    return {
        "summary": {row["scenario"]: row for row in table},
        "curves": curves,
        "map": _snapshot(base.with_overrides(**spec["i"]), spec["s"]),
    }


def _snapshot(config: SimConfig, seed: int, max_points: int = 2500) -> dict:
    """Final state of one intervention run, for the map."""
    sim = Simulation(config, seed=seed)
    sim.run()
    s = sim.state
    status = np.where(~s.alive, 2, np.where(s.in_treatment, 1, 0))
    idx = np.arange(config.n_agents)
    if idx.size > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    return {
        "agents": np.column_stack([s.x[idx], s.y[idx], status[idx]]).tolist(),
        "sites": [{"x": t.x, "y": t.y, "kind": t.kind} for t in config.sites],
        "naloxone_radius": config.naloxone_radius,
    }


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
