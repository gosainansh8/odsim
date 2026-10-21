"""Figures and a self-contained HTML report for scenario comparisons.

Requires matplotlib: ``pip install 'odsim[plots]'``.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

    from .experiments import Comparison
    from .config import SimConfig
    from .cox import CoxResult

# Categorical slots in fixed order (validated for colorblind separation).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install 'odsim[plots]'") from exc
    plt.rcParams.update(
        {
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
    return plt


def plot_survival(comparison: Comparison) -> Figure:
    """Kaplan-Meier curves, one line per scenario."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    horizon = next(iter(comparison.pooled.values())).n_steps
    lowest = 1.0
    for i, (name, km) in enumerate(comparison.curves.items()):
        t = np.concatenate([[0], km.times, [horizon]])
        s = np.concatenate([[1.0], km.survival, [km.survival[-1] if km.survival.size else 1.0]])
        lowest = min(lowest, s.min())
        ax.step(t, s, where="post", color=SERIES[i % len(SERIES)], linewidth=2, label=name)
    ax.set_xlim(0, horizon)
    ax.set_ylim(max(0.0, lowest - 0.02), 1.002)
    ax.set_xlabel("Day")
    ax.set_ylabel("Proportion surviving")
    ax.set_title("Survival by scenario (Kaplan-Meier)", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_hazard_ratios(comparison: Comparison) -> Figure:
    """Forest plot: hazard ratio of death vs baseline, with 95% CIs."""
    plt = _pyplot()
    table = comparison.table[comparison.table["scenario"] != comparison.baseline]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(table) + 1.4))
    y = np.arange(len(table))[::-1]
    names = list(comparison.table["scenario"])
    for yi, (_, row) in zip(y, table.iterrows()):
        color = SERIES[names.index(row["scenario"]) % len(SERIES)]
        ax.plot([row["hr_ci_low"], row["hr_ci_high"]], [yi, yi], color=color, linewidth=2)
        ax.plot(row["hazard_ratio"], yi, "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=2)  # fmt: skip
        ax.annotate(f"{row['hazard_ratio']:.2f}", (row["hr_ci_high"], yi),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK_2)  # fmt: skip
    ax.axvline(1.0, color=INK_2, linewidth=1, linestyle="--")
    ax.set_yticks(y, table["scenario"])
    ax.set_xlabel(f"Hazard ratio of death vs {comparison.baseline} (95% CI, < 1 is better)")
    ax.set_title("Intervention effects", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_sweep_heatmap(df: pd.DataFrame, x: str, y: str, metric: str = "mortality") -> Figure:
    """Heatmap of a metric averaged over replicates for a two-parameter sweep."""
    plt = _pyplot()
    grid = df.groupby([y, x])[metric].mean().unstack(x).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(1.1 * grid.shape[1] + 3, 0.7 * grid.shape[0] + 2))
    im = ax.imshow(grid.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [f"{v:g}" for v in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{v:g}" for v in grid.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(False)
    vmax = np.nanmax(grid.to_numpy())
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v > 0.6 * vmax else INK)  # fmt: skip
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{metric} across the sweep", loc="left")
    fig.tight_layout()
    return fig


def fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    _pyplot().close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, na_rep="", float_format=lambda v: f"{v:.4g}")


def build_report(
    config: SimConfig,
    comparison: Comparison,
    housing: CoxResult | None = None,
    reps: int | None = None,
) -> str:
    """A single self-contained HTML file (figures embedded as PNG)."""
    survival = fig_to_base64(plot_survival(comparison))
    forest = fig_to_base64(plot_hazard_ratios(comparison))
    table = comparison.table.drop(columns=["logrank_stat"])
    housing_html = ""
    if housing is not None:
        housing_html = f"""
<h2>Housing and mortality ({html.escape(comparison.baseline)})</h2>
<p>Cox model of time to death with an indicator for being unhoused.</p>
{_table_html(housing.to_frame())}"""
    params = pd.DataFrame(
        [{"parameter": k, "value": v} for k, v in config.to_dict().items() if k != "sites"]
    )
    n_sites = len(config.sites)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>odsim report</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; color: {INK}; background: #fcfcfb;
         max-width: 920px; margin: 0 auto; padding: 24px 16px; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }} h2 {{ font-size: 18px; margin-top: 32px; }}
  .meta {{ color: {INK_2}; }} img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; font-size: 13px; width: 100%; overflow-x: auto; display: block; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid {GRID}; text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: {INK_2}; font-weight: 600; }}
</style></head><body>
<h1>Intervention comparison</h1>
<p class="meta">Generated {date.today().isoformat()} by odsim. {config.n_agents:,} agents,
{config.n_steps} days, {n_sites} sites{f", {reps} replicates per scenario" if reps else ""}.
Parameters are illustrative, not calibrated to real data.</p>
<h2>Survival</h2>
<img alt="Kaplan-Meier survival curves by scenario" src="data:image/png;base64,{survival}">
<h2>Hazard ratios</h2>
<img alt="Forest plot of hazard ratios vs baseline" src="data:image/png;base64,{forest}">
<h2>Summary</h2>
{_table_html(table)}
{housing_html}
<h2>Base parameters</h2>
{_table_html(params)}
</body></html>"""
