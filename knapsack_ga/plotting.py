"""Plots of the best fitness over the iterations of the algorithm."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from knapsack_ga.experiments import ExperimentResult  # noqa: E402

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def plot_experiment(result: ExperimentResult, output_path: Path) -> None:
    """Plots the mean best fitness per iteration of every compared series.

    The known optimum of the problem is drawn as a dashed reference line.

    Args:
        result: Results of the experiment to plot.
        output_path: Path of the PNG file to create.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for color, series_result in zip(SERIES_COLORS, result.series_results):
        ax.plot(
            series_result.mean_history,
            color=color,
            linewidth=2,
            label=series_result.series.label,
        )

    if result.problem.optimum is not None:
        ax.axhline(
            result.problem.optimum,
            color=MUTED_COLOR,
            linewidth=1.2,
            linestyle="--",
            label=f"optimum = {result.problem.optimum:g}",
        )

    _style_axes(ax)
    ax.set_title(
        f"{result.problem.name}: {result.experiment.title.lower()}",
        color=TEXT_COLOR,
        fontsize=13,
        loc="left",
    )
    ax.set_xlabel("Iteration", color=MUTED_COLOR)
    ax.set_ylabel("Best fitness (mean of runs)", color=MUTED_COLOR)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    """Applies a clean, recessive style to the axes.

    Args:
        ax: Axes to style.
    """
    ax.margins(x=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR)
