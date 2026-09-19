"""Saving experiment results to CSV files."""

import csv
from pathlib import Path

import numpy as np

from knapsack_ga.experiments import ExperimentResult

OPTIMUM_TOLERANCE = 1e-6
SUMMARY_COLUMNS = (
    "dataset",
    "n_items",
    "capacity",
    "optimum",
    "experiment",
    "series",
    "runs",
    "best",
    "mean_best",
    "std_best",
    "best_to_optimum_pct",
    "mean_to_optimum_pct",
    "runs_reaching_optimum",
)
BEST_RESULTS_COLUMNS = (
    "dataset",
    "n_items",
    "optimum",
    "best",
    "best_to_optimum_pct",
    "experiment",
    "series",
)


def summarize(results: list[ExperimentResult]) -> list[dict]:
    """Computes statistics of every series of every experiment.

    Args:
        results: Results of the experiments.

    Returns:
        One row per series, with keys listed in ``SUMMARY_COLUMNS``.
    """
    rows = []
    for result in results:
        problem = result.problem
        for series_result in result.series_results:
            best_fitnesses = series_result.best_fitnesses
            rows.append(
                {
                    "dataset": problem.name,
                    "n_items": problem.n_items,
                    "capacity": problem.capacity,
                    "optimum": problem.optimum,
                    "experiment": result.experiment.name,
                    "series": series_result.series.label,
                    "runs": len(best_fitnesses),
                    "best": best_fitnesses.max(),
                    "mean_best": round(best_fitnesses.mean(), 2),
                    "std_best": round(best_fitnesses.std(), 2),
                    "best_to_optimum_pct": _percent(best_fitnesses.max(), problem.optimum),
                    "mean_to_optimum_pct": _percent(best_fitnesses.mean(), problem.optimum),
                    "runs_reaching_optimum": _count_optimal(best_fitnesses, problem.optimum),
                }
            )
    return rows


def best_results(summary: list[dict]) -> list[dict]:
    """Picks the best result found for every dataset.

    Ties are resolved in favor of the series with the higher mean result.

    Args:
        summary: Rows returned by ``summarize``.

    Returns:
        One row per dataset, with keys listed in ``BEST_RESULTS_COLUMNS``.
    """
    best_by_dataset: dict[str, dict] = {}
    for row in summary:
        current = best_by_dataset.get(row["dataset"])
        is_better = current is None or (row["best"], row["mean_best"]) > (
            current["best"],
            current["mean_best"],
        )
        if is_better:
            best_by_dataset[row["dataset"]] = row
    return [
        {column: row[column] for column in BEST_RESULTS_COLUMNS}
        for row in best_by_dataset.values()
    ]


def write_csv(rows: list[dict], columns: tuple[str, ...], path: Path) -> None:
    """Writes rows to a CSV file.

    Args:
        rows: Rows to write.
        columns: Column names, in order.
        path: Path of the CSV file to create.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: float, optimum: float | None) -> float | None:
    """Returns ``value`` as a percentage of ``optimum``, if it is known."""
    return round(100 * value / optimum, 2) if optimum else None


def _count_optimal(best_fitnesses: np.ndarray, optimum: float | None) -> int | None:
    """Returns how many runs reached the optimum, if it is known.

    A relative tolerance is used, because some optima are stored rounded.
    """
    if not optimum:
        return None
    return int((best_fitnesses >= optimum * (1 - OPTIMUM_TOLERANCE)).sum())
