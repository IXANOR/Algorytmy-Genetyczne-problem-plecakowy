"""Loading knapsack problem instances from text files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KnapsackProblem:
    """A single instance of the 0/1 knapsack problem.

    Attributes:
        name: Name of the instance (usually the data file name).
        values: Value of every item.
        weights: Weight of every item.
        capacity: Maximum total weight the knapsack can hold.
        optimum: Known optimal total value, or None if it is unknown.
    """

    name: str
    values: np.ndarray
    weights: np.ndarray
    capacity: float
    optimum: float | None = None

    @property
    def n_items(self) -> int:
        """Returns the number of items in the instance."""
        return len(self.values)


def load_problem(path: Path, optimum_path: Path | None = None) -> KnapsackProblem:
    """Loads a knapsack problem instance from a text file.

    The first line of the file contains the number of items and the knapsack
    capacity. Every following line describes one item as ``value weight``.
    Lines after the declared number of items are ignored.

    Args:
        path: Path to the file with the problem definition.
        optimum_path: Optional path to a file containing the known optimum.

    Returns:
        The loaded problem instance.

    Raises:
        ValueError: If the file contains fewer items than declared.
    """
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    n_items, capacity = int(lines[0][0]), float(lines[0][1])
    items = lines[1 : n_items + 1]
    if len(items) != n_items:
        raise ValueError(f"{path.name}: expected {n_items} items, found {len(items)}")

    data = np.array(items, dtype=float)
    optimum = load_optimum(optimum_path) if optimum_path else None
    return KnapsackProblem(
        name=path.name,
        values=data[:, 0],
        weights=data[:, 1],
        capacity=capacity,
        optimum=optimum,
    )


def load_optimum(path: Path) -> float:
    """Loads the known optimal value of a problem instance.

    Args:
        path: Path to the file whose first token is the optimal value.

    Returns:
        The optimal total value.
    """
    return float(path.read_text().split()[0])


def load_dataset_directory(data_dir: Path, optimum_dir: Path) -> list[KnapsackProblem]:
    """Loads all problem instances from a directory.

    Every instance is paired with the optimum file of the same name, if such a
    file exists in ``optimum_dir``.

    Args:
        data_dir: Directory containing the problem files.
        optimum_dir: Directory containing the optimum files.

    Returns:
        Loaded problems sorted by the number of items, then by name.
    """
    problems = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        optimum_path = optimum_dir / path.name
        problems.append(load_problem(path, optimum_path if optimum_path.exists() else None))
    return sorted(problems, key=lambda problem: (problem.n_items, problem.name))
