"""Crossover operators.

Every crossover function takes two parent chromosomes and returns two children.
"""

import numpy as np


def one_point_crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Exchanges the genes of two parents after one random cut point.

    Args:
        parent_a: First parent chromosome.
        parent_b: Second parent chromosome.
        rng: Random number generator.

    Returns:
        Two child chromosomes.
    """
    point = rng.integers(1, len(parent_a))
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    return child_a, child_b


def two_point_crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Exchanges the genes of two parents between two random cut points.

    Args:
        parent_a: First parent chromosome.
        parent_b: Second parent chromosome.
        rng: Random number generator.

    Returns:
        Two child chromosomes.
    """
    start, end = np.sort(rng.choice(np.arange(1, len(parent_a)), size=2, replace=False))
    child_a, child_b = parent_a.copy(), parent_b.copy()
    child_a[start:end], child_b[start:end] = parent_b[start:end], parent_a[start:end]
    return child_a, child_b
