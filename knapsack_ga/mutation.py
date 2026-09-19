"""Mutation operators."""

import numpy as np


def bit_flip_mutation(
    population: np.ndarray, mutation_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Flips every gene of every chromosome with the given probability.

    Args:
        population: Binary matrix of shape ``(population_size, n_items)``.
        mutation_rate: Probability of flipping a single gene.
        rng: Random number generator.

    Returns:
        The mutated population as a new matrix.
    """
    mask = rng.random(population.shape) < mutation_rate
    return np.where(mask, 1 - population, population)
