"""Selection operators.

Every selection function takes the fitness values of the population and returns
the indices of the chosen parents. The same individual may be chosen many times.
"""

import numpy as np


def roulette_selection(fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Selects parents with the roulette wheel method.

    The probability of choosing an individual is proportional to its fitness.
    When every individual has zero fitness, all of them are equally likely.

    Args:
        fitness: Fitness value of every individual.
        rng: Random number generator.

    Returns:
        Indices of the selected individuals, one per population slot.
    """
    total = fitness.sum()
    if total == 0:
        probabilities = np.full(len(fitness), 1 / len(fitness))
    else:
        probabilities = fitness / total
    return rng.choice(len(fitness), size=len(fitness), p=probabilities)


def rank_selection(fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Selects parents with the linear rank method.

    Individuals are sorted by fitness and receive ranks from 1 (the worst) to
    N (the best). The probability of choosing an individual is proportional to
    its rank, so it does not depend on the scale of fitness values.

    Args:
        fitness: Fitness value of every individual.
        rng: Random number generator.

    Returns:
        Indices of the selected individuals, one per population slot.
    """
    ranks = np.empty(len(fitness))
    ranks[np.argsort(fitness)] = np.arange(1, len(fitness) + 1)
    probabilities = ranks / ranks.sum()
    return rng.choice(len(fitness), size=len(fitness), p=probabilities)


def tournament_selection(
    fitness: np.ndarray, rng: np.random.Generator, tournament_size: int = 3
) -> np.ndarray:
    """Selects parents with the tournament method.

    For every population slot a group of ``tournament_size`` individuals is
    drawn at random and the best of them is selected.

    Args:
        fitness: Fitness value of every individual.
        rng: Random number generator.
        tournament_size: Number of individuals competing in one tournament.

    Returns:
        Indices of the selected individuals, one per population slot.
    """
    contestants = rng.integers(0, len(fitness), size=(len(fitness), tournament_size))
    winners = np.argmax(fitness[contestants], axis=1)
    return contestants[np.arange(len(fitness)), winners]
