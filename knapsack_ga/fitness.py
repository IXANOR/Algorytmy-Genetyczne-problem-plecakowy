"""Fitness functions for the knapsack problem."""

import numpy as np

from knapsack_ga.data_loader import KnapsackProblem


def knapsack_fitness(population: np.ndarray, problem: KnapsackProblem) -> np.ndarray:
    """Evaluates every chromosome in the population.

    The fitness of a chromosome is the total value of the packed items if their
    total weight does not exceed the knapsack capacity, otherwise it is 0.

    Args:
        population: Binary matrix of shape ``(population_size, n_items)``.
        problem: Knapsack problem instance.

    Returns:
        Fitness value of every chromosome.
    """
    total_values = population @ problem.values
    total_weights = population @ problem.weights
    return np.where(total_weights <= problem.capacity, total_values, 0.0)
