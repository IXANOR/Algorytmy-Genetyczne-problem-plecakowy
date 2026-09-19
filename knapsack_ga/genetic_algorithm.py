"""General genetic algorithm evolving a population of binary chromosomes."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from knapsack_ga.crossover import one_point_crossover
from knapsack_ga.data_loader import KnapsackProblem
from knapsack_ga.fitness import knapsack_fitness
from knapsack_ga.mutation import bit_flip_mutation
from knapsack_ga.selection import roulette_selection

FitnessFunction = Callable[[np.ndarray, KnapsackProblem], np.ndarray]
SelectionFunction = Callable[[np.ndarray, np.random.Generator], np.ndarray]
CrossoverFunction = Callable[
    [np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]
]
MutationFunction = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Parameters of the genetic algorithm.

    Attributes:
        population_size: Number of chromosomes in the population.
        generations: Number of iterations of the algorithm.
        crossover_rate: Probability of crossing over a pair of parents.
        mutation_rate: Probability of mutating a single gene.
        fitness: Function evaluating the population.
        selection: Function selecting parents from the population.
        crossover: Function crossing over two parents.
        mutation: Function mutating the population.
    """

    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01
    fitness: FitnessFunction = knapsack_fitness
    selection: SelectionFunction = roulette_selection
    crossover: CrossoverFunction = one_point_crossover
    mutation: MutationFunction = bit_flip_mutation


@dataclass(frozen=True)
class EvolutionResult:
    """Outcome of a single run of the genetic algorithm.

    Attributes:
        best_fitness_history: Fitness of the best chromosome in the population,
            starting with the initial population and then after every iteration.
        best_solution: Best chromosome found during the whole run.
        best_fitness: Fitness of ``best_solution``.
    """

    best_fitness_history: np.ndarray
    best_solution: np.ndarray
    best_fitness: float


class GeneticAlgorithm:
    """Genetic algorithm solving the 0/1 knapsack problem.

    Every chromosome is a binary vector whose i-th gene tells whether the i-th
    item is packed into the knapsack.
    """

    def __init__(
        self,
        problem: KnapsackProblem,
        config: GeneticAlgorithmConfig,
        seed: int | None = None,
    ) -> None:
        """Initializes the algorithm.

        Args:
            problem: Knapsack problem instance to solve.
            config: Parameters of the algorithm.
            seed: Seed of the random number generator, used for reproducibility.
        """
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(seed)

    def run(self) -> EvolutionResult:
        """Runs the evolution for the configured number of iterations.

        Returns:
            History of the best fitness and the best solution found.
        """
        population = self._initial_population()
        fitness = self.config.fitness(population, self.problem)
        history = [fitness.max()]
        best_solution, best_fitness = population[fitness.argmax()], fitness.max()

        for _ in range(self.config.generations):
            population = self._next_generation(population, fitness)
            fitness = self.config.fitness(population, self.problem)
            history.append(fitness.max())
            if fitness.max() > best_fitness:
                best_solution, best_fitness = population[fitness.argmax()], fitness.max()

        return EvolutionResult(
            best_fitness_history=np.array(history),
            best_solution=best_solution,
            best_fitness=float(best_fitness),
        )

    def _initial_population(self) -> np.ndarray:
        """Creates a random initial population.

        Each gene is set to 1 with probability 0.5, lowered to
        ``capacity / total_weight`` for instances where packing half of the
        items would almost always overload the knapsack. Otherwise the whole
        initial population of large instances would have zero fitness.

        Returns:
            Binary matrix of shape ``(population_size, n_items)``.
        """
        gene_probability = min(0.5, self.problem.capacity / self.problem.weights.sum())
        shape = (self.config.population_size, self.problem.n_items)
        return (self.rng.random(shape) < gene_probability).astype(np.int8)

    def _next_generation(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Creates a new population with selection, crossover and mutation.

        Args:
            population: Current population.
            fitness: Fitness of every chromosome in the current population.

        Returns:
            The new population.
        """
        parents = population[self.config.selection(fitness, self.rng)]
        children = self._crossover_population(parents)
        return self.config.mutation(children, self.config.mutation_rate, self.rng)

    def _crossover_population(self, parents: np.ndarray) -> np.ndarray:
        """Crosses over consecutive pairs of parents.

        Each pair is crossed over with probability ``crossover_rate``, otherwise
        the parents are copied unchanged. With an odd population size the last
        parent is always copied.

        Args:
            parents: Selected parent chromosomes.

        Returns:
            Children chromosomes.
        """
        children = parents.copy()
        for i in range(0, len(parents) - 1, 2):
            if self.rng.random() < self.config.crossover_rate:
                children[i], children[i + 1] = self.config.crossover(
                    parents[i], parents[i + 1], self.rng
                )
        return children
