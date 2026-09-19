"""Unit tests of the genetic algorithm components."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from knapsack_ga.crossover import one_point_crossover, two_point_crossover
from knapsack_ga.data_loader import KnapsackProblem, load_problem
from knapsack_ga.fitness import knapsack_fitness
from knapsack_ga.genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig
from knapsack_ga.mutation import bit_flip_mutation
from knapsack_ga.selection import rank_selection, roulette_selection, tournament_selection

PROBLEM = KnapsackProblem(
    name="test",
    values=np.array([10.0, 20.0, 30.0]),
    weights=np.array([5.0, 10.0, 15.0]),
    capacity=20.0,
    optimum=40.0,
)


class DataLoaderTest(unittest.TestCase):
    """Tests of loading problem instances from text files."""

    def test_loads_items_capacity_and_optimum(self) -> None:
        """Checks that every part of the problem is read from the files."""
        with tempfile.TemporaryDirectory() as directory:
            problem_path = Path(directory) / "problem"
            optimum_path = Path(directory) / "optimum"
            problem_path.write_text("2 10\n5 3\n7.5 4\n")
            optimum_path.write_text("12.5")

            problem = load_problem(problem_path, optimum_path)

        self.assertEqual(problem.n_items, 2)
        self.assertEqual(problem.capacity, 10)
        self.assertEqual(problem.optimum, 12.5)
        np.testing.assert_array_equal(problem.values, [5, 7.5])
        np.testing.assert_array_equal(problem.weights, [3, 4])


class FitnessTest(unittest.TestCase):
    """Tests of the knapsack fitness function."""

    def test_returns_value_of_feasible_and_zero_of_overweight_solutions(self) -> None:
        """Checks the fitness of a feasible and an overweight chromosome."""
        population = np.array([[1, 0, 1], [1, 1, 1]])

        fitness = knapsack_fitness(population, PROBLEM)

        np.testing.assert_array_equal(fitness, [40, 0])


class SelectionTest(unittest.TestCase):
    """Tests of the selection operators."""

    def test_selections_never_choose_individuals_without_chance(self) -> None:
        """Checks that roulette ignores zero fitness and ranking the worst one."""
        fitness = np.array([0.0, 5.0, 10.0])
        rng = np.random.default_rng(0)

        self.assertNotIn(0, roulette_selection(fitness, rng))
        self.assertEqual(len(rank_selection(fitness, rng)), len(fitness))

    def test_roulette_is_uniform_when_all_fitness_is_zero(self) -> None:
        """Checks that roulette works when no individual is feasible."""
        selected = roulette_selection(np.zeros(4), np.random.default_rng(0))

        self.assertEqual(len(selected), 4)

    def test_tournament_of_whole_population_always_chooses_the_best(self) -> None:
        """Checks that a tournament returns the best contestant."""
        fitness = np.array([1.0, 2.0, 3.0])

        selected = tournament_selection(fitness, np.random.default_rng(0), tournament_size=50)

        np.testing.assert_array_equal(selected, [2, 2, 2])


class CrossoverTest(unittest.TestCase):
    """Tests of the crossover operators."""

    def test_children_exchange_genes_of_parents(self) -> None:
        """Checks that children keep all genes of the parents at every locus."""
        parent_a, parent_b = np.zeros(10, dtype=np.int8), np.ones(10, dtype=np.int8)
        rng = np.random.default_rng(0)

        for crossover in (one_point_crossover, two_point_crossover):
            child_a, child_b = crossover(parent_a, parent_b, rng)
            np.testing.assert_array_equal(child_a + child_b, np.ones(10))
            self.assertTrue(0 < child_a.sum() < 10)


class MutationTest(unittest.TestCase):
    """Tests of the mutation operator."""

    def test_mutation_rate_controls_flipped_genes(self) -> None:
        """Checks the two extreme mutation rates."""
        population = np.array([[0, 1, 0, 1]], dtype=np.int8)
        rng = np.random.default_rng(0)

        np.testing.assert_array_equal(bit_flip_mutation(population, 0.0, rng), population)
        np.testing.assert_array_equal(bit_flip_mutation(population, 1.0, rng), 1 - population)


class GeneticAlgorithmTest(unittest.TestCase):
    """Tests of the whole evolution."""

    def test_finds_optimum_of_small_problem(self) -> None:
        """Checks that the algorithm solves a trivial instance."""
        config = GeneticAlgorithmConfig(population_size=20, generations=20)

        result = GeneticAlgorithm(PROBLEM, config, seed=0).run()

        self.assertEqual(result.best_fitness, PROBLEM.optimum)
        self.assertEqual(len(result.best_fitness_history), config.generations + 1)


if __name__ == "__main__":
    unittest.main()
