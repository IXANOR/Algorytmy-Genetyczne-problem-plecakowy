"""Experiments comparing parameters and operators of the genetic algorithm."""

from dataclasses import dataclass, replace

import numpy as np

from knapsack_ga.crossover import one_point_crossover, two_point_crossover
from knapsack_ga.data_loader import KnapsackProblem
from knapsack_ga.genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig
from knapsack_ga.selection import rank_selection, roulette_selection, tournament_selection

SELECTIONS = {
    "roulette": roulette_selection,
    "rank": rank_selection,
    "tournament": tournament_selection,
}
CROSSOVERS = {
    "one-point": one_point_crossover,
    "two-point": two_point_crossover,
}

MUTATION_RATE_MULTIPLIERS = (0.1, 0.5, 1.0, 5.0, 10.0)
CROSSOVER_RATES = (0.5, 0.7, 0.9, 1.0)
LARGE_PROBLEM_SIZE = 100


@dataclass(frozen=True)
class Series:
    """A single configuration of the algorithm compared within an experiment.

    Attributes:
        label: Name of the series shown in the plot legend.
        config: Parameters of the algorithm.
    """

    label: str
    config: GeneticAlgorithmConfig


@dataclass(frozen=True)
class Experiment:
    """A group of configurations compared on one plot.

    Attributes:
        name: Identifier of the experiment, used in file names.
        title: Human-readable description of the experiment.
        series: Configurations compared in the experiment.
    """

    name: str
    title: str
    series: list[Series]


@dataclass(frozen=True)
class SeriesResult:
    """Results of repeated runs of one configuration.

    Attributes:
        series: The configuration that was run.
        histories: Best fitness per iteration, one row per run.
        best_fitnesses: Best fitness found in every run.
    """

    series: Series
    histories: np.ndarray
    best_fitnesses: np.ndarray

    @property
    def mean_history(self) -> np.ndarray:
        """Returns the best fitness per iteration averaged over all runs."""
        return self.histories.mean(axis=0)


@dataclass(frozen=True)
class ExperimentResult:
    """Results of all configurations of one experiment on one problem.

    Attributes:
        problem: Problem instance the experiment was run on.
        experiment: Definition of the experiment.
        series_results: Results of every configuration.
    """

    problem: KnapsackProblem
    experiment: Experiment
    series_results: list[SeriesResult]


def base_config(problem: KnapsackProblem) -> GeneticAlgorithmConfig:
    """Creates the reference configuration of the algorithm for a problem.

    The reference uses roulette selection and one-point crossover. The mutation
    rate is ``1 / n_items`` (on average one flipped gene per chromosome), capped
    at 0.01. Large problems get more iterations, as they converge slower.

    Args:
        problem: Knapsack problem instance.

    Returns:
        The reference configuration.
    """
    is_large = problem.n_items >= LARGE_PROBLEM_SIZE
    return GeneticAlgorithmConfig(
        population_size=100,
        generations=1000 if is_large else 100,
        crossover_rate=0.8,
        mutation_rate=min(0.01, 1 / problem.n_items),
    )


def build_experiments(base: GeneticAlgorithmConfig) -> list[Experiment]:
    """Builds all experiments by varying one aspect of the base configuration.

    Args:
        base: Reference configuration of the algorithm.

    Returns:
        Experiments comparing mutation rates, crossover rates, selection
        methods and crossover methods.
    """
    mutation_rates = [base.mutation_rate * multiplier for multiplier in MUTATION_RATE_MULTIPLIERS]
    return [
        Experiment(
            name="mutation_rate",
            title="Mutation rate",
            series=[
                Series(f"p_m = {rate:.3g}", replace(base, mutation_rate=rate))
                for rate in mutation_rates
            ],
        ),
        Experiment(
            name="crossover_rate",
            title="Crossover rate",
            series=[
                Series(f"p_c = {rate:g}", replace(base, crossover_rate=rate))
                for rate in CROSSOVER_RATES
            ],
        ),
        Experiment(
            name="selection",
            title="Selection method",
            series=[
                Series(name, replace(base, selection=selection))
                for name, selection in SELECTIONS.items()
            ],
        ),
        Experiment(
            name="crossover",
            title="Crossover method",
            series=[
                Series(name, replace(base, crossover=crossover))
                for name, crossover in CROSSOVERS.items()
            ],
        ),
    ]


def run_series(problem: KnapsackProblem, series: Series, runs: int) -> SeriesResult:
    """Runs one configuration several times with seeds ``0 .. runs - 1``.

    The same seeds are used for every configuration, so all of them start from
    the same initial populations.

    Args:
        problem: Knapsack problem instance.
        series: Configuration to run.
        runs: Number of independent runs.

    Returns:
        Collected results of all runs.
    """
    results = [GeneticAlgorithm(problem, series.config, seed=seed).run() for seed in range(runs)]
    return SeriesResult(
        series=series,
        histories=np.array([result.best_fitness_history for result in results]),
        best_fitnesses=np.array([result.best_fitness for result in results]),
    )


def run_experiment(problem: KnapsackProblem, experiment: Experiment, runs: int) -> ExperimentResult:
    """Runs every configuration of an experiment on a problem.

    Args:
        problem: Knapsack problem instance.
        experiment: Experiment to run.
        runs: Number of independent runs per configuration.

    Returns:
        Results of all configurations.
    """
    return ExperimentResult(
        problem=problem,
        experiment=experiment,
        series_results=[run_series(problem, series, runs) for series in experiment.series],
    )
