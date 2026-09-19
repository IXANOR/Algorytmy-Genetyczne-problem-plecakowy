"""Command line interface of the genetic algorithm solving the knapsack problem.

Examples:
    Run all experiments on every dataset::

        python main.py experiments

    Run the experiments on chosen datasets only::

        python main.py experiments --datasets f1_l-d_kp_10_269 knapPI_1_100_1000_1

    Solve a single instance with custom parameters::

        python main.py solve "dane AG/low-dimensional/f1_l-d_kp_10_269" --selection tournament
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

from knapsack_ga.data_loader import KnapsackProblem, load_dataset_directory, load_problem
from knapsack_ga.experiments import (
    CROSSOVERS,
    SELECTIONS,
    base_config,
    build_experiments,
    run_experiment,
)
from knapsack_ga.genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig
from knapsack_ga.plotting import plot_experiment
from knapsack_ga.reporting import (
    BEST_RESULTS_COLUMNS,
    SUMMARY_COLUMNS,
    best_results,
    summarize,
    write_csv,
)

DATA_DIR = Path("dane AG")
DATASET_GROUPS = ("low-dimensional", "large_scale")


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Genetic algorithm for the 0/1 knapsack problem.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiments = subparsers.add_parser("experiments", help="run experiments and save plots")
    experiments.add_argument("--data-dir", type=Path, default=DATA_DIR, help="dataset directory")
    experiments.add_argument(
        "--groups",
        nargs="+",
        choices=DATASET_GROUPS,
        default=list(DATASET_GROUPS),
        help="dataset groups to use",
    )
    experiments.add_argument("--datasets", nargs="+", help="dataset names (default: all)")
    experiments.add_argument("--runs", type=int, default=10, help="runs averaged per configuration")
    experiments.add_argument("--generations", type=int, help="override the number of iterations")
    experiments.add_argument("--output", type=Path, default=Path("results"), help="output dir")

    solve = subparsers.add_parser("solve", help="solve a single problem instance")
    solve.add_argument("problem", type=Path, help="path to the problem file")
    solve.add_argument("--optimum", type=Path, help="path to the file with the known optimum")
    solve.add_argument("--population-size", type=int, default=100)
    solve.add_argument("--generations", type=int, default=200)
    solve.add_argument("--crossover-rate", type=float, default=0.8)
    solve.add_argument("--mutation-rate", type=float, default=0.01)
    solve.add_argument("--selection", choices=SELECTIONS, default="roulette")
    solve.add_argument("--crossover", choices=CROSSOVERS, default="one-point")
    solve.add_argument("--seed", type=int, help="seed of the random number generator")

    return parser.parse_args()


def load_problems(
    data_dir: Path, groups: list[str], names: list[str] | None
) -> list[KnapsackProblem]:
    """Loads the problem instances chosen for the experiments.

    Args:
        data_dir: Directory containing the dataset groups.
        groups: Names of the dataset groups to load.
        names: Names of the datasets to keep, or None to keep all of them.

    Returns:
        The chosen problem instances.

    Raises:
        SystemExit: If some of the requested datasets do not exist.
    """
    problems = []
    for group in groups:
        problems += load_dataset_directory(data_dir / group, data_dir / f"{group}-optimum")
    if names is None:
        return problems

    missing = set(names) - {problem.name for problem in problems}
    if missing:
        raise SystemExit(f"Unknown datasets: {', '.join(sorted(missing))}")
    return [problem for problem in problems if problem.name in names]


def run_experiments(args: argparse.Namespace) -> None:
    """Runs all experiments on the chosen datasets and saves the results.

    Plots are saved to ``<output>/plots/<dataset>/<experiment>.png``, statistics
    of every series to ``<output>/summary.csv`` and the best result for every
    dataset to ``<output>/best_results.csv``.

    Args:
        args: Parsed command line arguments.
    """
    problems = load_problems(args.data_dir, args.groups, args.datasets)
    results = []
    for problem in problems:
        start = time.perf_counter()
        config = base_config(problem)
        if args.generations:
            config = replace(config, generations=args.generations)
        for experiment in build_experiments(config):
            result = run_experiment(problem, experiment, args.runs)
            plot_experiment(result, args.output / "plots" / problem.name / f"{experiment.name}.png")
            results.append(result)
        print(f"{problem.name:<24} done in {time.perf_counter() - start:6.1f} s")

    summary = summarize(results)
    best = best_results(summary)
    write_csv(summary, SUMMARY_COLUMNS, args.output / "summary.csv")
    write_csv(best, BEST_RESULTS_COLUMNS, args.output / "best_results.csv")
    print_best_results(best)


def print_best_results(rows: list[dict]) -> None:
    """Prints the best result found for every dataset.

    Args:
        rows: Rows returned by ``best_results``.
    """
    print(f"\n{'dataset':<24} {'optimum':>10} {'best':>10} {'% opt':>7}  configuration")
    for row in rows:
        print(
            f"{row['dataset']:<24} {row['optimum']:>10g} {row['best']:>10g} "
            f"{row['best_to_optimum_pct']:>7.2f}  {row['experiment']}: {row['series']}"
        )


def solve(args: argparse.Namespace) -> None:
    """Solves a single problem instance and prints the best solution found.

    Args:
        args: Parsed command line arguments.
    """
    problem = load_problem(args.problem, args.optimum)
    config = GeneticAlgorithmConfig(
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        selection=SELECTIONS[args.selection],
        crossover=CROSSOVERS[args.crossover],
    )
    result = GeneticAlgorithm(problem, config, seed=args.seed).run()

    packed_items = [index + 1 for index, gene in enumerate(result.best_solution) if gene]
    print(f"Problem:       {problem.name} ({problem.n_items} items, capacity {problem.capacity:g})")
    print(f"Best fitness:  {result.best_fitness:g}")
    if problem.optimum is not None:
        percent = 100 * result.best_fitness / problem.optimum
        print(f"Optimum:       {problem.optimum:g} ({percent:.2f}%)")
    print(f"Total weight:  {result.best_solution @ problem.weights:g}")
    print(f"Packed items:  {packed_items}")


def main() -> None:
    """Entry point of the program."""
    args = parse_args()
    if args.command == "experiments":
        run_experiments(args)
    else:
        solve(args)


if __name__ == "__main__":
    main()
