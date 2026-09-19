# Genetic algorithm for the 0/1 knapsack problem

Project for the course *Algorytmy genetyczne i sztuczne sieci neuronowe*.

Authors: Igor Podlecki (56088), Michał Perczak (56085)

## Features

- loading knapsack instances and their known optima from text files,
- a general genetic algorithm parameterized with population size, number of iterations,
  crossover and mutation rates, elitism and the fitness, selection, crossover and mutation
  functions,
- knapsack fitness function (value of the packed items, or 0 when the knapsack is overloaded),
- roulette, rank and tournament selection,
- one-point and two-point crossover,
- bit-flip mutation,
- experiments comparing mutation rates, crossover rates, selection methods, crossover methods
  and elitism, with a plot of the best fitness per iteration for every dataset.

## Project structure

```
knapsack_ga/
    data_loader.py        loading problem instances
    fitness.py            fitness function
    selection.py          roulette, rank and tournament selection
    crossover.py          one-point and two-point crossover
    mutation.py           bit-flip mutation
    genetic_algorithm.py  evolution loop
    experiments.py        experiment definitions
    plotting.py           plots
    reporting.py          CSV summaries
tests/                    unit tests
dane AG/                  datasets and their optima
results/                  generated plots and CSV files
main.py                   command line interface
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all experiments on every dataset (the largest instances take the most time):

```bash
make run                         # or: python main.py experiments
```

Run the experiments on a few chosen datasets only:

```bash
make run-quick
python main.py experiments --datasets f1_l-d_kp_10_269 knapPI_1_100_1000_1 --runs 5
```

Solve a single instance with custom parameters:

```bash
python main.py solve "dane AG/low-dimensional/f8_l-d_kp_23_10000" \
    --optimum "dane AG/low-dimensional-optimum/f8_l-d_kp_23_10000" \
    --selection tournament --crossover two-point --generations 300 --elitism 1 --seed 1
```

Run the unit tests:

```bash
make test                        # or: python -m unittest discover -s tests
```

`python main.py experiments --help` and `python main.py solve --help` list all options.

## Experiments

Every experiment starts from a reference configuration and changes one aspect of it:

| parameter        | reference value                                          |
|------------------|----------------------------------------------------------|
| population size  | 100                                                      |
| iterations       | 100 (fewer than 100 items), 1000 (100 items or more)     |
| crossover rate   | 0.8                                                      |
| mutation rate    | `min(0.01, 1 / n_items)`                                 |
| elitism          | 1                                                        |
| selection        | roulette                                                 |
| crossover        | one-point                                                |

Compared variants:

- mutation rate: 0.1, 0.5, 1, 5 and 10 times the reference value,
- crossover rate: 0.5, 0.7, 0.9, 1.0,
- selection: roulette, rank, tournament (3 contestants),
- crossover: one-point, two-point,
- elitism: 0, 1.

Every configuration is run 10 times with seeds 0-9, and the plots show the best fitness in
the population after each iteration, averaged over the runs.

## Results

- `results/plots/<dataset>/<experiment>.png`: plots for every dataset and experiment,
- `results/summary.csv`: statistics of every configuration (best, mean and standard deviation
  of the best fitness, percentage of the optimum, number of runs that reached the optimum),
- `results/best_results.csv`: the best solution found for every dataset.
