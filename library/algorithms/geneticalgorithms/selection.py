"""
Selection operators for the Genetic Algorithm.

Copied and adapted from the P09 practice library (identical interface).

Both operators return a deep copy of the selected individual so the population
list is never mutated in place.

Tournament selection is preferred for this project because:
- RMSE values are small floats (typically 30–120); FPS with 1/fitness can
  introduce numerical instability when fitness values are very close.
- Tournament selection pressure is easy to control via tournament_size.
"""

import random
from copy import deepcopy
from library.problems.solution import Solution


def fitness_proportionate_selection(population: list, maximization: bool):
    """
    Roulette-wheel (fitness proportionate) selection.

    For maximization: P(select i) ∝ fitness(i)
    For minimization: P(select i) ∝ 1/fitness(i)

    Parameters
    ----------
    population : list[Solution]
    maximization : bool

    Returns
    -------
    Solution — a deep copy of the selected individual.
    """
    if maximization:
        fitness_values = [ind.fitness() for ind in population]
    else:
        fitness_values = [1.0 / ind.fitness() for ind in population]

    total_fitness = sum(fitness_values)
    r = random.uniform(0, total_fitness)
    cumulative = 0.0
    for ind, f in zip(population, fitness_values):
        cumulative += f
        if r <= cumulative:
            return deepcopy(ind)
    return deepcopy(population[-1])


def tournament_selection(population: list, maximization: bool, tournament_size: int = 2):
    """
    Tournament selection.

    Randomly samples tournament_size individuals and returns the best one.
    Higher tournament_size → more selection pressure toward the best individuals.

    Parameters
    ----------
    population : list[Solution]
    maximization : bool
    tournament_size : int  (default 2)

    Returns
    -------
    Solution — a deep copy of the winner.
    """
    tournament = random.choices(population, k=tournament_size)
    if maximization:
        winner = max(tournament, key=lambda ind: ind.fitness())
    else:
        winner = min(tournament, key=lambda ind: ind.fitness())
    return deepcopy(winner)
