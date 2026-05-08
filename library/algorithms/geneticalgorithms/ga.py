"""
Genetic Algorithm for the Triangle Image Problem.

Structure
---------
genetic_algorithm()   — main GA loop (adapted from P09's genetic_algorithm_tpclass)
get_elite()           — copied verbatim from P09 library
save_run_log()        — saves JSON log + image snapshots to results/
load_all_runs()       — loads all JSON logs for comparison plots
plot_runs()           — convenience plotting function for the notebook

Design notes
------------
- The GA is minimization-only in this project (RMSE) but the maximization
  flag is kept for interface compatibility with the practice library.
- Fitness is evaluated lazily via the cached .fitness() on each Solution;
  the algorithm never calls fitness() more times than necessary.
- Each run is saved to results/run_<timestamp>/ with a log.json and PNG
  snapshots every save_every_n_gens generations.
"""

import os
import json
import time
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ── Elite selection ───────────────────────────────────────────────────────────

def get_elite(population: list, maximization: bool, elitesize: int) -> list:
    """
    Return the top-k individuals from the population sorted by fitness.

    Copied from P09 practice library.

    Parameters
    ----------
    population : list[Solution]
    maximization : bool
    elitesize : int — number of elite individuals to keep (0 = no elitism)

    Returns
    -------
    list[Solution] — the elite individuals (deep copies NOT made; handle in GA loop)
    """
    if elitesize == 0:
        return []
    sorted_pop = sorted(population, key=lambda ind: ind.fitness(), reverse=maximization)
    return sorted_pop[:elitesize]


# ── Main GA loop ───────────────────────────────────────────────────────────────

def genetic_algorithm(
    solution_class,
    population_size: int,
    max_generations: int,
    selection_algorithm,
    xo_method,
    mut_method,
    maximization: bool = False,
    xo_prob: float = 0.8,
    mut_prob: float = 0.05,
    elitesize: int = 1,
    verbose: bool = False,
    run_id: str = None,
    save_every_n_gens: int = 100,
    initial_population: list = None,
    config: dict = None,
    patience: int = None,
    min_delta: float = 0.05,
):
    """
    Run the Genetic Algorithm and optionally save results to disk.

    Parameters
    ----------
    solution_class : class
        The Solution subclass to instantiate for random individuals
        (must implement random_initial_representation and with_repr).
    population_size : int
    max_generations : int
    selection_algorithm : callable — e.g. tournament_selection
    xo_method : callable — e.g. uniform_triangle_crossover
    mut_method : callable — e.g. triangle_mutation
    maximization : bool — False for RMSE minimization
    xo_prob : float — probability of crossover occurring (per pair)
    mut_prob : float — probability of mutating each triangle
    elitesize : int — number of elite individuals carried over unchanged
    verbose : bool — print progress every 50 generations
    run_id : str | None — if provided, saves logs to results/<run_id>/
    save_every_n_gens : int — save PNG snapshot every N generations
    initial_population : list | None — if None, population is randomly initialised
    config : dict | None — hyperparameter dict saved into the JSON log
    patience : int | None — early stopping: number of generations to look back
        when measuring improvement.  None disables early stopping (default).
        If the best fitness has not improved by more than `min_delta` over the
        last `patience` generations, the run stops early.
    min_delta : float — minimum RMSE improvement required over `patience`
        generations to continue (default 0.05).

    Returns
    -------
    best_individual : Solution
    fitness_history : list[float] — best fitness at end of each generation
    """
    # ── Initialise population ────────────────────────────────────────────────
    if initial_population is not None:
        population = list(initial_population)
    else:
        population = [solution_class() for _ in range(population_size)]

    fitness_history = []
    start_time = time.time()
    stopped_early_at = None

    # Set up run logging folder
    run_dir = None
    if run_id is not None:
        run_dir = os.path.join("results", run_id)
        os.makedirs(run_dir, exist_ok=True)

    # ── Generation loop ──────────────────────────────────────────────────────
    for gen in range(max_generations):

        # 1. Elitism — carry over the best individuals unchanged
        elite = get_elite(population, maximization, elitesize)
        new_population = [deepcopy(ind) for ind in elite]

        # 2. Fill the rest of the population with offspring
        while len(new_population) < population_size:
            parent1 = selection_algorithm(population, maximization)
            parent2 = selection_algorithm(population, maximization)

            offspring1, offspring2 = xo_method(parent1, parent2, xo_prob)
            offspring1 = mut_method(offspring1, mut_prob)
            offspring2 = mut_method(offspring2, mut_prob)

            new_population.append(offspring1)
            if len(new_population) < population_size:
                new_population.append(offspring2)

        population = new_population

        # 3. Track best fitness this generation
        best_ind = get_elite(population, maximization, 1)[0]
        best_fitness = best_ind.fitness()
        fitness_history.append(best_fitness)

        # 4. Verbose logging
        if verbose and (gen == 0 or (gen + 1) % 50 == 0):
            elapsed = time.time() - start_time
            print(f"Generation {gen + 1:4d}/{max_generations} | "
                  f"Best RMSE: {best_fitness:.4f} | "
                  f"Elapsed: {elapsed:.1f}s")

        # 5. Save image snapshot
        if run_dir is not None and (gen == 0 or (gen + 1) % save_every_n_gens == 0):
            img_array = best_ind.draw_image().astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(run_dir, f"gen_{gen + 1:04d}.png"))

        # 6. Early stopping — check plateau once we have enough history
        if patience is not None and (gen + 1) > patience:
            improvement = fitness_history[-(patience + 1)] - fitness_history[-1]
            if not maximization:
                plateau = improvement < min_delta
            else:
                plateau = improvement < min_delta
            if plateau:
                stopped_early_at = gen + 1
                if verbose:
                    print(f"\n[Early stop] Plateau detected at generation {stopped_early_at}. "
                          f"Improvement over last {patience} gens: {improvement:.4f} < "
                          f"min_delta={min_delta}")
                # Save final snapshot
                if run_dir is not None:
                    img_array = best_ind.draw_image().astype(np.uint8)
                    Image.fromarray(img_array).save(
                        os.path.join(run_dir, f"gen_{gen + 1:04d}_final.png")
                    )
                break

    # ── Final best individual ────────────────────────────────────────────────
    final_best = get_elite(population, maximization, 1)[0]
    runtime = time.time() - start_time

    # ── Save JSON log ────────────────────────────────────────────────────────
    if run_dir is not None:
        save_run_log(
            run_dir, config or {}, fitness_history, runtime, final_best.fitness(),
            stopped_early_at=stopped_early_at,
        )

    if verbose:
        if stopped_early_at:
            print(f"\nDone (early stop at gen {stopped_early_at}). "
                  f"Best RMSE: {final_best.fitness():.4f} | Total time: {runtime:.1f}s")
        else:
            print(f"\nDone. Best RMSE: {final_best.fitness():.4f} | Total time: {runtime:.1f}s")

    return final_best, fitness_history


# ── Run logging ───────────────────────────────────────────────────────────────

def save_run_log(run_dir: str, config: dict, fitness_history: list,
                 runtime: float, final_fitness: float, stopped_early_at: int = None):
    """Save experiment metadata and fitness history to results/<run_id>/log.json."""
    log = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "runtime_seconds": round(runtime, 2),
        "final_fitness": round(final_fitness, 6),
        "fitness_history": [round(f, 6) for f in fitness_history],
        "n_generations": len(fitness_history),
        "stopped_early_at": stopped_early_at,
    }
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def load_all_runs(results_dir: str = "results") -> list:
    """
    Load all log.json files from results/ and return a list of log dicts.

    Each dict contains 'config', 'fitness_history', 'final_fitness', etc.
    Use this in the notebook to compare runs from different hyperparameter
    configurations.
    """
    logs = []
    if not os.path.isdir(results_dir):
        return logs
    for run_name in sorted(os.listdir(results_dir)):
        log_path = os.path.join(results_dir, run_name, "log.json")
        if os.path.isfile(log_path):
            with open(log_path) as f:
                data = json.load(f)
            data["run_name"] = run_name
            logs.append(data)
    return logs


# ── Plateau analysis ─────────────────────────────────────────────────────────

def find_plateau(fitness_history: list, patience: int, min_delta: float) -> int | None:
    """
    Retrospectively scan a completed fitness history and return the generation
    at which early stopping WOULD have triggered for given (patience, min_delta).

    Useful for calibrating stopping parameters without re-running the GA.

    Parameters
    ----------
    fitness_history : list[float] — one value per generation (from log.json)
    patience : int — look-back window in generations
    min_delta : float — minimum improvement required over that window

    Returns
    -------
    int — generation number (1-indexed) where stopping would fire, or
    None if the run would never have stopped early.
    """
    for g in range(patience, len(fitness_history)):
        improvement = fitness_history[g - patience] - fitness_history[g]
        if improvement < min_delta:
            return g + 1   # 1-indexed generation number
    return None


def scan_patience_grid(
    fitness_history: list,
    patience_values: list,
    min_delta_values: list,
) -> list:
    """
    Evaluate a grid of (patience, min_delta) combinations on one fitness history.

    Returns a list of dicts with keys: patience, min_delta, stop_gen, rmse_at_stop,
    final_rmse, gens_saved, pct_saved.  Useful for choosing stopping parameters
    before running new experiments.
    """
    final_rmse = fitness_history[-1]
    total_gens = len(fitness_history)
    results = []
    for pat in patience_values:
        for delta in min_delta_values:
            stop_gen = find_plateau(fitness_history, pat, delta)
            if stop_gen is not None:
                rmse_at_stop = fitness_history[stop_gen - 1]
                gens_saved = total_gens - stop_gen
                pct_saved = 100 * gens_saved / total_gens
            else:
                rmse_at_stop = final_rmse
                gens_saved = 0
                pct_saved = 0.0
            results.append({
                "patience": pat,
                "min_delta": delta,
                "stop_gen": stop_gen,
                "rmse_at_stop": round(rmse_at_stop, 4) if stop_gen else None,
                "final_rmse": round(final_rmse, 4),
                "gens_saved": gens_saved,
                "pct_saved": round(pct_saved, 1),
            })
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_fitness_over_generations(
    fitness_histories: list,
    labels: list = None,
    title: str = "Fitness over Generations",
    figsize: tuple = (10, 5),
):
    """
    Plot one or more fitness curves on the same axes.

    Parameters
    ----------
    fitness_histories : list[list[float]]
        Each inner list is the fitness_history from one GA run.
    labels : list[str] | None
        Legend labels (one per history).
    title : str
    figsize : tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = labels or [f"Run {i+1}" for i in range(len(fitness_histories))]

    for history, label in zip(fitness_histories, labels):
        ax.plot(history, label=label, linewidth=1.5)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best RMSE (lower is better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_image_evolution(run_dir: str, figsize: tuple = (16, 4)):
    """
    Display the saved PNG snapshots from a run in a single row.

    Parameters
    ----------
    run_dir : str — path to results/<run_id>/
    """
    snapshots = sorted(
        [f for f in os.listdir(run_dir) if f.endswith(".png")]
    )
    if not snapshots:
        print(f"No PNG snapshots found in {run_dir}")
        return

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, fname in zip(axes, snapshots):
        img = Image.open(os.path.join(run_dir, fname))
        gen_num = fname.replace("gen_", "").replace(".png", "")
        ax.imshow(img)
        ax.set_title(f"Gen {int(gen_num)}")
        ax.axis("off")

    plt.suptitle(f"Image evolution — {run_dir}", fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig
