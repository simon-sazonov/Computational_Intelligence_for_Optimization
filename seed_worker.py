"""
Top-level worker for parallel seed execution.
Must live in a real module file (not the notebook) so it is picklable
on macOS, which uses the 'spawn' multiprocessing start method.
"""
import os, sys, random
import numpy as np

# Ensure the project root is on sys.path regardless of where the worker spawns
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one_seed(seed, ga_cfg, sa_cfg):
    """
    Run GA + SA for one seed. Returns:
        (seed, ga_rmse, sa_rmse, ga_fitness_history, sa_fitness_history)
    """
    sys.path.insert(0, _PROJECT_DIR)
    os.chdir(_PROJECT_DIR)

    from functools import partial
    from library.problems.triangle_image import TriangleImageSolution
    from library.algorithms.geneticalgorithms.ga import (
        load_or_run_ga, load_run_log, build_logged_solution,
    )
    from library.algorithms.geneticalgorithms.selection import tournament_selection
    from library.algorithms.geneticalgorithms.crossover import (
        uniform_triangle_crossover, single_point_triangle_crossover,
    )
    from library.algorithms.geneticalgorithms.mutation import triangle_mutation
    from library.algorithms.simulated_annealing import simulated_annealing

    # Each spawned worker process has a blank class state — reload the target image.
    TriangleImageSolution.load_target(
        os.path.join(_PROJECT_DIR, 'girl_pearl_earing.png')
    )

    random.seed(seed)
    np.random.seed(seed)

    # ── GA ──────────────────────────────────────────────────────────────────
    xo = (single_point_triangle_crossover
          if ga_cfg.get('crossover_type') == 'single_point'
          else uniform_triangle_crossover)
    sel = partial(tournament_selection,
                  tournament_size=ga_cfg.get('tournament_size', 3))

    best_ga, stats_ga, _, _, _ = load_or_run_ga(
        f'challenge2_ga_seed{seed}',
        solution_class=TriangleImageSolution,
        cfg=ga_cfg,
        selection_fn=sel, xo_fn=xo, mut_fn=triangle_mutation,
        verbose=False,
    )
    ga_rmse = best_ga.fitness()
    ga_history = stats_ga['fitness_history']

    # ── SA ──────────────────────────────────────────────────────────────────
    label_sa = f'challenge2_sa_seed{seed}'
    if not os.path.isdir(os.path.join('results', label_sa)):
        initial_sa = TriangleImageSolution()
        random.seed(seed)
        np.random.seed(seed)
        best_sa, hist_sa = simulated_annealing(
            initial_solution=initial_sa,
            C=sa_cfg['C'], L=sa_cfg['L'], H=sa_cfg['H'],
            maximization=False, max_iter=sa_cfg['max_iter'],
            run_id=label_sa,
            save_every_n_steps=sa_cfg['save_every_n_steps'],
            config=sa_cfg,
        )
    else:
        log_sa = load_run_log(label_sa)
        best_sa = build_logged_solution(log_sa, TriangleImageSolution)
        hist_sa = list(log_sa.get('fitness_history', []))

    sa_rmse = best_sa.fitness()
    sa_history = hist_sa

    print(f"  [worker] Seed {seed}: GA={ga_rmse:.4f}, SA={sa_rmse:.4f}", flush=True)
    return seed, ga_rmse, sa_rmse, ga_history, sa_history


def run_val_seed(seed, ga_cfg):
    """
    Run GA-only validation for one seed (Section 10).
    Uses seed offset +100 to stay independent from Section 9 runs.
    Returns: (seed, rmse, fitness_history)
    """
    sys.path.insert(0, _PROJECT_DIR)
    os.chdir(_PROJECT_DIR)

    from functools import partial
    from library.problems.triangle_image import TriangleImageSolution
    from library.algorithms.geneticalgorithms.ga import (
        load_or_run_ga, load_run_log, build_logged_solution,
    )
    from library.algorithms.geneticalgorithms.selection import tournament_selection
    from library.algorithms.geneticalgorithms.crossover import (
        uniform_triangle_crossover, single_point_triangle_crossover,
    )
    from library.algorithms.geneticalgorithms.mutation import triangle_mutation

    TriangleImageSolution.load_target(
        os.path.join(_PROJECT_DIR, 'girl_pearl_earing.png')
    )

    actual_seed = seed + 100
    random.seed(actual_seed)
    np.random.seed(actual_seed)

    xo = (single_point_triangle_crossover
          if ga_cfg.get('crossover_type') == 'single_point'
          else uniform_triangle_crossover)
    sel = partial(tournament_selection,
                  tournament_size=ga_cfg.get('tournament_size', 3))

    best, stats, _, _, _ = load_or_run_ga(
        f'final_validation_seed{seed}',
        solution_class=TriangleImageSolution,
        cfg=ga_cfg,
        selection_fn=sel, xo_fn=xo, mut_fn=triangle_mutation,
        verbose=False,
    )
    rmse = best.fitness()
    print(f"  [worker] Val seed {seed}: RMSE={rmse:.4f}", flush=True)
    return seed, rmse, stats['fitness_history']
