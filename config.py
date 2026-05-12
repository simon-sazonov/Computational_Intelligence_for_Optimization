"""
Central configuration file for all hyperparameters.

Change values here before running the notebook — no need to touch algorithm code.
Each run automatically saves these values into its results/run_<id>/log.json
so you can always reconstruct exactly which config produced which result.

Experiment guide
----------------
To compare two configs:
  1. Edit GA_CONFIG to the first config and run the GA cell in the notebook.
  2. Edit GA_CONFIG to the second config and run again.
  Both runs are saved to different results/ folders and load_all_runs()
  will return both for comparison plotting.

Recommended experiments (vary ONE parameter at a time, keep others at baseline):

  population_size   : [10, 20, 30, 50, 100]
  elitesize         : [0, 1, 2, 5]
  mut_prob          : [0.01, 0.03, 0.05, 0.1, 0.2]
  xo_prob           : [0.5, 0.7, 0.9]
  crossover_type    : ["uniform", "single_point"]
  mutation_type     : ["vertex_shift", "replace", "random"]
  tournament_size   : [2, 3, 5]

  SA C (temperature): [1.0, 5.0, 20.0, 100.0]
  SA H (cooling)    : [1.02, 1.05, 1.1, 1.2]
  SA L (inner steps): [10, 50, 100, 200]
"""

import time

# ── GA baseline configuration (used in Section 5) ─────────────────────────────
GA_CONFIG = {
    "population_size"   : 30,
    "max_generations"   : 1000,
    "xo_prob"           : 0.8,
    "mut_prob"          : 0.05,
    "elitesize"         : 2,
    "tournament_size"   : 3,
    "crossover_type"    : "uniform",      # "uniform" | "single_point"
    "mutation_type"     : "random",       # "vertex_shift" | "color_shift" | "alpha_shift"
                                          # | "triangle_replace" | "triangle_swap" | "random"
    "maximization"      : False,          # RMSE is minimised
    "save_image_every_n_gens": 100,       # save PNG snapshot every N generations
}

# ── Best configuration (OAT experiments → Section 9 extended run) ─────────────
# pop=50: more diversity, better recombination
# elite=1: prevents backsliding without reducing population diversity
# single_point XO: preserves z-order co-adaptation between triangles
# mut_prob=0.05: confirmed sweet spot in Experiment C
BEST_CONFIG = {
    "population_size"   : 50,
    "max_generations"   : 3000,
    "xo_prob"           : 0.8,
    "mut_prob"          : 0.05,
    "elitesize"         : 1,
    "tournament_size"   : 3,
    "crossover_type"    : "single_point",
    "mutation_type"     : "random",
    "maximization"      : False,
    "save_image_every_n_gens": 300,
}

# ── SA configuration (for Challenge 2 comparison) ─────────────────────────────
# Equal evaluation budget: GA uses population_size × max_generations evaluations.
# SA uses L × max_iter evaluations in its inner loop.
# Default: 30 × 1000 = 30,000 for GA  →  L=150 × max_iter=200 = 30,000 for SA.
SA_CONFIG = {
    "C"                 : 10.0,   # initial temperature
    "L"                 : 150,    # inner iterations per temperature cycle
    "H"                 : 1.05,   # cooling rate (C = C/H each outer loop)
    "max_iter"          : 200,    # outer loops  → total inner steps = L × max_iter
    "maximization"      : False,
    "save_every_n_steps": 1500,   # save PNG snapshot every N inner steps
}


def make_run_id(prefix: str = "run") -> str:
    """Generate a unique run ID based on current timestamp."""
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


# ── Optuna search space (populated by Section 6 range-derivation cell) ────────
# Each entry has type ("int" | "float" | "categorical") and bounds/choices.
# The numeric bounds (low/high) are None until the Section 6 closing cell runs
# and derives them from the OAT experiment results automatically.
OPTUNA_SEARCH_SPACE =  { "population_size": {"type": "int",         "low": 10, "high": 50},
    "tournament_size": {"type": "int",         "low": 2, "high": 6},
    "mut_prob":        {"type": "float",       "low": 0.01, "high": 0.1},
    "xo_prob":         {"type": "float",       "low": 0.5, "high": 1.0},
    "elitesize":       {"type": "int",         "low": 0, "high": 5},
    "crossover_type":  {"type": "categorical", "choices": ["uniform", "single_point"]},
    "mutation_type":   {"type": "categorical", "choices": ["random", "vertex_shift",
                                                            "color_shift", "alpha_shift"]},
}

# ── Optuna-optimized configuration (filled in by Section 7 notebook cell) ─────
# Replaces BEST_CONFIG as the canonical best parameters for all later sections.
OPTUNA_BEST_CONFIG = {
    "population_size"        : 40,
    "max_generations"        : 1000,
    "xo_prob"                : 0.8557,
    "mut_prob"               : 0.0518,
    "elitesize"              : 0,
    "tournament_size"        : 5,
    "crossover_type"         : "uniform",
    "mutation_type"          : "random",
    "maximization"           : False,
    "save_image_every_n_gens": 100,
}
