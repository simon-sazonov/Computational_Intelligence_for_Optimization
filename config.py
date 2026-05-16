"""
Central configuration — single source of truth for all hyperparameters.

The pipeline flows left to right through this file:
  GA_CONFIG  →  OAT grids  →  OPTUNA_SEARCH_SPACE  →  OPTUNA_BEST_CONFIG
             →  LAYER2/3 overrides  →  FINAL_BEST_CONFIG

Nothing is computed here; only values are declared.
"""

from datetime import datetime


# ── Baseline (un-tuned starting point, used in Sections 5–6) ──────────────────
GA_CONFIG = {
    'population_size': 30,
    'max_generations': 1000,
    'tournament_size': 3,
    'elitesize': 1,
    'xo_prob': 0.8,
    'mut_prob': 0.05,
    'crossover_type': 'uniform',
    'mutation_type': 'random',
    'maximization': False,
    'track_diversity': True,         # ALWAYS on — diagnostic is free
    'save_image_every_n_gens': 100,
}

# ── SA config for Challenge 2 (Section 9) ────────────────────────────────────
SA_CONFIG = {
    'C': 20.0, 'L': 150, 'H': 1.05,
    'max_iter': 200,                 # 150 × 200 = 30,000 = GA budget
    'save_every_n_steps': 1000,
}

# ── SA config for Section 10 hybrid GA → SA refinement ───────────────────────
# Lower C (start closer to GA's optimum) and faster cooling (H higher).
# Budget: L × max_iter = 100 × 60 = 6,000 evals ≈ 20% of 30,000 total.
HYBRID_SA_CONFIG = {
    'C': 10.0, 'L': 100, 'H': 1.05,
    'max_iter': 60,                  # 100 × 60 = 6,000 evals — higher C lets SA escape GA's local minimum
    'save_every_n_steps': 500,
}

# ── Section 6 OAT experiment grids ───────────────────────────────────────────
EXPERIMENT_A_POPULATION = [10, 20, 50]
EXPERIMENT_B_ELITE      = [0, 1, 5]
EXPERIMENT_C_MUTRATE    = [0.01, 0.05, 0.1, 0.2]
EXPERIMENT_D_CROSSOVER  = ['uniform', 'single_point']
EXPERIMENT_E_MUTTYPE    = ['vertex_shift', 'triangle_replace', 'random']
EXPERIMENT_F_TOURNAMENT = [2, 3, 4, 5, 6]
EXPERIMENT_G_XOPROB     = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXPERIMENT_H_INVERTED   = [
    {'mut_prob': 0.10, 'xo_prob': 0.6},
    {'mut_prob': 0.15, 'xo_prob': 0.6},
    {'mut_prob': 0.20, 'xo_prob': 0.5},
]
EXPERIMENT_I_STEPSIZE   = [
    {'vertex_step': 10, 'color_step': 15, 'alpha_step': 15},
    {'vertex_step': 30, 'color_step': 30, 'alpha_step': 30},   # default
    {'vertex_step': 50, 'color_step': 80, 'alpha_step': 50},
]

# ── Optuna search space (filled by Section 6 derivation cell) ─────────────────
OPTUNA_SEARCH_SPACE = {'population_size': {'type': 'int', 'low': 30, 'high': 80}, 'tournament_size': {'type': 'int', 'low': 3, 'high': 6}, 'mut_prob': {'type': 'float', 'low': 0.03, 'high': 0.12}, 'xo_prob': {'type': 'float', 'low': 0.85, 'high': 1.0}, 'elitesize': {'type': 'int', 'low': 2, 'high': 8}, 'crossover_type': {'type': 'categorical', 'choices': ['uniform', 'single_point']}, 'mutation_type': {'type': 'categorical', 'choices': ['random', 'vertex_shift', 'triangle_replace']}, 'vertex_step': {'type': 'int', 'low': 30, 'high': 60}, 'color_step': {'type': 'int', 'low': 30, 'high': 100}, 'alpha_step': {'type': 'int', 'low': 30, 'high': 60}}
OPTUNA_BEST_CONFIG = {'population_size': 66, 'max_generations': 1000, 'tournament_size': 6, 'elitesize': 6, 'xo_prob': 0.9862861192072796, 'mut_prob': 0.052665809643785456, 'crossover_type': 'uniform', 'mutation_type': 'vertex_shift', 'maximization': False, 'track_diversity': True, 'save_image_every_n_gens': 100, 'vertex_step': 50, 'color_step': 66, 
                      'alpha_step': 41}

# ── Section 8 structural improvement overrides ────────────────────────────────
LAYER2_GEOMETRIC       = {'mutation_type': 'geometric'}
LAYER2_REPLACE         = {'innovation_replace_prob': 0.02}
LAYER2_SWAP            = {'zorder_swap_prob': 0.01}
LAYER3_FITNESS_SHARING = {'fitness_sharing': True, 'sigma_share': 0.1}
LAYER3_DYNAMIC         = {
    'dynamic_schedule': True,
    'mut_prob_start': 0.15, 'mut_prob_end': 0.03,
    'tournament_start': 2,  'tournament_end': 5,
}

# ── Final combined config (filled after Section 8.6 picks winner) ─────────────
FINAL_BEST_CONFIG = {'population_size': 66, 'max_generations': 1000, 'tournament_size': 6, 'elitesize': 6, 'xo_prob': 0.9862861192072796, 'mut_prob': 0.052665809643785456, 'crossover_type': 'uniform', 'mutation_type': 'vertex_shift', 'maximization': False, 'track_diversity': True, 'save_image_every_n_gens': 100, 'vertex_step': 50, 'color_step': 66, 'alpha_step': 41, 'zorder_swap_prob': 0.01}  # OPTUNA_BEST_CONFIG + LAYER2_SWAP; 8.7 winner: 24.3291 vs Optuna 24.4738






def make_run_id(label):
    return f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
