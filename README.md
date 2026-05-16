Final project pipeline — file-by-file logic
The pipeline is built around one principle: every artifact in the report traces to one logical step, and every step uses one canonical config. No more orphan baselines or pre-Optuna leftovers.
The pipeline at a glance
The flow goes: baseline run → diagnose problem → tune parameters (OAT + Optuna) → apply structural improvements → validate winner → compare against SA. Each notebook section consumes exactly one upstream config and produces exactly one downstream config.
StageSectionInput configOutput configQuestion answeredSetup1–4——What is the problem and representation?Baseline + diagnosis5GA_CONFIGGA_CONFIG (unchanged)Does the baseline lose diversity prematurely?Parameter exploration6GA_CONFIGOAT-derived boundsWhich parameter ranges are competitive?Joint tuning7OAT boundsOPTUNA_BEST_CONFIGWhat is the best parameter combination?Structural improvements8OPTUNA_BEST_CONFIGFINAL_BEST_CONFIGDo new operators improve on tuning alone?Validation10FINAL_BEST_CONFIG—Is the improvement statistically real?Algorithm comparison9FINAL_BEST_CONFIG vs SA_CONFIG—Does GA beat SA at equal compute?

File-by-file specification
config.py — single source of truth for all parameters
Holds every parameter dict used anywhere in the project. Nothing computes here; this file only declares values. Keeping all configs in one file means a reader can audit every choice in one place and the notebook never hardcodes numbers.
python# === Baseline (un-tuned starting point, used in Sections 5–6) ===
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

# === SA config for Challenge 2 (Section 9) ===
SA_CONFIG = {
    'C': 20.0, 'L': 150, 'H': 1.05,
    'max_iter': 200,                 # 150 × 200 = 30,000 = GA budget
    'save_every_n_steps': 1000,
}

# === Section 6 OAT experiment grids ===
EXPERIMENT_A_POPULATION = [10, 20, 50]
EXPERIMENT_B_ELITE = [0, 1, 5]
EXPERIMENT_C_MUTRATE = [0.01, 0.05, 0.1, 0.2]
EXPERIMENT_D_CROSSOVER = ['uniform', 'single_point']
EXPERIMENT_E_MUTTYPE = ['vertex_shift', 'triangle_replace', 'random']
EXPERIMENT_F_TOURNAMENT = [2, 3, 4, 5, 6]
EXPERIMENT_G_XOPROB = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXPERIMENT_H_INVERTED = [
    {'mut_prob': 0.10, 'xo_prob': 0.6},
    {'mut_prob': 0.15, 'xo_prob': 0.6},
    {'mut_prob': 0.20, 'xo_prob': 0.5},
]
EXPERIMENT_I_STEPSIZE = [
    {'vertex_step': 10, 'color_step': 15, 'alpha_step': 15},
    {'vertex_step': 30, 'color_step': 30, 'alpha_step': 30},   # default
    {'vertex_step': 50, 'color_step': 80, 'alpha_step': 50},
]

# === Optuna search space (populated by Section 6 after OAT runs) ===
OPTUNA_SEARCH_SPACE = {}  # filled by Section 6 derivation cell
OPTUNA_BEST_CONFIG = {}   # filled by Section 7 after study completes

# === Section 8 structural improvement overrides ===
LAYER2_GEOMETRIC = {'mutation_type': 'geometric'}
LAYER2_REPLACE = {'innovation_replace_prob': 0.02}
LAYER2_SWAP = {'zorder_swap_prob': 0.01}
LAYER3_FITNESS_SHARING = {'fitness_sharing': True, 'sigma_share': 0.1}
LAYER3_DYNAMIC = {
    'dynamic_schedule': True,
    'mut_prob_start': 0.15, 'mut_prob_end': 0.03,
    'tournament_start': 2, 'tournament_end': 5,
}

# === Final combined config (filled after Section 8.6 picks winner) ===
FINAL_BEST_CONFIG = {}

# === Helper ===
def make_run_id(label): return f"{label}_{datetime.now():%Y%m%d_%H%M%S}"

library/problems/solution.py — abstract base class
Defines the contract every solution must satisfy. The genetic algorithm code never references TriangleImageSolution directly; it works against this interface. This allows swapping in a different problem (e.g., a benchmark function) without touching the GA.
class Solution(ABC):
    repr: list                          # the genome
    
    @abstractmethod
    def random_initial_representation(self) -> list
    @abstractmethod
    def fitness(self) -> float          # cached after first call
    @abstractmethod
    def with_repr(self, new_repr) -> Solution   # immutable update
    
    def distance_to(self, other) -> float
        # Euclidean distance between self.repr and other.repr
        # used by diversity metrics and fitness sharing

library/problems/triangle_image.py — the concrete problem
Encodes the 100-triangle representation, renders it via PIL alpha compositing, and computes RMSE against the target image. This is where every problem-specific decision lives.
The class loads the target image once at module load time and shares it across all instances via a class attribute. Each instance owns only its repr list of 1000 integers. Fitness is cached in _fitness after the first call because rendering 100 triangles is expensive.
class TriangleImageSolution(Solution):
    target_array: np.ndarray = None     # class-level, shared
    IMG_W = 300; IMG_H = 400
    N_TRIANGLES = 100; GENES_PER_TRI = 10
    
    @classmethod
    def load_target(cls, path)
        # resize to IMG_W × IMG_H, convert to RGB numpy array
    
    def random_initial_representation(self)
        # for each of 100 triangles:
        #   3 vertices: x in [0, IMG_W], y in [0, IMG_H]
        #   RGB: each in [0, 255]
        #   alpha: in [20, 180]    # avoids invisible/fully-opaque extremes
    
    def render(self) -> Image
        # start with black canvas (RGBA)
        # for each triangle (in genome order = z-order):
        #     create transparent layer, draw triangle with its RGBA
        #     canvas = alpha_composite(canvas, layer)   # Porter-Duff Over
        # return canvas
    
    def fitness(self) -> float
        # if cached: return self._fitness
        # rendered = render() → np.array
        # rmse = sqrt(mean((rendered - target)**2))
        # cache and return
    
    def with_repr(self, new_repr)
        # return fresh instance — never mutate self.repr
        # this guarantees the fitness cache is never stale
The with_repr pattern matters: every crossover and mutation returns a new solution rather than modifying the parent in place. This eliminates an entire class of cache-invalidation bugs.

library/algorithms/geneticalgorithms/selection.py
Two selection algorithms. Tournament is the default; the others are kept for completeness so the report can reference them.
def tournament_selection(population, maximization, tournament_size=3)
    # 1. sample `tournament_size` individuals with replacement
    # 2. return the best by fitness
    # repetition allowed → worst individual retains nonzero probability

def fitness_proportionate_selection(population, maximization)
    # roulette wheel
    # for minimization: invert fitness via f' = f_max + 1 - f
    # build cumulative probability, sample uniformly in [0, 1]
    # included for completeness; not used by default because RMSE differences
    # are tight late in the run and roulette loses pressure

library/algorithms/geneticalgorithms/crossover.py
Three crossover operators, all triangle-level (atomic 10-gene blocks). Splitting inside a triangle would break shape-color co-adaptation, which the schema theorem tells us destroys building blocks faster than selection can preserve them.
def uniform_triangle_crossover(parent1, parent2, crossover_prob)
    # if random() > crossover_prob: return parents unchanged (replication)
    # for each of 100 triangle slots: coin flip → inherit from p1 or p2
    # returns two offspring (mirror images)

def single_point_triangle_crossover(parent1, parent2, crossover_prob)
    # if random() > crossover_prob: return parents unchanged
    # pick random cut k in [1, 99]
    # offspring1 = p1[:k] + p2[k:]
    # offspring2 = p2[:k] + p1[k:]
    # preserves z-order coherence within each half — relevant for alpha blending

def block_triangle_crossover(parent1, parent2, crossover_prob)
    # z-order preservation operator
    # pick random block [a, b) of contiguous z-positions
    # offspring inherits that block from one parent, rest from the other
    # used in Section 8.5 (Layer 4) if Sections 8.1–8.4 prove insufficient

library/algorithms/geneticalgorithms/mutation.py
The mutation file holds the most innovation. It must support random, geometric, and the triangle_replace/triangle_swap innovation operators behind config flags. All step sizes are configurable rather than hardcoded — Experiment I tunes them.
def triangle_mutation(individual, mut_prob, mutation_type='random',
                       vertex_step=30, color_step=30, alpha_step=30,
                       innovation_replace_prob=0.0, zorder_swap_prob=0.0)
    # work on a copy of individual.repr
    
    # === Standard per-triangle mutation ===
    for each triangle slot t in [0, 99]:
        if random() < mut_prob:
            if mutation_type == 'random':
                apply one of: vertex_shift, color_shift, alpha_shift (uniform pick)
            elif mutation_type == 'geometric':
                # additive perturbation on ALL genes of this triangle
                vertex coords += uniform(-vertex_step, vertex_step)
                RGB           += uniform(-color_step, color_step)
                alpha         += uniform(-alpha_step, alpha_step)
                clamp to valid ranges
            elif mutation_type == 'vertex_shift':
                shift one vertex by ±vertex_step
            elif mutation_type == 'color_shift':
                shift one RGB channel by ±color_step
            elif mutation_type == 'alpha_shift':
                shift alpha by ±alpha_step
    
    # === Innovation operators ===
    # applied INDEPENDENTLY of the standard mutation above
    for each triangle slot t in [0, 99]:
        if random() < innovation_replace_prob:
            replace triangle t with a fresh random triangle
    
    if random() < zorder_swap_prob:
        swap two random z-positions in the genome
    
    return individual.with_repr(new_repr)
Two design decisions matter here. First, geometric mutation perturbs the whole triangle atomically — vertex, color, and alpha together — because the unimodality intuition only holds when all dimensions move together. Second, innovation_replace_prob and zorder_swap_prob are independent of mut_prob so Section 8.2 can enable them without changing the base mutation rate.

library/algorithms/geneticalgorithms/ga.py — the main algorithm
The largest file. Contains the GA loop, diversity tracking, fitness sharing, dynamic schedules, plotting helpers, and run logging.
Core loop
def genetic_algorithm(solution_class, cfg, selection_fn, xo_fn, mut_fn,
                       run_id=None, verbose=False)
    
    # === Initialize ===
    population = [solution_class() for _ in range(cfg['population_size'])]
    stats = {'fitness_history': [], 'mean_dist_history': [],
             'var_fitness_history': [], 'entropy_fitness_history': []}
    
    for gen in range(cfg['max_generations']):
        
        # === Dynamic schedule ===
        if cfg.get('dynamic_schedule'):
            mut_prob = linear_interp(cfg['mut_prob_start'], cfg['mut_prob_end'], gen)
            tournament_size = linear_interp(cfg['tournament_start'], cfg['tournament_end'], gen)
        else:
            mut_prob = cfg['mut_prob']
            tournament_size = cfg['tournament_size']
        
        # === Fitness sharing ===
        if cfg.get('fitness_sharing'):
            shared_fitness = compute_shared_fitness(population, cfg['sigma_share'])
            selection_fitness = shared_fitness   # selection sees penalized values
        else:
            selection_fitness = [ind.fitness() for ind in population]
        
        # === Elitism ===
        elite = get_elite(population, cfg['elitesize'], maximization=False)
        
        # === Generate offspring ===
        new_pop = list(elite)
        while len(new_pop) < cfg['population_size']:
            p1 = selection_fn(population, selection_fitness)
            p2 = selection_fn(population, selection_fitness)
            c1, c2 = xo_fn(p1, p2, cfg['xo_prob'])
            c1 = mut_fn(c1, mut_prob)
            c2 = mut_fn(c2, mut_prob)
            new_pop.extend([c1, c2])
        population = new_pop[:cfg['population_size']]
        
        # === Log statistics ===
        record_stats(stats, population)
        
        # === Early stopping ===
        if cfg.get('patience') and plateau_detected(stats, cfg['patience'], cfg['min_delta']):
            break
    
    if run_id: save_log(run_id, stats, cfg)
    return best_individual(population), stats
Supporting functions in the same file
def compute_shared_fitness(population, sigma_share)
    # for minimization
    # pairwise normalized Euclidean distances
    # sharing function S(d) = max(0, 1 - d / sigma_share)
    # sigma(x) = 1 + sum over y != x of S(d(x, y))
    # return [f(x) * sigma(x) for x in population]
    # multiply (not divide) because we minimize → crowded individuals get penalized

def record_stats(stats, population)
    # called every generation
    # appends: best fitness, variance of fitness, entropy of fitness,
    #          mean pairwise distance, std of pairwise distance

def get_elite(population, n, maximization)
    # return top n individuals by fitness (no copy needed; solutions are immutable)

def plateau_detected(stats, patience, min_delta)
    # adaptive termination
    # if last `patience` generations improved best fitness by less than min_delta, stop

def load_or_run_ga(label, ...) 
    # check results/ for cached run with matching config
    # if found and config matches exactly: load and return
    # else: run fresh, save, return
    # this is what makes the notebook re-runnable without losing hours

library/algorithms/simulated_annealing.py
Used only in Section 9. Single-solution metaheuristic. Per-iteration fitness evaluation count is 1, so the total budget to match GA is population_size × max_generations SA iterations.
def simulated_annealing(initial_solution, C, L, H, maximization, max_iter)
    # standard SA from P06
    # C = initial temperature
    # L = inner loop length (neighbors per temperature level)
    # H = cooling factor (T <- T / H)
    # neighborhood = pick 1 triangle, apply small mutation (vertex/color/alpha shift only)
    #                NOT triangle_replace — keep neighbors local
    # return best solution found, fitness history
The neighborhood deliberately excludes triangle_replace so SA's exploration is governed by the temperature schedule, not by neighbor size. This is the methodologically clean comparison.

main_notebook.ipynb — the orchestration
The notebook is the only place where files talk to each other. It has 12 sections matching the pipeline table at the top. Each section reads one config from config.py, calls library functions, and writes its results to results/ and (for Optuna and Section 8) back into config.py.
Section flow
SectionActionReadsWrites1Setup, imports, target image——2Representation explanation + schema framing——3Fitness function explanation——4Crossover and mutation operator demos——5Baseline run + diversity diagnosticGA_CONFIGresults/baseline_*6OAT experiments A–I, then derive Optuna boundsGA_CONFIG + experiment gridsOPTUNA_SEARCH_SPACE in config.py7Optuna TPE studyOPTUNA_SEARCH_SPACEOPTUNA_BEST_CONFIG in config.py8Structural ablations 8.1–8.4, synthesis in 8.6OPTUNA_BEST_CONFIGbest override9GA vs SA Challenge 2 (uses FINAL_BEST_CONFIG)FINAL_BEST_CONFIG, SA_CONFIGresults/challenge2_*10Multi-seed validation of FINAL_BEST_CONFIGFINAL_BEST_CONFIGresults/final_validation_*11Adaptive termination demonstrationFINAL_BEST_CONFIG + patienceresults/early_stop_*12Conclusions——
The key consistency rule
Every section either uses one config or compares against one config. Sections 5 and 6 anchor on GA_CONFIG. Sections 7–11 anchor on the post-tuning config. There is no orphan run, no historical reference floating in the middle of a tuned analysis.

How this pipeline answers every grading criterion
The project grade has four components: code functionality (40%), code structure (10%), challenge implementation (10%), and report content (40%).
Code functionality. Each library file has one responsibility, tested by one section. The Solution ABC means problem and algorithm are decoupled. The with_repr immutability pattern eliminates cache bugs. The load_or_run_ga cache means experiments are reproducible without re-running.
Code structure. Six library files, one config file, one notebook. Imports flow strictly from problems → algorithms → notebook. No circular references, no hardcoded paths inside library code.
Challenge implementation. Challenge 2 (GA vs SA) is in Section 9 with proper methodology — 30 seeds, equal evaluation budget, Mann-Whitney test, Success Rate with pre-defined threshold.
Report content. Every Section 8 intervention cites a specific mechanism. Every parameter choice traces to either an OAT experiment, Optuna's TPE, or an explicit theoretical motivation. Every claim has a Wilcoxon p-value. The conclusions tie the schema theorem and the continuous optimization framework to the measured RMSE plateau, explaining what worked, what did not, and why the problem is harder than the baseline framework suggests.
