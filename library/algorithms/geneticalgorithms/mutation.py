"""
Mutation operators for the Triangle Image GA.

Five operators are defined, each targeting a different aspect of the triangle
representation.  They can be used individually or combined via "random" mode.

Design rationale
----------------
- vertex_shift, color_shift, alpha_shift are *exploitation* operators: they
  make small local improvements to existing triangles.
- triangle_replace is an *exploration* operator: it escapes local optima by
  completely randomising one triangle.
- triangle_swap changes the rendering order (z-order) without changing any
  triangle's intrinsic properties — useful when two triangles occlude each
  other in a suboptimal order.

The "random" default mode picks uniformly from all five operators for each
mutation event.  Experiment with individual modes in the notebook (Section 6)
to understand their individual contributions.
"""

import random
from copy import deepcopy


_GENES_PER_TRIANGLE = 10


def triangle_mutation(individual, mut_prob: float,
                      mutation_type: str = "random",
                      verbose: bool = False):
    """
    Apply per-triangle mutation to an individual.

    For each of the 100 triangles, with probability mut_prob, one mutation
    operator fires.  The operator is selected by mutation_type.

    Parameters
    ----------
    individual : TriangleImageSolution
    mut_prob : float — probability that any given triangle is mutated.
    mutation_type : str — one of:
        "vertex_shift"    small displacement of one vertex (±30 px)
        "color_shift"     small change to one RGB channel (±30)
        "alpha_shift"     small change to alpha / transparency (±30)
        "triangle_replace" replace the entire triangle with a random one
        "triangle_swap"   swap the z-order of two random triangles
        "random"          choose uniformly from vertex_shift, color_shift,
                          alpha_shift (exploitation-only; avoids destructive
                          triangle_replace / triangle_swap)
    verbose : bool

    Returns
    -------
    TriangleImageSolution — a new mutated individual.
    """
    img_w = individual.__class__.IMG_W
    img_h = individual.__class__.IMG_H
    n_triangles = len(individual.repr) // _GENES_PER_TRIANGLE

    new_repr = individual.repr.copy()

    _OPERATORS = ["vertex_shift", "color_shift", "alpha_shift"]

    i = 0
    while i < n_triangles:
        if random.random() > mut_prob:
            i += 1
            continue

        # Select which operator to apply this event
        op = random.choice(_OPERATORS) if mutation_type == "random" else mutation_type

        if op == "triangle_swap":
            # Swap z-order of triangle i with another random triangle.
            j = random.randint(0, n_triangles - 1)
            if j == i:
                j = (i + 1) % n_triangles
            base_i = i * _GENES_PER_TRIANGLE
            base_j = j * _GENES_PER_TRIANGLE
            # Swap all 10 genes between triangle i and j
            new_repr[base_i:base_i + _GENES_PER_TRIANGLE], \
            new_repr[base_j:base_j + _GENES_PER_TRIANGLE] = \
                new_repr[base_j:base_j + _GENES_PER_TRIANGLE], \
                new_repr[base_i:base_i + _GENES_PER_TRIANGLE]
            if verbose:
                print(f"triangle_swap: triangles {i} and {j} swapped z-order.")
            # Skip j as well since we already mutated it
            i += 1
            continue

        base = i * _GENES_PER_TRIANGLE

        if op == "vertex_shift":
            vertex = random.randint(0, 2)
            axis   = random.randint(0, 1)
            gene   = base + vertex * 2 + axis
            bound  = img_w if axis == 0 else img_h
            delta  = random.randint(-30, 30)
            new_repr[gene] = max(0, min(bound, new_repr[gene] + delta))
            if verbose:
                print(f"vertex_shift: triangle {i}, vertex {vertex}, axis {'x' if axis==0 else 'y'}, delta={delta}")

        elif op == "color_shift":
            channel = random.randint(0, 2)
            gene    = base + 6 + channel
            delta   = random.randint(-30, 30)
            new_repr[gene] = max(0, min(255, new_repr[gene] + delta))
            if verbose:
                print(f"color_shift: triangle {i}, channel {['R','G','B'][channel]}, delta={delta}")

        elif op == "alpha_shift":
            gene  = base + 9
            delta = random.randint(-30, 30)
            new_repr[gene] = max(0, min(255, new_repr[gene] + delta))
            if verbose:
                print(f"alpha_shift: triangle {i}, delta={delta}")

        elif op == "triangle_replace":
            # Completely randomise this triangle
            new_repr[base]     = random.randint(0, img_w)
            new_repr[base + 1] = random.randint(0, img_h)
            new_repr[base + 2] = random.randint(0, img_w)
            new_repr[base + 3] = random.randint(0, img_h)
            new_repr[base + 4] = random.randint(0, img_w)
            new_repr[base + 5] = random.randint(0, img_h)
            new_repr[base + 6] = random.randint(0, 255)
            new_repr[base + 7] = random.randint(0, 255)
            new_repr[base + 8] = random.randint(0, 255)
            new_repr[base + 9] = random.randint(0, 255)
            if verbose:
                print(f"triangle_replace: triangle {i} fully randomised.")

        i += 1

    return individual.with_repr(new_repr)
