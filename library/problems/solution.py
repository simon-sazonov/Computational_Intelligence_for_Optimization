"""
Abstract base class for all optimization solutions.

Every problem-specific solution (TriangleImageSolution, etc.) inherits from this
class and must implement fitness() and random_initial_representation().

The algorithms (GA, SA, HC) only depend on this interface, never on the
problem-specific details — that separation is what makes them reusable.

Adapted from the P07 practice library (identical interface).
"""

import numpy as np
from abc import ABC, abstractmethod


class Solution(ABC):

    def __init__(self, repr=None):
        # If no representation is given, generate a random one.
        if repr is None:
            repr = self.random_initial_representation()
        self.repr = repr

    def __repr__(self):
        return str(self.repr)

    @abstractmethod
    def fitness(self):
        """Return a scalar quality score for this solution."""
        pass

    @abstractmethod
    def random_initial_representation(self):
        """Return a random representation for this solution type."""
        pass
