"""Discrete probability distributions."""

from .poisson import Poisson
from .binomial import Binomial
from .negative_binomial import NegativeBinomial
from .geometric import Geometric

__all__ = [
    "Poisson",
    "Binomial",
    "NegativeBinomial",
    "Geometric",
]

