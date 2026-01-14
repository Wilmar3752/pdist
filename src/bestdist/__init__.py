"""
bestdist - Find the best probability distribution for your data
================================================================

bestdist is a Python package for fitting probability distributions to data
and identifying the best fit using various statistical tests.

Basic Usage
-----------
```python
from bestdist import DistributionFitter
import numpy as np

# Generate sample data
data = np.random.gamma(2, 2, 1000)

# Fit distributions
fitter = DistributionFitter(data)
results = fitter.fit()

# Get best distribution
best = fitter.get_best_distribution()
print(f"Best fit: {best['distribution']}")

# View summary
print(fitter.summary())

# Plot results
fitter.plot_best_fit()
```

For more information, see the documentation at https://github.com/Wilmar3752/pdist
"""

__version__ = "0.1.0"
__author__ = "Wilmar Sepulveda"
__license__ = "MIT"

# Core functionality
from .core.fitter import DistributionFitter
from .core.base import BaseDistribution
from .core.base_discrete import BaseDiscreteDistribution

# Continuous distributions
from .distributions.continuous.normal import Normal
from .distributions.continuous.gamma import Gamma
from .distributions.continuous.beta import Beta
from .distributions.continuous.weibull import Weibull
from .distributions.continuous.lognormal import Lognormal
from .distributions.continuous.exponential import Exponential
from .distributions.continuous.uniform import Uniform
from .distributions.continuous.cauchy import Cauchy
from .distributions.continuous.student_t import StudentT

# Discrete distributions
from .distributions.discrete.poisson import Poisson
from .distributions.discrete.binomial import Binomial
from .distributions.discrete.negative_binomial import NegativeBinomial
from .distributions.discrete.geometric import Geometric

# Exceptions
from .utils.exceptions import (
    BestdistException,
    DataValidationError,
    FittingError,
    InsufficientDataError,
    InvalidDistributionError,
    ConvergenceError,
)

__all__ = [
    # Main classes
    "DistributionFitter",
    "BaseDistribution",
    "BaseDiscreteDistribution",
    
    # Continuous distributions
    "Normal",
    "Gamma",
    "Beta",
    "Weibull",
    "Lognormal",
    "Exponential",
    "Uniform",
    "Cauchy",
    "StudentT",
    
    # Discrete distributions
    "Poisson",
    "Binomial",
    "NegativeBinomial",
    "Geometric",
    
    # Exceptions
    "BestdistException",
    "DataValidationError",
    "FittingError",
    "InsufficientDataError",
    "InvalidDistributionError",
    "ConvergenceError",
    
    # Metadata
    "__version__",
]

