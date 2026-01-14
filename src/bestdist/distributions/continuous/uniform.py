"""Uniform distribution implementation."""

from typing import Tuple
from scipy.stats import uniform, rv_continuous

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Uniform(BaseDistribution):
    """
    Uniform (continuous) distribution.
    
    The uniform distribution assigns equal probability to all values in a 
    given interval [a, b]. Useful for modeling random variables where all 
    outcomes in a range are equally likely.
    
    PDF: f(x) = 1/(b-a) for a <= x <= b, 0 otherwise
    
    Parameters:
        loc: Lower bound (a)
        scale: Range width (b - a)
        
    Example:
        ```python
        from bestdist.distributions.continuous import Uniform
        import numpy as np
        
        # Generate sample data (e.g., random measurements)
        data = np.random.uniform(low=10, high=50, size=1000)
        
        # Fit distribution
        dist = Uniform(data)
        params = dist.fit()
        print(f"Lower bound: {dist.lower_bound:.2f}")
        print(f"Upper bound: {dist.upper_bound:.2f}")
        print(f"Mean: {dist.mean:.2f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy uniform distribution."""
        return uniform
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract location and scale parameters from fit result.
        
        Args:
            fit_result: Tuple of (loc, scale) from scipy.stats.uniform.fit
            
        Returns:
            Dictionary with 'loc' (lower bound) and 'scale' (range) parameters
        """
        return {
            'loc': float(fit_result[0]),
            'scale': float(fit_result[1])
        }
    
    @property
    def lower_bound(self) -> float:
        """Lower bound (a) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['loc']
    
    @property
    def upper_bound(self) -> float:
        """Upper bound (b) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['loc'] + self.params['scale']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        return (self.lower_bound + self.upper_bound) / 2
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        return (self.params['scale'] ** 2) / 12
    
    @property
    def median(self) -> float:
        """Median of the fitted distribution."""
        return self.mean

