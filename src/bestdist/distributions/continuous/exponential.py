"""Exponential distribution implementation."""

from typing import Tuple
from scipy.stats import expon, rv_continuous
import numpy as np

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Exponential(BaseDistribution):
    """
    Exponential distribution.
    
    The exponential distribution models the time between events in a Poisson 
    point process. Commonly used for modeling lifetimes, waiting times, and 
    time until failure.
    
    PDF: f(x) = (1/scale) * exp(-x/scale)
    
    Parameters:
        loc: Location parameter (default 0)
        scale: Scale parameter (mean = scale)
        
    Properties:
        - Memoryless property: P(X > s+t | X > s) = P(X > t)
        - Only continuous distribution with this property
        
    Example:
        ```python
        from bestdist.distributions.continuous import Exponential
        import numpy as np
        
        # Generate sample data (e.g., time between customer arrivals)
        data = np.random.exponential(scale=5.0, size=1000)
        
        # Fit distribution
        dist = Exponential(data)
        params = dist.fit()
        print(f"Mean time: {dist.mean:.2f} units")
        print(f"Rate (lambda): {dist.rate:.4f}")
        
        # Probability of event within 10 time units
        prob = dist.cdf(10)
        print(f"P(X <= 10) = {prob:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy exponential distribution."""
        return expon
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract location and scale parameters from fit result.
        
        Args:
            fit_result: Tuple of (loc, scale) from scipy.stats.expon.fit
            
        Returns:
            Dictionary with 'loc' and 'scale' parameters
        """
        return {
            'loc': float(fit_result[0]),
            'scale': float(fit_result[1])
        }
    
    @property
    def scale_param(self) -> float:
        """Scale parameter of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['scale']
    
    @property
    def rate(self) -> float:
        """Rate parameter (lambda = 1/scale) of the fitted distribution."""
        return 1.0 / self.scale_param
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        return self.params['loc'] + self.scale_param
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        return self.scale_param ** 2
    
    @property
    def median(self) -> float:
        """Median of the fitted distribution."""
        return self.params['loc'] + self.scale_param * np.log(2)

