"""Gamma distribution implementation."""

from typing import Tuple
from scipy.stats import gamma, rv_continuous

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Gamma(BaseDistribution):
    """
    Gamma distribution.
    
    The gamma distribution is a two-parameter family of continuous probability
    distributions. It's commonly used to model waiting times and is the 
    conjugate prior for the precision of the normal distribution.
    
    PDF: f(x) = (x^(a-1) * exp(-x/scale)) / (scale^a * Gamma(a))
    
    Parameters:
        a (alpha): Shape parameter (must be positive)
        scale (beta): Scale parameter (must be positive)
        loc: Location parameter (default 0)
        
    Example:
        ```python
        from pdist.distributions.continuous import Gamma
        import numpy as np
        
        # Generate sample data
        data = np.random.gamma(shape=2, scale=2, size=1000)
        
        # Fit distribution
        dist = Gamma(data)
        params = dist.fit()
        print(f"Shape: {params['a']:.4f}, Scale: {params['scale']:.4f}")
        
        # Test goodness of fit
        ks_stat, p_value = dist.test_goodness_of_fit()
        print(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy gamma distribution."""
        return gamma
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract shape and scale parameters from fit result.
        
        Args:
            fit_result: Tuple of (a, loc, scale) from scipy.stats.gamma.fit
            
        Returns:
            Dictionary with 'a' (shape), 'loc', and 'scale' parameters
        """
        return {
            'a': float(fit_result[0]),
            'loc': float(fit_result[1]),
            'scale': float(fit_result[2])
        }
    
    @property
    def shape(self) -> float:
        """Shape parameter (alpha) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['a']
    
    @property
    def scale(self) -> float:
        """Scale parameter (beta) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['scale']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        return self.shape * self.scale
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        return self.shape * (self.scale ** 2)

