"""Weibull distribution implementation."""

from typing import Tuple
from scipy.stats import weibull_min, rv_continuous

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Weibull(BaseDistribution):
    """
    Weibull distribution.
    
    The Weibull distribution is widely used in reliability engineering and
    survival analysis. It's particularly useful for modeling time-to-failure
    data and wind speed distributions.
    
    PDF: f(x) = (c/scale) * (x/scale)^(c-1) * exp(-(x/scale)^c)
    
    Parameters:
        c: Shape parameter (must be positive)
            - c < 1: failure rate decreases over time
            - c = 1: constant failure rate (exponential)
            - c > 1: failure rate increases over time
        scale: Scale parameter (must be positive)
        loc: Location parameter (default 0)
        
    Example:
        ```python
        from pdist.distributions.continuous import Weibull
        import numpy as np
        from scipy.stats import weibull_min
        
        # Generate sample data
        data = weibull_min.rvs(c=1.5, scale=1.0, size=1000)
        
        # Fit distribution
        dist = Weibull(data)
        params = dist.fit()
        print(f"Shape: {params['c']:.4f}, Scale: {params['scale']:.4f}")
        
        # Test goodness of fit
        ks_stat, p_value = dist.test_goodness_of_fit()
        print(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
        ```
        
    Note:
        This implements the Weibull minimum distribution (weibull_min),
        which is the most common form used in reliability engineering.
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy Weibull minimum distribution."""
        return weibull_min
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract shape and scale parameters from fit result.
        
        Args:
            fit_result: Tuple of (c, loc, scale) from scipy.stats.weibull_min.fit
            
        Returns:
            Dictionary with 'c' (shape), 'loc', and 'scale' parameters
        """
        return {
            'c': float(fit_result[0]),
            'loc': float(fit_result[1]),
            'scale': float(fit_result[2])
        }
    
    @property
    def shape(self) -> float:
        """Shape parameter (c) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['c']
    
    @property
    def scale(self) -> float:
        """Scale parameter of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['scale']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        from scipy.special import gamma as gamma_func
        return self.scale * gamma_func(1 + 1/self.shape)
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        from scipy.special import gamma as gamma_func
        c = self.shape
        scale = self.scale
        return (scale ** 2) * (gamma_func(1 + 2/c) - gamma_func(1 + 1/c) ** 2)
    
    @property
    def mode(self) -> float:
        """Mode of the fitted distribution."""
        c = self.shape
        scale = self.scale
        if c > 1:
            return scale * ((c - 1) / c) ** (1 / c)
        else:
            return 0.0  # Mode is at 0 when c <= 1

