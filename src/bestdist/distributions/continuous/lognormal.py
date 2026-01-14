"""Lognormal distribution implementation."""

from typing import Tuple
from scipy.stats import lognorm, rv_continuous
import numpy as np

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Lognormal(BaseDistribution):
    """
    Lognormal distribution.
    
    The lognormal distribution is used to model positive data that are 
    right-skewed, such as income, asset prices, and biological measurements.
    If X is lognormally distributed, then ln(X) is normally distributed.
    
    PDF: f(x) = 1/(x*s*sqrt(2*pi)) * exp(-(ln(x)-loc)^2 / (2*s^2))
    
    Parameters:
        s (sigma): Shape parameter (standard deviation of log)
        loc: Location parameter (default 0)
        scale: Scale parameter (median when loc=0)
        
    Example:
        ```python
        from bestdist.distributions.continuous import Lognormal
        import numpy as np
        
        # Generate sample data (e.g., income data)
        data = np.random.lognormal(mean=3, sigma=1, size=1000)
        
        # Fit distribution
        dist = Lognormal(data)
        params = dist.fit()
        print(f"Sigma: {params['s']:.4f}, Scale: {params['scale']:.4f}")
        
        # Calculate median (50th percentile)
        median = dist.ppf(0.5)
        print(f"Median: {median:.2f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy lognormal distribution."""
        return lognorm
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract shape (sigma) and scale parameters from fit result.
        
        Args:
            fit_result: Tuple of (s, loc, scale) from scipy.stats.lognorm.fit
            
        Returns:
            Dictionary with 's' (sigma), 'loc', and 'scale' parameters
        """
        return {
            's': float(fit_result[0]),
            'loc': float(fit_result[1]),
            'scale': float(fit_result[2])
        }
    
    @property
    def sigma(self) -> float:
        """Shape parameter (sigma) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['s']
    
    @property
    def scale_param(self) -> float:
        """Scale parameter of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['scale']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        s = self.sigma
        scale = self.scale_param
        loc = self.params['loc']
        return loc + scale * np.exp(s**2 / 2)
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        s = self.sigma
        scale = self.scale_param
        return (scale**2) * np.exp(s**2) * (np.exp(s**2) - 1)
    
    @property
    def median(self) -> float:
        """Median of the fitted distribution."""
        return self.ppf(0.5)

