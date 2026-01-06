"""Normal (Gaussian) distribution implementation."""

from typing import Tuple
from scipy.stats import norm, rv_continuous

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Normal(BaseDistribution):
    """
    Normal (Gaussian) distribution.
    
    The normal distribution is characterized by its mean (mu) and 
    standard deviation (sigma). It's symmetric and bell-shaped.
    
    PDF: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
    
    Parameters:
        loc (mu): Mean of the distribution
        scale (sigma): Standard deviation (must be positive)
        
    Example:
        ```python
        from pdist.distributions.continuous import Normal
        import numpy as np
        
        # Generate sample data
        data = np.random.normal(loc=5, scale=2, size=1000)
        
        # Fit distribution
        dist = Normal(data)
        params = dist.fit()
        print(f"Fitted parameters: {params}")
        
        # Test goodness of fit
        ks_stat, p_value = dist.test_goodness_of_fit()
        print(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy normal distribution."""
        return norm
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract mean and standard deviation from fit result.
        
        Args:
            fit_result: Tuple of (loc, scale) from scipy.stats.norm.fit
            
        Returns:
            Dictionary with 'loc' (mean) and 'scale' (std dev)
        """
        return {
            'loc': float(fit_result[0]),
            'scale': float(fit_result[1])
        }
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['loc']
    
    @property
    def std(self) -> float:
        """Standard deviation of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['scale']
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        return self.std ** 2

