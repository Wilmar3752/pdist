"""Geometric distribution implementation."""

from typing import Tuple
from scipy.stats import geom, rv_discrete
import numpy as np

from ...core.base_discrete import BaseDiscreteDistribution
from ...utils.types import Parameters


class Geometric(BaseDiscreteDistribution):
    """
    Geometric distribution.
    
    The geometric distribution models the number of trials needed to get the
    first success in a sequence of independent Bernoulli trials. Special case
    of the negative binomial distribution with r=1.
    
    PMF: P(X=k) = (1-p)^(k-1) * p
    
    Parameters:
        p: Probability of success (0 < p <= 1)
        
    Properties:
        - Mean = 1/p
        - Variance = (1-p) / p^2
        - Memoryless property
        
    Example:
        ```python
        from bestdist.distributions.discrete import Geometric
        import numpy as np
        
        # Generate sample data (e.g., number of attempts until first success)
        data = np.random.geometric(p=0.3, size=1000)
        
        # Fit distribution
        dist = Geometric(data)
        params = dist.fit()
        print(f"p (success probability): {dist.p:.4f}")
        print(f"Mean attempts: {dist.mean:.4f}")
        
        # Probability of first success on 3rd trial
        prob = dist.pmf(3)
        print(f"P(X=3) = {prob:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_discrete:
        """Return scipy Geometric distribution."""
        return geom
    
    def _fit_custom(self) -> Parameters:
        """
        Fit Geometric distribution using method of moments.
        
        For geometric distribution, p = 1 / mean
        """
        mean = np.mean(self.data)
        
        # Avoid division by zero
        if mean == 0:
            p = 1.0
        else:
            p = 1.0 / mean
        
        # Ensure valid range
        p = np.clip(p, 0.001, 1.0)
        
        return {'p': float(p)}
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract p parameter from fit result.
        
        Args:
            fit_result: Not used for Geometric (uses _fit_custom)
            
        Returns:
            Dictionary with 'p' parameter
        """
        return {'p': float(fit_result[0])}
    
    @property
    def p(self) -> float:
        """Success probability of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['p']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        return 1.0 / self.p
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        return (1 - self.p) / (self.p ** 2)
    
    @property
    def std(self) -> float:
        """Standard deviation of the fitted distribution."""
        return np.sqrt(self.variance)

