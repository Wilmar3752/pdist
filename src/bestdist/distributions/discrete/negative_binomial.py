"""Negative Binomial distribution implementation."""

from typing import Tuple
from scipy.stats import nbinom, rv_discrete
import numpy as np

from ...core.base_discrete import BaseDiscreteDistribution
from ...utils.types import Parameters


class NegativeBinomial(BaseDiscreteDistribution):
    """
    Negative Binomial distribution.
    
    The negative binomial distribution models the number of failures before
    achieving a specified number of successes in a sequence of independent
    Bernoulli trials. Also used for overdispersed count data.
    
    PMF: P(X=k) = C(k+r-1, k) * p^r * (1-p)^k
    
    Parameters:
        n (r): Number of successes (must be positive)
        p: Probability of success (0 < p <= 1)
        
    Properties:
        - Mean = r * (1-p) / p
        - Variance = r * (1-p) / p^2
        - Used when variance > mean (overdispersion)
        
    Example:
        ```python
        from bestdist.distributions.discrete import NegativeBinomial
        import numpy as np
        
        # Generate sample data (e.g., number of failures before 5 successes)
        data = np.random.negative_binomial(n=5, p=0.5, size=1000)
        
        # Fit distribution
        dist = NegativeBinomial(data)
        params = dist.fit()
        print(f"r (successes): {dist.r:.4f}")
        print(f"p (success probability): {dist.p:.4f}")
        print(f"Mean: {dist.mean:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_discrete:
        """Return scipy Negative Binomial distribution."""
        return nbinom
    
    def _fit_custom(self) -> Parameters:
        """
        Fit Negative Binomial distribution using method of moments.
        
        Estimates:
        - p: mean / (mean + variance)
        - r: mean * p / (1 - p)
        """
        mean = np.mean(self.data)
        var = np.var(self.data)
        
        # Avoid division by zero
        if var <= mean or mean == 0:
            # Fall back to Poisson-like parameters
            p = 0.5
            r = mean
        else:
            # Method of moments
            p = mean / var
            r = mean * p / (1 - p)
        
        # Ensure valid range
        p = np.clip(p, 0.001, 0.999)
        r = max(r, 0.1)
        
        return {'n': float(r), 'p': float(p)}
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract n and p parameters from fit result.
        
        Args:
            fit_result: Not used for NegativeBinomial (uses _fit_custom)
            
        Returns:
            Dictionary with 'n' (r) and 'p' parameters
        """
        return {'n': float(fit_result[0]), 'p': float(fit_result[1])}
    
    @property
    def r(self) -> float:
        """Number of successes (r) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['n']
    
    @property
    def p(self) -> float:
        """Success probability of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['p']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        return self.r * (1 - self.p) / self.p
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        return self.r * (1 - self.p) / (self.p ** 2)
    
    @property
    def std(self) -> float:
        """Standard deviation of the fitted distribution."""
        return np.sqrt(self.variance)

