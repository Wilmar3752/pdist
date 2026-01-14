"""Binomial distribution implementation."""

from typing import Tuple
from scipy.stats import binom, rv_discrete
import numpy as np

from ...core.base_discrete import BaseDiscreteDistribution
from ...utils.types import Parameters


class Binomial(BaseDiscreteDistribution):
    """
    Binomial distribution.
    
    The binomial distribution models the number of successes in a fixed number
    of independent Bernoulli trials, each with the same probability of success.
    
    PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    
    Parameters:
        n: Number of trials (must be positive integer)
        p: Probability of success (0 <= p <= 1)
        
    Properties:
        - Mean = n * p
        - Variance = n * p * (1-p)
        - Used for binary outcomes over fixed trials
        
    Example:
        ```python
        from bestdist.distributions.discrete import Binomial
        import numpy as np
        
        # Generate sample data (e.g., number of successful sales out of 10 attempts)
        data = np.random.binomial(n=10, p=0.3, size=1000)
        
        # Fit distribution
        dist = Binomial(data)
        params = dist.fit()
        print(f"n (trials): {dist.n:.0f}")
        print(f"p (success probability): {dist.p:.4f}")
        print(f"Mean: {dist.mean:.4f}")
        
        # Probability of exactly 3 successes
        prob = dist.pmf(3)
        print(f"P(X=3) = {prob:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_discrete:
        """Return scipy Binomial distribution."""
        return binom
    
    def _fit_custom(self) -> Parameters:
        """
        Fit Binomial distribution using method of moments.
        
        Estimates:
        - n: maximum value observed (approximation)
        - p: mean / n
        """
        # Estimate n as the maximum observed value
        n = int(np.max(self.data))
        
        # If all values are the same, increase n slightly
        if n == 0:
            n = 1
        
        # Estimate p as mean / n
        mean = np.mean(self.data)
        p = min(mean / n, 1.0)  # Ensure p <= 1
        
        # Adjust if variance suggests different n
        var = np.var(self.data)
        if p > 0 and p < 1:
            # var = n * p * (1-p)
            # Solve for n: n = var / (p * (1-p))
            estimated_n = var / (p * (1 - p))
            if estimated_n > n:
                n = int(np.ceil(estimated_n))
                p = mean / n
        
        return {'n': n, 'p': float(p)}
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract n and p parameters from fit result.
        
        Args:
            fit_result: Not used for Binomial (uses _fit_custom)
            
        Returns:
            Dictionary with 'n' and 'p' parameters
        """
        return {'n': int(fit_result[0]), 'p': float(fit_result[1])}
    
    @property
    def n(self) -> int:
        """Number of trials of the fitted distribution."""
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
        return self.n * self.p
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        return self.n * self.p * (1 - self.p)
    
    @property
    def std(self) -> float:
        """Standard deviation of the fitted distribution."""
        return np.sqrt(self.variance)

