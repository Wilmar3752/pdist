"""Poisson distribution implementation."""

from typing import Tuple
from scipy.stats import poisson, rv_discrete
import numpy as np

from ...core.base_discrete import BaseDiscreteDistribution
from ...utils.types import Parameters


class Poisson(BaseDiscreteDistribution):
    """
    Poisson distribution.
    
    The Poisson distribution models the number of events occurring in a fixed
    interval of time or space, given a known average rate. Common applications
    include counting rare events, customer arrivals, and defect occurrences.
    
    PMF: P(X=k) = (λ^k * e^(-λ)) / k!
    
    Parameters:
        mu (λ): Average rate parameter (must be positive)
        
    Properties:
        - Mean = Variance = λ
        - Used for count data
        - Memoryless in continuous time
        
    Example:
        ```python
        from bestdist.distributions.discrete import Poisson
        import numpy as np
        
        # Generate sample data (e.g., number of calls per hour)
        data = np.random.poisson(lam=3.5, size=1000)
        
        # Fit distribution
        dist = Poisson(data)
        params = dist.fit()
        print(f"Lambda (rate): {dist.mu:.4f}")
        print(f"Mean: {dist.mean:.4f}")
        
        # Probability of exactly 5 events
        prob = dist.pmf(5)
        print(f"P(X=5) = {prob:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_discrete:
        """Return scipy Poisson distribution."""
        return poisson
    
    def _fit_custom(self) -> Parameters:
        """
        Fit Poisson distribution using method of moments.
        For Poisson, the MLE of λ is simply the sample mean.
        """
        mu = float(np.mean(self.data))
        return {'mu': mu}
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract mu parameter from fit result.
        
        Args:
            fit_result: Not used for Poisson (uses _fit_custom)
            
        Returns:
            Dictionary with 'mu' parameter
        """
        return {'mu': float(fit_result[0])}
    
    @property
    def mu(self) -> float:
        """Rate parameter (lambda) of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['mu']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution (equal to mu)."""
        return self.mu
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution (equal to mu)."""
        return self.mu
    
    @property
    def std(self) -> float:
        """Standard deviation of the fitted distribution."""
        return np.sqrt(self.mu)

