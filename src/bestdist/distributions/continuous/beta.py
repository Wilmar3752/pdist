"""Beta distribution implementation."""

from typing import Tuple
from scipy.stats import beta, rv_continuous

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Beta(BaseDistribution):
    """
    Beta distribution.
    
    The beta distribution is defined on the interval [0, 1] and is commonly
    used to model probabilities and proportions. It's the conjugate prior 
    for the Bernoulli and binomial distributions.
    
    PDF: f(x) = (x^(a-1) * (1-x)^(b-1)) / B(a, b)
    
    Parameters:
        a (alpha): First shape parameter (must be positive)
        b (beta): Second shape parameter (must be positive)
        loc: Location parameter (default 0)
        scale: Scale parameter (default 1)
        
    Example:
        ```python
        from pdist.distributions.continuous import Beta
        import numpy as np
        
        # Generate sample data (proportions)
        data = np.random.beta(a=2, b=5, size=1000)
        
        # Fit distribution
        dist = Beta(data)
        params = dist.fit()
        print(f"Alpha: {params['a']:.4f}, Beta: {params['b']:.4f}")
        
        # Test goodness of fit
        ks_stat, p_value = dist.test_goodness_of_fit()
        print(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
        ```
        
    Note:
        - Data should be in the range [0, 1] for standard beta distribution
        - For data outside [0, 1], loc and scale parameters will be fitted
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy beta distribution."""
        return beta
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract alpha and beta parameters from fit result.
        
        Args:
            fit_result: Tuple of (a, b, loc, scale) from scipy.stats.beta.fit
            
        Returns:
            Dictionary with 'a' (alpha), 'b' (beta), 'loc', and 'scale'
        """
        return {
            'a': float(fit_result[0]),
            'b': float(fit_result[1]),
            'loc': float(fit_result[2]),
            'scale': float(fit_result[3])
        }
    
    @property
    def alpha(self) -> float:
        """Alpha (first shape) parameter of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['a']
    
    @property
    def beta_param(self) -> float:
        """Beta (second shape) parameter of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['b']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution."""
        a = self.alpha
        b = self.beta_param
        return a / (a + b)
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution."""
        a = self.alpha
        b = self.beta_param
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    @property
    def mode(self) -> float:
        """Mode of the fitted distribution (if a, b > 1)."""
        a = self.alpha
        b = self.beta_param
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        elif a == 1 and b == 1:
            return 0.5  # Uniform distribution, any value is a mode
        elif a < 1 and b < 1:
            raise ValueError("Mode is undefined when both a < 1 and b < 1")
        elif a <= 1:
            return 0.0
        else:  # b <= 1
            return 1.0

