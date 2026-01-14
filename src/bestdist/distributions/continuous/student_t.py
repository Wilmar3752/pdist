"""Student-t distribution implementation."""

from typing import Tuple
from scipy.stats import t as student_t, rv_continuous
import numpy as np

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class StudentT(BaseDistribution):
    """
    Student's t-distribution.
    
    The t-distribution arises when estimating the mean of a normally distributed 
    population when the sample size is small and population standard deviation 
    is unknown. It has heavier tails than the normal distribution, making it 
    more robust to outliers.
    
    PDF: f(x) = Gamma((df+1)/2) / (sqrt(df*pi) * Gamma(df/2)) * (1 + x^2/df)^(-(df+1)/2)
    
    Parameters:
        df: Degrees of freedom (controls tail heaviness)
        loc: Location parameter (mean when df > 1)
        scale: Scale parameter
        
    Properties:
        - As df → ∞, converges to Normal distribution
        - df ≤ 1: mean undefined
        - df ≤ 2: variance undefined
        
    Example:
        ```python
        from bestdist.distributions.continuous import StudentT
        import numpy as np
        
        # Generate sample data (e.g., returns with outliers)
        data = np.random.standard_t(df=5, size=1000)
        
        # Fit distribution
        dist = StudentT(data)
        params = dist.fit()
        print(f"Degrees of freedom: {dist.df:.2f}")
        print(f"Mean: {dist.mean:.4f}")
        print(f"Variance: {dist.variance:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy Student-t distribution."""
        return student_t
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract df, location, and scale parameters from fit result.
        
        Args:
            fit_result: Tuple of (df, loc, scale) from scipy.stats.t.fit
            
        Returns:
            Dictionary with 'df', 'loc', and 'scale' parameters
        """
        return {
            'df': float(fit_result[0]),
            'loc': float(fit_result[1]),
            'scale': float(fit_result[2])
        }
    
    @property
    def df(self) -> float:
        """Degrees of freedom of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['df']
    
    @property
    def location(self) -> float:
        """Location parameter of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['loc']
    
    @property
    def scale_param(self) -> float:
        """Scale parameter of the fitted distribution."""
        if self.params is None:
            self.fit()
        return self.params['scale']
    
    @property
    def mean(self) -> float:
        """Mean of the fitted distribution (undefined if df <= 1)."""
        if self.df <= 1:
            return np.nan
        return self.location
    
    @property
    def variance(self) -> float:
        """Variance of the fitted distribution (undefined if df <= 2)."""
        df = self.df
        if df <= 1:
            return np.nan
        elif df <= 2:
            return np.inf
        return (self.scale_param ** 2) * df / (df - 2)
    
    @property
    def median(self) -> float:
        """Median of the fitted distribution."""
        return self.location

