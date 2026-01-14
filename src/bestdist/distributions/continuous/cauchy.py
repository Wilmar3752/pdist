"""Cauchy distribution implementation."""

from typing import Tuple
from scipy.stats import cauchy, rv_continuous
import numpy as np

from ...core.base import BaseDistribution
from ...utils.types import Parameters


class Cauchy(BaseDistribution):
    """
    Cauchy distribution.
    
    The Cauchy distribution is notable for having undefined mean and variance
    (heavy tails). It's the distribution of the ratio of two independent 
    standard normal random variables. Used in physics and robust statistics.
    
    PDF: f(x) = 1/(pi*scale * (1 + ((x-loc)/scale)^2))
    
    Parameters:
        loc: Location parameter (median and mode)
        scale: Scale parameter (half-width at half-maximum)
        
    Warning:
        Mean and variance are undefined (do not exist mathematically).
        Use median and IQR for central tendency and spread.
        
    Example:
        ```python
        from bestdist.distributions.continuous import Cauchy
        import numpy as np
        
        # Generate sample data (e.g., physics measurements with outliers)
        data = np.random.standard_cauchy(size=1000)
        
        # Fit distribution
        dist = Cauchy(data)
        params = dist.fit()
        print(f"Location: {dist.location:.4f}")
        print(f"Scale: {dist.scale_param:.4f}")
        print(f"Median: {dist.median:.4f}")
        print(f"IQR: {dist.iqr:.4f}")
        ```
    """
    
    def _get_scipy_dist(self) -> rv_continuous:
        """Return scipy Cauchy distribution."""
        return cauchy
    
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract location and scale parameters from fit result.
        
        Args:
            fit_result: Tuple of (loc, scale) from scipy.stats.cauchy.fit
            
        Returns:
            Dictionary with 'loc' and 'scale' parameters
        """
        return {
            'loc': float(fit_result[0]),
            'scale': float(fit_result[1])
        }
    
    @property
    def location(self) -> float:
        """Location parameter (median) of the fitted distribution."""
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
    def median(self) -> float:
        """Median of the fitted distribution."""
        return self.location
    
    @property
    def iqr(self) -> float:
        """Interquartile range (Q3 - Q1)."""
        q1 = self.ppf(0.25)
        q3 = self.ppf(0.75)
        return q3 - q1
    
    @property
    def mean(self) -> float:
        """Mean is undefined for Cauchy distribution."""
        return np.nan
    
    @property
    def variance(self) -> float:
        """Variance is undefined for Cauchy distribution."""
        return np.nan

