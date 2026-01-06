"""Abstract base class for probability distributions."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.stats import rv_continuous

from ..utils.types import ArrayLike, TestResult, Parameters
from ..utils.exceptions import FittingError, InsufficientDataError


class BaseDistribution(ABC):
    """
    Abstract base class for all probability distributions.
    
    This class defines the interface that all distribution implementations
    must follow, ensuring consistency across the package.
    
    Attributes:
        name: Name of the distribution
        data: Input data array
        params: Fitted distribution parameters
        dist: Fitted scipy distribution object
        
    Example:
        ```python
        class Normal(BaseDistribution):
            def _get_scipy_dist(self):
                from scipy.stats import norm
                return norm
                
            def _extract_params(self, fit_result):
                return {'loc': fit_result[0], 'scale': fit_result[1]}
        ```
    """
    
    def __init__(self, data: ArrayLike, name: Optional[str] = None):
        """
        Initialize the distribution with data.
        
        Args:
            data: Input data to fit the distribution to
            name: Optional name for the distribution (defaults to class name)
            
        Raises:
            InsufficientDataError: If data has fewer than 3 observations
            ValueError: If data contains NaN or infinite values
        """
        self.name = name or self.__class__.__name__
        self.data = self._validate_and_prepare_data(data)
        self.params: Optional[Parameters] = None
        self.dist: Optional[rv_continuous] = None
        self._fitted = False
        
    def _validate_and_prepare_data(self, data: ArrayLike) -> np.ndarray:
        """
        Validate and convert input data to numpy array.
        
        Args:
            data: Input data
            
        Returns:
            Validated numpy array
            
        Raises:
            InsufficientDataError: If data has fewer than 3 observations
            ValueError: If data contains NaN or infinite values
        """
        # Convert to numpy array
        if isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=float)
        elif hasattr(data, 'values'):  # pandas Series
            arr = data.values
        else:
            arr = np.asarray(data, dtype=float)
            
        # Remove NaN values
        arr = arr[~np.isnan(arr)]
        
        # Check for insufficient data
        if len(arr) < 3:
            raise InsufficientDataError(
                f"Need at least 3 observations, got {len(arr)}"
            )
            
        # Check for infinite values
        if not np.isfinite(arr).all():
            raise ValueError("Data contains infinite values")
            
        return arr
    
    @abstractmethod
    def _get_scipy_dist(self) -> rv_continuous:
        """
        Return the scipy.stats distribution object.
        
        Returns:
            scipy.stats continuous distribution object
        """
        pass
    
    @abstractmethod
    def _extract_params(self, fit_result: Tuple) -> Parameters:
        """
        Extract parameters from scipy fit result.
        
        Args:
            fit_result: Result from scipy distribution fit method
            
        Returns:
            Dictionary of parameter names and values
        """
        pass
    
    def fit(self) -> Parameters:
        """
        Fit the distribution to the data.
        
        Returns:
            Dictionary of fitted parameters
            
        Raises:
            FittingError: If fitting fails
        """
        try:
            scipy_dist = self._get_scipy_dist()
            fit_result = scipy_dist.fit(self.data)
            self.params = self._extract_params(fit_result)
            self.dist = scipy_dist(**self.params)
            self._fitted = True
            return self.params
            
        except Exception as e:
            raise FittingError(
                f"Failed to fit {self.name} distribution: {str(e)}"
            ) from e
    
    def test_goodness_of_fit(
        self, 
        method: str = 'ks'
    ) -> TestResult:
        """
        Perform goodness-of-fit test.
        
        Args:
            method: Test method ('ks' for Kolmogorov-Smirnov,
                   'ad' for Anderson-Darling, 'chi2' for Chi-square)
                   
        Returns:
            Tuple of (test_statistic, p_value)
            
        Raises:
            ValueError: If distribution hasn't been fitted
        """
        if not self._fitted:
            self.fit()
            
        if method == 'ks':
            from scipy.stats import ks_1samp
            statistic, p_value = ks_1samp(self.data, self.dist.cdf)
        elif method == 'ad':
            from scipy.stats import anderson
            result = anderson(self.data)
            statistic, p_value = result.statistic, None  # AD doesn't return p-value directly
        elif method == 'chi2':
            from scipy.stats import chisquare
            observed, _ = np.histogram(self.data, bins='auto')
            expected = len(self.data) * np.diff(
                self.dist.cdf(np.histogram_bin_edges(self.data, bins='auto'))
            )
            statistic, p_value = chisquare(observed, expected)
        else:
            raise ValueError(f"Unknown test method: {method}")
            
        return statistic, p_value
    
    def pdf(self, x: ArrayLike) -> np.ndarray:
        """
        Probability density function.
        
        Args:
            x: Points at which to evaluate the PDF
            
        Returns:
            PDF values
        """
        if not self._fitted:
            self.fit()
        return self.dist.pdf(x)
    
    def cdf(self, x: ArrayLike) -> np.ndarray:
        """
        Cumulative distribution function.
        
        Args:
            x: Points at which to evaluate the CDF
            
        Returns:
            CDF values
        """
        if not self._fitted:
            self.fit()
        return self.dist.cdf(x)
    
    def ppf(self, q: ArrayLike) -> np.ndarray:
        """
        Percent point function (inverse of CDF).
        
        Args:
            q: Quantiles (0 to 1)
            
        Returns:
            Values at given quantiles
        """
        if not self._fitted:
            self.fit()
        return self.dist.ppf(q)
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the distribution.
        
        Args:
            size: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            Random samples
        """
        if not self._fitted:
            self.fit()
        return self.dist.rvs(size=size, random_state=random_state)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get distribution information.
        
        Returns:
            Dictionary with distribution info including name, parameters,
            and goodness-of-fit statistics
        """
        if not self._fitted:
            self.fit()
            
        ks_stat, p_value = self.test_goodness_of_fit(method='ks')
        
        return {
            'name': self.name,
            'parameters': self.params,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'n_observations': len(self.data),
        }
    
    def __repr__(self) -> str:
        """String representation of the distribution."""
        if self._fitted:
            param_str = ', '.join(f"{k}={v:.4f}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return f"{self.name}(not fitted)"

