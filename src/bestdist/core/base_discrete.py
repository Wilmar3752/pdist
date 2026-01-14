"""Abstract base class for discrete probability distributions."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy.stats import rv_discrete

from ..utils.types import ArrayLike, TestResult, Parameters
from ..utils.exceptions import FittingError, InsufficientDataError


class BaseDiscreteDistribution(ABC):
    """
    Abstract base class for all discrete probability distributions.
    
    This class defines the interface that all discrete distribution implementations
    must follow, ensuring consistency across the package.
    
    Attributes:
        name: Name of the distribution
        data: Input data array (integers)
        params: Fitted distribution parameters
        dist: Fitted scipy distribution object
        
    Example:
        ```python
        class Poisson(BaseDiscreteDistribution):
            def _get_scipy_dist(self):
                from scipy.stats import poisson
                return poisson
                
            def _extract_params(self, fit_result):
                return {'mu': fit_result[0]}
        ```
    """
    
    def __init__(self, data: ArrayLike, name: Optional[str] = None):
        """
        Initialize the distribution with data.
        
        Args:
            data: Input data to fit the distribution to (should be integers)
            name: Optional name for the distribution (defaults to class name)
            
        Raises:
            InsufficientDataError: If data has fewer than 3 observations
            ValueError: If data contains NaN or infinite values
        """
        self.name = name or self.__class__.__name__
        self.data = self._validate_and_prepare_data(data)
        self.params: Optional[Parameters] = None
        self.dist: Optional[rv_discrete] = None
        self._fitted = False
        
    def _validate_and_prepare_data(self, data: ArrayLike) -> np.ndarray:
        """
        Validate and convert input data to numpy array of integers.
        
        Args:
            data: Input data
            
        Returns:
            Validated numpy array of integers
            
        Raises:
            InsufficientDataError: If data has fewer than 3 observations
            ValueError: If data contains NaN or infinite values
        """
        # Convert to numpy array
        if isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=int)
        elif hasattr(data, 'values'):  # pandas Series
            arr = data.values.astype(int)
        else:
            arr = np.asarray(data, dtype=int)
            
        # Remove NaN values (convert to int first removes floats)
        original_arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(original_arr)]
        
        # Check for insufficient data
        if len(arr) < 3:
            raise InsufficientDataError(
                f"Need at least 3 observations, got {len(arr)}"
            )
            
        # Check for negative values in discrete data
        if np.any(arr < 0):
            raise ValueError("Discrete data cannot contain negative values")
            
        return arr
    
    @abstractmethod
    def _get_scipy_dist(self) -> rv_discrete:
        """
        Return the scipy.stats distribution object.
        
        Returns:
            scipy.stats discrete distribution object
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
            
            # For discrete distributions, we often need to use method of moments
            # or maximum likelihood estimation
            if hasattr(self, '_fit_custom'):
                self.params = self._fit_custom()
            else:
                # Default: use scipy's fit method
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
        method: str = 'chi2'
    ) -> TestResult:
        """
        Perform goodness-of-fit test.
        
        Args:
            method: Test method ('chi2' for Chi-square, 'ks' for Kolmogorov-Smirnov)
                   
        Returns:
            Tuple of (test_statistic, p_value)
            
        Raises:
            ValueError: If distribution hasn't been fitted
        """
        if not self._fitted:
            self.fit()
            
        if method == 'chi2':
            from scipy.stats import chisquare
            
            # Get observed frequencies
            unique_vals, observed_counts = np.unique(self.data, return_counts=True)
            
            # Get expected frequencies
            expected_probs = self.pmf(unique_vals)
            expected_counts = len(self.data) * expected_probs
            
            # Remove categories with expected count < 5
            mask = expected_counts >= 5
            observed_counts = observed_counts[mask]
            expected_counts = expected_counts[mask]
            
            if len(observed_counts) < 2:
                return 0.0, 1.0
            
            # Normalize expected counts to match observed total (fix scipy issue)
            expected_counts = expected_counts * (observed_counts.sum() / expected_counts.sum())
            
            statistic, p_value = chisquare(observed_counts, expected_counts)
            
        elif method == 'ks':
            from scipy.stats import ks_1samp
            statistic, p_value = ks_1samp(self.data, self.dist.cdf)
        else:
            raise ValueError(f"Unknown test method: {method}")
            
        return statistic, p_value
    
    def pmf(self, k: ArrayLike) -> np.ndarray:
        """
        Probability mass function.
        
        Args:
            k: Points at which to evaluate the PMF
            
        Returns:
            PMF values
        """
        if not self._fitted:
            self.fit()
        return self.dist.pmf(k)
    
    def cdf(self, k: ArrayLike) -> np.ndarray:
        """
        Cumulative distribution function.
        
        Args:
            k: Points at which to evaluate the CDF
            
        Returns:
            CDF values
        """
        if not self._fitted:
            self.fit()
        return self.dist.cdf(k)
    
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
            
        chi2_stat, p_value = self.test_goodness_of_fit(method='chi2')
        
        return {
            'name': self.name,
            'parameters': self.params,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'n_observations': len(self.data),
        }
    
    def __repr__(self) -> str:
        """String representation of the distribution."""
        if self._fitted:
            param_str = ', '.join(f"{k}={v:.4f}" for k, v in self.params.items())
            return f"{self.name}({param_str})"
        return f"{self.name}(not fitted)"

