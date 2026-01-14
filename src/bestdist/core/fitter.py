"""Main distribution fitter for finding the best distribution."""

from typing import List, Optional, Dict, Any, Type, Union, Literal
import warnings
import numpy as np
import pandas as pd

from ..core.base import BaseDistribution
from ..core.base_discrete import BaseDiscreteDistribution
from ..distributions.continuous.normal import Normal
from ..distributions.continuous.gamma import Gamma
from ..distributions.continuous.beta import Beta
from ..distributions.continuous.weibull import Weibull
from ..distributions.continuous.lognormal import Lognormal
from ..distributions.continuous.exponential import Exponential
from ..distributions.continuous.uniform import Uniform
from ..distributions.continuous.cauchy import Cauchy
from ..distributions.continuous.student_t import StudentT
from ..distributions.discrete.poisson import Poisson
from ..distributions.discrete.binomial import Binomial
from ..distributions.discrete.negative_binomial import NegativeBinomial
from ..distributions.discrete.geometric import Geometric
from ..utils.types import ArrayLike, FitResult
from ..utils.exceptions import FittingError, InvalidDistributionError


class DistributionFitter:
    """
    Fit multiple probability distributions and find the best one.
    
    This class provides a convenient interface for fitting multiple
    distributions to data and ranking them by goodness-of-fit.
    
    Example:
        ```python
        from pdist import DistributionFitter
        import numpy as np
        
        # Generate sample data
        data = np.random.gamma(2, 2, 1000)
        
        # Fit all available distributions
        fitter = DistributionFitter(data)
        results = fitter.fit()
        
        # Get best distribution
        best = fitter.get_best_distribution()
        print(f"Best fit: {best['distribution']}")
        print(f"Parameters: {best['parameters']}")
        print(f"P-value: {best['p_value']:.4f}")
        
        # Get summary
        summary = fitter.summary()
        print(summary)
        ```
    """
    
    # Default distributions to try
    DEFAULT_CONTINUOUS_DISTRIBUTIONS = [
        Normal, Gamma, Beta, Weibull,
        Lognormal, Exponential, Uniform, Cauchy, StudentT
    ]
    
    DEFAULT_DISCRETE_DISTRIBUTIONS = [
        Poisson, Binomial, NegativeBinomial, Geometric
    ]
    
    def __init__(
        self,
        data: ArrayLike,
        distributions: Optional[List[Union[Type[BaseDistribution], Type[BaseDiscreteDistribution]]]] = None,
        dist_type: Literal['continuous', 'discrete'] = 'continuous',
        method: Optional[str] = None
    ):
        """
        Initialize the fitter.
        
        Args:
            data: Input data to fit distributions to
            distributions: List of distribution classes to try.
                          If None, uses defaults based on dist_type
            dist_type: Type of distributions ('continuous' or 'discrete')
            method: Goodness-of-fit test method. 
                   If None, uses 'ks' for continuous, 'chi2' for discrete
        """
        self.dist_type = dist_type
        self.data = self._prepare_data(data)
        
        # Set default distributions based on type
        if distributions is None:
            if dist_type == 'continuous':
                self.distributions = self.DEFAULT_CONTINUOUS_DISTRIBUTIONS
            else:
                self.distributions = self.DEFAULT_DISCRETE_DISTRIBUTIONS
        else:
            self.distributions = distributions
        
        # Set default method based on type
        if method is None:
            self.method = 'ks' if dist_type == 'continuous' else 'chi2'
        else:
            self.method = method
            
        self.results: List[FitResult] = []
        self._fitted = False
        
    def _prepare_data(self, data: ArrayLike) -> np.ndarray:
        """Prepare and validate input data."""
        if isinstance(data, pd.Series):
            data = data.dropna().values
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
            
        # Remove NaN values
        data_float = np.asarray(data, dtype=float)
        data = data[~np.isnan(data_float)]
        
        if len(data) == 0:
            raise ValueError("Data is empty after removing NaN values")
        
        # For discrete distributions, convert to integers
        if self.dist_type == 'discrete':
            data = np.asarray(data, dtype=int)
        else:
            data = np.asarray(data, dtype=float)
            
        return data
    
    def fit(self, verbose: bool = True, suppress_warnings: bool = True) -> List[FitResult]:
        """
        Fit all distributions to the data.
        
        Args:
            verbose: If True, print progress and fitting errors
            suppress_warnings: If True, suppress scipy/numpy warnings during fitting
                              (recommended, as some distributions may not be suitable)
            
        Returns:
            List of fit results, sorted by p-value (descending)
        """
        self.results = []
        
        # Context manager for suppressing warnings
        warning_context = warnings.catch_warnings() if suppress_warnings else None
        
        if suppress_warnings:
            warning_context.__enter__()
            warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        try:
            for dist_class in self.distributions:
                try:
                    # Create and fit distribution
                    dist = dist_class(self.data)
                    params = dist.fit()
                    
                    # Test goodness of fit
                    statistic, p_value = dist.test_goodness_of_fit(method=self.method)
                    
                    # Store results
                    result: FitResult = {
                        'distribution': dist.name,
                        'distribution_object': dist,
                        'parameters': params,
                        'test_statistic': float(statistic),
                        'p_value': float(p_value) if p_value is not None else None,
                        'aic': self._calculate_aic(dist),
                        'bic': self._calculate_bic(dist),
                    }
                    
                    self.results.append(result)
                    
                except Exception as e:
                    if verbose:
                        warnings.warn(
                            f"Failed to fit {dist_class.__name__}: {str(e)}",
                            RuntimeWarning
                        )
                    continue
        finally:
            if suppress_warnings and warning_context:
                warning_context.__exit__(None, None, None)
        
        # Sort by p-value (higher is better)
        self.results.sort(
            key=lambda x: x['p_value'] if x['p_value'] is not None else -1,
            reverse=True
        )
        
        self._fitted = True
        return self.results
    
    def _calculate_aic(self, dist: Union[BaseDistribution, BaseDiscreteDistribution]) -> float:
        """
        Calculate Akaike Information Criterion.
        
        AIC = 2k - 2ln(L)
        where k is number of parameters and L is likelihood
        """
        k = len(dist.params)
        
        # Use pmf for discrete, pdf for continuous
        if isinstance(dist, BaseDiscreteDistribution):
            probs = dist.pmf(self.data)
        else:
            probs = dist.pdf(self.data)
        
        log_likelihood = np.sum(np.log(probs + 1e-10))
        return 2 * k - 2 * log_likelihood
    
    def _calculate_bic(self, dist: Union[BaseDistribution, BaseDiscreteDistribution]) -> float:
        """
        Calculate Bayesian Information Criterion.
        
        BIC = k*ln(n) - 2ln(L)
        where k is number of parameters, n is sample size, L is likelihood
        """
        k = len(dist.params)
        n = len(self.data)
        
        # Use pmf for discrete, pdf for continuous
        if isinstance(dist, BaseDiscreteDistribution):
            probs = dist.pmf(self.data)
        else:
            probs = dist.pdf(self.data)
        
        log_likelihood = np.sum(np.log(probs + 1e-10))
        return k * np.log(n) - 2 * log_likelihood
    
    def get_best_distribution(
        self,
        criterion: str = 'p_value'
    ) -> Optional[FitResult]:
        """
        Get the best fitting distribution.
        
        Args:
            criterion: Criterion for selection ('p_value', 'aic', 'bic')
                      - 'p_value': Higher is better
                      - 'aic': Lower is better
                      - 'bic': Lower is better
                      
        Returns:
            Best fit result or None if no distributions were fitted
        """
        if not self._fitted:
            self.fit()
            
        if not self.results:
            return None
            
        if criterion == 'p_value':
            return self.results[0]  # Already sorted by p_value
        elif criterion == 'aic':
            return min(self.results, key=lambda x: x['aic'])
        elif criterion == 'bic':
            return min(self.results, key=lambda x: x['bic'])
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def summary(self, top_n: int = None) -> pd.DataFrame:
        """
        Get a summary DataFrame of all fitted distributions.
        
        Args:
            top_n: Number of top results to include (None for all)
            
        Returns:
            DataFrame with distribution names, parameters, and test statistics
        """
        if not self._fitted:
            self.fit()
            
        if not self.results:
            return pd.DataFrame()
            
        results = self.results[:top_n] if top_n else self.results
        
        summary_data = []
        for result in results:
            row = {
                'Distribution': result['distribution'],
                'Test Statistic': result['test_statistic'],
                'P-Value': result['p_value'],
                'AIC': result['aic'],
                'BIC': result['bic'],
            }
            # Add parameters
            for param_name, param_value in result['parameters'].items():
                row[f'param_{param_name}'] = param_value
                
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)
    
    def plot_best_fit(
        self,
        bins: int = 30,
        figsize: tuple = (12, 6)
    ) -> Any:
        """
        Plot histogram of data with best fit distribution overlay.
        
        Args:
            bins: Number of histogram bins
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        
        if not self._fitted:
            self.fit()
            
        best = self.get_best_distribution()
        if best is None:
            raise FittingError("No distributions were successfully fitted")
            
        dist_obj = best['distribution_object']
        is_discrete = isinstance(dist_obj, BaseDiscreteDistribution)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram with PDF/PMF overlay
        if is_discrete:
            # For discrete: use bar plot
            unique_vals, counts = np.unique(self.data, return_counts=True)
            probs = counts / len(self.data)
            ax1.bar(unique_vals, probs, alpha=0.7, color='skyblue', 
                   edgecolor='black', label='Data', width=0.8)
            
            # Plot PMF
            x_range = np.arange(self.data.min(), self.data.max() + 1)
            ax1.plot(x_range, dist_obj.pmf(x_range), 'ro-', lw=2,
                    label=f'{best["distribution"]} PMF', markersize=4)
            ax1.set_ylabel('Probability')
        else:
            # For continuous: use histogram
            ax1.hist(self.data, bins=bins, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', label='Data')
            
            x_range = np.linspace(self.data.min(), self.data.max(), 1000)
            ax1.plot(x_range, dist_obj.pdf(x_range), 'r-', lw=2,
                    label=f'{best["distribution"]} PDF')
            ax1.set_ylabel('Density')
        
        ax1.set_xlabel('Value')
        ax1.set_title(f'Best Fit: {best["distribution"]} (p={best["p_value"]:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        theoretical_quantiles = np.linspace(0.01, 0.99, 100)
        empirical_quantiles = np.percentile(self.data, theoretical_quantiles * 100)
        fitted_quantiles = dist_obj.ppf(theoretical_quantiles)
        
        ax2.scatter(fitted_quantiles, empirical_quantiles, alpha=0.5)
        
        # Perfect fit line
        min_val = min(fitted_quantiles.min(), empirical_quantiles.min())
        max_val = max(fitted_quantiles.max(), empirical_quantiles.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Perfect Fit')
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Empirical Quantiles')
        ax2.set_title('Q-Q Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compare_distributions(
        self,
        bins: int = 30,
        figsize: tuple = (15, 10)
    ) -> Any:
        """
        Create comparison plots for all fitted distributions.
        
        Args:
            bins: Number of histogram bins
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        
        if not self._fitted:
            self.fit()
            
        n_dists = len(self.results)
        if n_dists == 0:
            raise FittingError("No distributions were successfully fitted")
            
        n_cols = 2
        n_rows = (n_dists + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        x_range = np.linspace(self.data.min(), self.data.max(), 1000)
        
        for idx, result in enumerate(self.results):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            dist_obj = result['distribution_object']
            is_discrete = isinstance(dist_obj, BaseDiscreteDistribution)
            
            # Plot data and fitted distribution
            if is_discrete:
                # For discrete: use bar plot
                unique_vals, counts = np.unique(self.data, return_counts=True)
                probs = counts / len(self.data)
                ax.bar(unique_vals, probs, alpha=0.5, color='skyblue', 
                      edgecolor='black', label='Data', width=0.8)
                
                # Plot PMF
                x_discrete = np.arange(self.data.min(), self.data.max() + 1)
                ax.plot(x_discrete, dist_obj.pmf(x_discrete), 'ro-', lw=2,
                       label=f'{result["distribution"]} PMF', markersize=3)
            else:
                # For continuous: use histogram
                ax.hist(self.data, bins=bins, density=True, alpha=0.5,
                       color='skyblue', edgecolor='black', label='Data')
                
                # PDF overlay
                ax.plot(x_range, dist_obj.pdf(x_range), 'r-', lw=2,
                       label=f'{result["distribution"]} PDF')
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Density' if not is_discrete else 'Probability')
            ax.set_title(
                f'{result["distribution"]}\n'
                f'p-value: {result["p_value"]:.4f}, '
                f'AIC: {result["aic"]:.2f}'
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide extra subplots
        for idx in range(n_dists, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
            
        plt.tight_layout()
        return fig

