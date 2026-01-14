"""Tests for DistributionFitter."""

import pytest
import numpy as np
import pandas as pd
from bestdist import DistributionFitter
from bestdist.distributions.continuous import Normal, Gamma


class TestDistributionFitter:
    """Test suite for DistributionFitter."""
    
    def test_initialization(self, normal_data):
        """Test fitter initialization."""
        fitter = DistributionFitter(normal_data)
        assert len(fitter.data) > 0
        assert not fitter._fitted
    
    def test_fit_all_distributions(self, normal_data):
        """Test fitting all distributions."""
        fitter = DistributionFitter(normal_data)
        results = fitter.fit(verbose=False)
        
        assert len(results) > 0
        assert fitter._fitted
        
        # Check result structure
        for result in results:
            assert 'distribution' in result
            assert 'parameters' in result
            assert 'test_statistic' in result
            assert 'p_value' in result
            assert 'aic' in result
            assert 'bic' in result
    
    def test_best_distribution_normal(self, normal_data):
        """Test finding best distribution for normal data."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        best = fitter.get_best_distribution()
        
        assert best is not None
        # Normal should be among top distributions for normal data
        # (sometimes Gamma or other flexible distributions can have similar p-values)
        top_3_names = [r['distribution'] for r in fitter.results[:3]]
        assert 'Normal' in top_3_names, f"Normal not in top 3: {top_3_names}"
        assert best['p_value'] > 0.05
    
    def test_best_distribution_gamma(self, gamma_data):
        """Test finding best distribution for gamma data."""
        fitter = DistributionFitter(gamma_data)
        fitter.fit(verbose=False)
        
        best = fitter.get_best_distribution()
        
        assert best is not None
        # Gamma should be best or have high p-value
        assert best['p_value'] > 0.01
    
    def test_custom_distributions(self, normal_data):
        """Test with custom distribution list."""
        fitter = DistributionFitter(
            normal_data,
            distributions=[Normal, Gamma]
        )
        results = fitter.fit(verbose=False)
        
        assert len(results) == 2
        dist_names = [r['distribution'] for r in results]
        assert 'Normal' in dist_names
        assert 'Gamma' in dist_names
    
    def test_summary(self, normal_data):
        """Test summary DataFrame generation."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        summary = fitter.summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
        assert 'Distribution' in summary.columns
        assert 'P-Value' in summary.columns
        assert 'AIC' in summary.columns
        assert 'BIC' in summary.columns
    
    def test_summary_top_n(self, normal_data):
        """Test summary with top_n parameter."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        summary = fitter.summary(top_n=2)
        
        assert len(summary) == 2
    
    def test_criterion_aic(self, normal_data):
        """Test best distribution selection by AIC."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        best_pval = fitter.get_best_distribution(criterion='p_value')
        best_aic = fitter.get_best_distribution(criterion='aic')
        
        assert best_pval is not None
        assert best_aic is not None
        # AIC selection might differ from p-value
    
    def test_criterion_bic(self, normal_data):
        """Test best distribution selection by BIC."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        best_bic = fitter.get_best_distribution(criterion='bic')
        
        assert best_bic is not None
    
    def test_plot_best_fit(self, normal_data):
        """Test plotting functionality."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        fig = fitter.plot_best_fit()
        
        assert fig is not None
        # Check that figure has 2 subplots
        assert len(fig.axes) == 2
    
    def test_compare_distributions(self, normal_data):
        """Test comparison plots."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        fig = fitter.compare_distributions()
        
        assert fig is not None
        # Should have multiple subplots
        assert len(fig.axes) >= len(fitter.results)
    
    def test_pandas_series_input(self, pandas_series_data):
        """Test with pandas Series input."""
        fitter = DistributionFitter(pandas_series_data)
        results = fitter.fit(verbose=False)
        
        assert len(results) > 0
    
    def test_list_input(self):
        """Test with list input."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        fitter = DistributionFitter(data)
        results = fitter.fit(verbose=False)
        
        assert len(results) > 0
    
    def test_empty_data(self):
        """Test with empty data."""
        data = []
        
        with pytest.raises(ValueError):
            DistributionFitter(data)
    
    def test_invalid_criterion(self, normal_data):
        """Test invalid criterion."""
        fitter = DistributionFitter(normal_data)
        fitter.fit(verbose=False)
        
        with pytest.raises(ValueError):
            fitter.get_best_distribution(criterion='invalid')

