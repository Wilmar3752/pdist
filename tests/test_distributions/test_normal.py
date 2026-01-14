"""Tests for Normal distribution."""

import pytest
import numpy as np
from bestdist.distributions.continuous import Normal
from bestdist.utils.exceptions import InsufficientDataError


class TestNormal:
    """Test suite for Normal distribution."""
    
    def test_initialization(self, normal_data):
        """Test distribution initialization."""
        dist = Normal(normal_data)
        assert dist.name == "Normal"
        assert len(dist.data) == len(normal_data)
        assert not dist._fitted
    
    def test_fit(self, normal_data):
        """Test distribution fitting."""
        dist = Normal(normal_data)
        params = dist.fit()
        
        assert 'loc' in params
        assert 'scale' in params
        assert dist._fitted
        assert pytest.approx(params['loc'], abs=0.2) == 10.0
        assert pytest.approx(params['scale'], abs=0.3) == 3.0
    
    def test_properties(self, normal_data):
        """Test distribution properties."""
        dist = Normal(normal_data)
        dist.fit()
        
        assert pytest.approx(dist.mean, abs=0.2) == 10.0
        assert pytest.approx(dist.std, abs=0.3) == 3.0
        assert pytest.approx(dist.variance, abs=2.0) == 9.0
    
    def test_goodness_of_fit(self, normal_data):
        """Test goodness-of-fit test."""
        dist = Normal(normal_data)
        statistic, p_value = dist.test_goodness_of_fit()
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1
        # For data from normal distribution, p_value should be high
        assert p_value > 0.01
    
    def test_pdf(self, normal_data):
        """Test PDF calculation."""
        dist = Normal(normal_data)
        dist.fit()
        
        x = np.array([3, 5, 7])
        pdf_values = dist.pdf(x)
        
        assert len(pdf_values) == len(x)
        assert all(pdf_values >= 0)
    
    def test_cdf(self, normal_data):
        """Test CDF calculation."""
        dist = Normal(normal_data)
        dist.fit()
        
        x = np.array([3, 5, 7])
        cdf_values = dist.cdf(x)
        
        assert len(cdf_values) == len(x)
        assert all(0 <= v <= 1 for v in cdf_values)
        # CDF should be increasing
        assert cdf_values[0] < cdf_values[1] < cdf_values[2]
    
    def test_ppf(self, normal_data):
        """Test percent point function."""
        dist = Normal(normal_data)
        dist.fit()
        
        quantiles = np.array([0.25, 0.5, 0.75])
        values = dist.ppf(quantiles)
        
        assert len(values) == len(quantiles)
        # Values should be increasing
        assert values[0] < values[1] < values[2]
        # Median should be close to mean
        assert pytest.approx(values[1], abs=0.2) == dist.mean
    
    def test_rvs(self, normal_data):
        """Test random variate generation."""
        dist = Normal(normal_data)
        dist.fit()
        
        samples = dist.rvs(size=100, random_state=42)
        
        assert len(samples) == 100
        # Generated samples should have similar statistics
        assert pytest.approx(np.mean(samples), abs=1.0) == dist.mean
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = np.array([1.0, 2.0])  # Only 2 observations
        
        with pytest.raises(InsufficientDataError):
            Normal(data)
    
    def test_pandas_series(self, pandas_series_data):
        """Test with pandas Series input."""
        dist = Normal(pandas_series_data)
        params = dist.fit()
        
        assert 'loc' in params
        assert 'scale' in params
    
    def test_data_with_nans(self, data_with_nans):
        """Test with data containing NaN values."""
        dist = Normal(data_with_nans)
        params = dist.fit()
        
        # NaN values should be removed
        assert not np.isnan(dist.data).any()
        assert 'loc' in params
    
    def test_repr(self, normal_data):
        """Test string representation."""
        dist = Normal(normal_data)
        
        # Before fitting
        assert "not fitted" in repr(dist)
        
        # After fitting
        dist.fit()
        assert "Normal" in repr(dist)
        assert "loc=" in repr(dist)
        assert "scale=" in repr(dist)
    
    def test_get_info(self, normal_data):
        """Test get_info method."""
        dist = Normal(normal_data)
        info = dist.get_info()
        
        assert 'name' in info
        assert 'parameters' in info
        assert 'ks_statistic' in info
        assert 'p_value' in info
        assert 'n_observations' in info
        assert info['name'] == 'Normal'
        assert info['n_observations'] == len(normal_data)

