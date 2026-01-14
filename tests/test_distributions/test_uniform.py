"""Tests for Uniform distribution."""

import numpy as np
import pytest

from bestdist.distributions.continuous import Uniform
from bestdist.utils.exceptions import InsufficientDataError


class TestUniform:
    """Test cases for Uniform distribution."""
    
    def test_fit_uniform(self):
        """Test fitting uniform distribution to data."""
        np.random.seed(42)
        data = np.random.uniform(low=10, high=50, size=1000)
        
        dist = Uniform(data)
        params = dist.fit()
        
        assert 'loc' in params
        assert 'scale' in params
        assert params['loc'] >= 10
        assert params['loc'] + params['scale'] <= 50
        
    def test_uniform_properties(self):
        """Test uniform distribution properties."""
        np.random.seed(42)
        data = np.random.uniform(low=10, high=50, size=1000)
        
        dist = Uniform(data)
        dist.fit()
        
        assert dist.lower_bound >= 9.5  # Allow some tolerance
        assert dist.upper_bound <= 50.5
        assert 25 <= dist.mean <= 35  # Mean should be around 30
        assert dist.variance > 0
        
    def test_uniform_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1.0, 2.0]
        
        with pytest.raises(InsufficientDataError):
            Uniform(data)
            
    def test_uniform_mean_is_median(self):
        """Test that mean equals median for uniform distribution."""
        np.random.seed(42)
        data = np.random.uniform(low=10, high=50, size=1000)
        
        dist = Uniform(data)
        dist.fit()
        
        assert np.isclose(dist.mean, dist.median, rtol=0.01)
        
    def test_uniform_pdf_constant(self):
        """Test that PDF is approximately constant within bounds."""
        np.random.seed(42)
        data = np.random.uniform(low=10, high=50, size=1000)
        
        dist = Uniform(data)
        dist.fit()
        
        # Test PDF at multiple points within the range
        x = np.linspace(dist.lower_bound + 1, dist.upper_bound - 1, 5)
        pdf_vals = dist.pdf(x)
        
        # All PDF values should be approximately equal
        assert np.allclose(pdf_vals, pdf_vals[0], rtol=0.1)
        
    def test_uniform_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.uniform(low=10, high=50, size=1000)
        
        dist = Uniform(data)
        dist.fit()
        
        ks_stat, p_value = dist.test_goodness_of_fit(method='ks')
        
        assert ks_stat >= 0
        assert 0 <= p_value <= 1

