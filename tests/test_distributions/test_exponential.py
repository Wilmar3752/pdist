"""Tests for Exponential distribution."""

import numpy as np
import pytest

from bestdist.distributions.continuous import Exponential
from bestdist.utils.exceptions import InsufficientDataError


class TestExponential:
    """Test cases for Exponential distribution."""
    
    def test_fit_exponential(self):
        """Test fitting exponential distribution to data."""
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=1000)
        
        dist = Exponential(data)
        params = dist.fit()
        
        assert 'loc' in params
        assert 'scale' in params
        assert params['scale'] > 0
        assert np.isclose(params['scale'], 2.0, atol=0.2)
        
    def test_exponential_properties(self):
        """Test exponential distribution properties."""
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=1000)
        
        dist = Exponential(data)
        dist.fit()
        
        assert dist.scale_param > 0
        assert dist.rate > 0
        assert dist.mean > 0
        assert np.isclose(dist.scale_param, 1/dist.rate, rtol=0.01)
        
    def test_exponential_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1.0, 2.0]
        
        with pytest.raises(InsufficientDataError):
            Exponential(data)
            
    def test_exponential_memoryless_property(self):
        """Test memoryless property approximately."""
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=10000)
        
        dist = Exponential(data)
        dist.fit()
        
        # P(X > 5) should equal P(X > 3+2 | X > 3) approximately
        prob_greater_5 = 1 - dist.cdf(5)
        prob_greater_3 = 1 - dist.cdf(3)
        prob_greater_2 = 1 - dist.cdf(2)
        
        # Due to memoryless: P(X>5)/P(X>3) ≈ P(X>2)
        assert np.isclose(prob_greater_5/prob_greater_3, prob_greater_2, rtol=0.01)
        
    def test_exponential_median(self):
        """Test exponential median."""
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=1000)
        
        dist = Exponential(data)
        dist.fit()
        
        # Median should be approximately scale * ln(2)
        expected_median = dist.scale_param * np.log(2)
        assert np.isclose(dist.median, expected_median, rtol=0.01)
        
    def test_exponential_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=1000)
        
        dist = Exponential(data)
        dist.fit()
        
        ks_stat, p_value = dist.test_goodness_of_fit(method='ks')
        
        assert ks_stat >= 0
        assert 0 <= p_value <= 1

