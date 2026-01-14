"""Tests for Geometric distribution."""

import numpy as np
import pytest

from bestdist.distributions.discrete import Geometric
from bestdist.utils.exceptions import InsufficientDataError


class TestGeometric:
    """Test cases for Geometric distribution."""
    
    def test_fit_geometric(self):
        """Test fitting Geometric distribution to data."""
        np.random.seed(42)
        data = np.random.geometric(p=0.3, size=1000)
        
        dist = Geometric(data)
        params = dist.fit()
        
        assert 'p' in params
        assert 0 < params['p'] <= 1
        assert np.isclose(params['p'], 0.3, atol=0.1)
        
    def test_geometric_properties(self):
        """Test Geometric distribution properties."""
        np.random.seed(42)
        data = np.random.geometric(p=0.2, size=1000)
        
        dist = Geometric(data)
        dist.fit()
        
        assert 0 < dist.p <= 1
        assert dist.mean > 0
        assert dist.variance > 0
        # For Geometric with p=0.2, mean should be around 5
        assert np.isclose(dist.mean, 1/0.2, rtol=0.3)
        
    def test_geometric_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1, 2]
        
        with pytest.raises(InsufficientDataError):
            Geometric(data)
            
    def test_geometric_pmf_cdf(self):
        """Test PMF and CDF methods."""
        np.random.seed(42)
        data = np.random.geometric(p=0.5, size=1000)
        
        dist = Geometric(data)
        dist.fit()
        
        k = np.array([1, 2, 3, 4, 5])
        pmf_vals = dist.pmf(k)
        cdf_vals = dist.cdf(k)
        
        assert len(pmf_vals) == 5
        assert len(cdf_vals) == 5
        assert np.all(pmf_vals >= 0)
        assert np.all(pmf_vals <= 1)
        assert np.all((cdf_vals >= 0) & (cdf_vals <= 1))
        # PMF should decrease monotonically for geometric
        assert np.all(np.diff(pmf_vals) <= 0)
        
    def test_geometric_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.geometric(p=0.3, size=1000)
        
        dist = Geometric(data)
        dist.fit()
        
        chi2_stat, p_value = dist.test_goodness_of_fit(method='chi2')
        
        assert chi2_stat >= 0
        assert 0 <= p_value <= 1

