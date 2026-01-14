"""Tests for Poisson distribution."""

import numpy as np
import pytest

from bestdist.distributions.discrete import Poisson
from bestdist.utils.exceptions import InsufficientDataError


class TestPoisson:
    """Test cases for Poisson distribution."""
    
    def test_fit_poisson(self):
        """Test fitting Poisson distribution to data."""
        np.random.seed(42)
        data = np.random.poisson(lam=3.5, size=1000)
        
        dist = Poisson(data)
        params = dist.fit()
        
        assert 'mu' in params
        assert params['mu'] > 0
        assert np.isclose(params['mu'], 3.5, atol=0.3)
        
    def test_poisson_properties(self):
        """Test Poisson distribution properties."""
        np.random.seed(42)
        data = np.random.poisson(lam=5.0, size=1000)
        
        dist = Poisson(data)
        dist.fit()
        
        assert dist.mu > 0
        assert dist.mean > 0
        assert dist.variance > 0
        # For Poisson, mean equals variance
        assert np.isclose(dist.mean, dist.variance, rtol=0.1)
        
    def test_poisson_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1, 2]
        
        with pytest.raises(InsufficientDataError):
            Poisson(data)
            
    def test_poisson_pmf_cdf(self):
        """Test PMF and CDF methods."""
        np.random.seed(42)
        data = np.random.poisson(lam=3.5, size=1000)
        
        dist = Poisson(data)
        dist.fit()
        
        k = np.array([0, 1, 2, 3, 4, 5])
        pmf_vals = dist.pmf(k)
        cdf_vals = dist.cdf(k)
        
        assert len(pmf_vals) == 6
        assert len(cdf_vals) == 6
        assert np.all(pmf_vals >= 0)
        assert np.all(pmf_vals <= 1)
        assert np.all((cdf_vals >= 0) & (cdf_vals <= 1))
        # CDF should be monotonically increasing
        assert np.all(np.diff(cdf_vals) >= 0)
        
    def test_poisson_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.poisson(lam=3.5, size=1000)
        
        dist = Poisson(data)
        dist.fit()
        
        chi2_stat, p_value = dist.test_goodness_of_fit(method='chi2')
        
        assert chi2_stat >= 0
        assert 0 <= p_value <= 1

