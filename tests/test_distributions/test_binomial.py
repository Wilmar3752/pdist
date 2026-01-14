"""Tests for Binomial distribution."""

import numpy as np
import pytest

from bestdist.distributions.discrete import Binomial
from bestdist.utils.exceptions import InsufficientDataError


class TestBinomial:
    """Test cases for Binomial distribution."""
    
    def test_fit_binomial(self):
        """Test fitting Binomial distribution to data."""
        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.3, size=1000)
        
        dist = Binomial(data)
        params = dist.fit()
        
        assert 'n' in params
        assert 'p' in params
        assert params['n'] > 0
        assert 0 < params['p'] < 1
        
    def test_binomial_properties(self):
        """Test Binomial distribution properties."""
        np.random.seed(42)
        data = np.random.binomial(n=20, p=0.4, size=1000)
        
        dist = Binomial(data)
        dist.fit()
        
        assert dist.n > 0
        assert 0 < dist.p < 1
        assert dist.mean > 0
        assert dist.variance > 0
        # For Binomial, variance < mean (underdispersion relative to Poisson)
        
    def test_binomial_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1, 2]
        
        with pytest.raises(InsufficientDataError):
            Binomial(data)
            
    def test_binomial_pmf_cdf(self):
        """Test PMF and CDF methods."""
        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.5, size=1000)
        
        dist = Binomial(data)
        dist.fit()
        
        k = np.array([0, 2, 4, 6, 8, 10])
        pmf_vals = dist.pmf(k)
        cdf_vals = dist.cdf(k)
        
        assert len(pmf_vals) == 6
        assert len(cdf_vals) == 6
        assert np.all(pmf_vals >= 0)
        assert np.all(pmf_vals <= 1)
        assert np.all((cdf_vals >= 0) & (cdf_vals <= 1))
        
    def test_binomial_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.5, size=1000)
        
        dist = Binomial(data)
        dist.fit()
        
        chi2_stat, p_value = dist.test_goodness_of_fit(method='chi2')
        
        assert chi2_stat >= 0
        assert 0 <= p_value <= 1

