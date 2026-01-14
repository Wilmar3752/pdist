"""Tests for Lognormal distribution."""

import numpy as np
import pytest

from bestdist.distributions.continuous import Lognormal
from bestdist.utils.exceptions import InsufficientDataError


class TestLognormal:
    """Test cases for Lognormal distribution."""
    
    def test_fit_lognormal(self):
        """Test fitting lognormal distribution to data."""
        np.random.seed(42)
        data = np.random.lognormal(mean=1, sigma=0.5, size=1000)
        
        dist = Lognormal(data)
        params = dist.fit()
        
        assert 's' in params
        assert 'loc' in params
        assert 'scale' in params
        assert params['s'] > 0
        assert params['scale'] > 0
        
    def test_lognormal_properties(self):
        """Test lognormal distribution properties."""
        np.random.seed(42)
        data = np.random.lognormal(mean=1, sigma=0.5, size=1000)
        
        dist = Lognormal(data)
        dist.fit()
        
        assert dist.sigma > 0
        assert dist.mean > 0
        assert dist.variance > 0
        assert dist.median > 0
        
    def test_lognormal_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1.0, 2.0]
        
        with pytest.raises(InsufficientDataError):
            Lognormal(data)
            
    def test_lognormal_pdf_cdf(self):
        """Test PDF and CDF methods."""
        np.random.seed(42)
        data = np.random.lognormal(mean=1, sigma=0.5, size=1000)
        
        dist = Lognormal(data)
        dist.fit()
        
        x = np.array([1.0, 2.0, 3.0])
        pdf_vals = dist.pdf(x)
        cdf_vals = dist.cdf(x)
        
        assert len(pdf_vals) == 3
        assert len(cdf_vals) == 3
        assert np.all(pdf_vals >= 0)
        assert np.all((cdf_vals >= 0) & (cdf_vals <= 1))
        
    def test_lognormal_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.lognormal(mean=1, sigma=0.5, size=1000)
        
        dist = Lognormal(data)
        dist.fit()
        
        ks_stat, p_value = dist.test_goodness_of_fit(method='ks')
        
        assert ks_stat >= 0
        assert 0 <= p_value <= 1

