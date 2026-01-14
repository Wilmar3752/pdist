"""Tests for Cauchy distribution."""

import numpy as np
import pytest

from bestdist.distributions.continuous import Cauchy
from bestdist.utils.exceptions import InsufficientDataError


class TestCauchy:
    """Test cases for Cauchy distribution."""
    
    def test_fit_cauchy(self):
        """Test fitting Cauchy distribution to data."""
        np.random.seed(42)
        data = np.random.standard_cauchy(size=1000)
        
        dist = Cauchy(data)
        params = dist.fit()
        
        assert 'loc' in params
        assert 'scale' in params
        assert params['scale'] > 0
        
    def test_cauchy_properties(self):
        """Test Cauchy distribution properties."""
        np.random.seed(42)
        data = np.random.standard_cauchy(size=1000)
        
        dist = Cauchy(data)
        dist.fit()
        
        assert dist.location is not None
        assert dist.scale_param > 0
        assert dist.median is not None
        assert dist.iqr > 0
        
    def test_cauchy_undefined_mean_variance(self):
        """Test that mean and variance are undefined (NaN)."""
        np.random.seed(42)
        data = np.random.standard_cauchy(size=1000)
        
        dist = Cauchy(data)
        dist.fit()
        
        assert np.isnan(dist.mean)
        assert np.isnan(dist.variance)
        
    def test_cauchy_median_is_location(self):
        """Test that median equals location parameter."""
        np.random.seed(42)
        data = np.random.standard_cauchy(size=1000)
        
        dist = Cauchy(data)
        dist.fit()
        
        assert np.isclose(dist.median, dist.location, rtol=0.01)
        
    def test_cauchy_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1.0, 2.0]
        
        with pytest.raises(InsufficientDataError):
            Cauchy(data)
            
    def test_cauchy_pdf_symmetric(self):
        """Test that PDF is symmetric around location."""
        np.random.seed(42)
        data = np.random.standard_cauchy(size=1000)
        
        dist = Cauchy(data)
        dist.fit()
        
        loc = dist.location
        offset = 2.0
        
        # PDF should be symmetric
        pdf_left = dist.pdf(loc - offset)
        pdf_right = dist.pdf(loc + offset)
        
        assert np.isclose(pdf_left, pdf_right, rtol=0.01)
        
    def test_cauchy_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.standard_cauchy(size=1000)
        
        dist = Cauchy(data)
        dist.fit()
        
        ks_stat, p_value = dist.test_goodness_of_fit(method='ks')
        
        assert ks_stat >= 0
        assert 0 <= p_value <= 1

