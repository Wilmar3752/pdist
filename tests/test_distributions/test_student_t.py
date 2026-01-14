"""Tests for Student-t distribution."""

import numpy as np
import pytest

from bestdist.distributions.continuous import StudentT
from bestdist.utils.exceptions import InsufficientDataError


class TestStudentT:
    """Test cases for Student-t distribution."""
    
    def test_fit_student_t(self):
        """Test fitting Student-t distribution to data."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)
        
        dist = StudentT(data)
        params = dist.fit()
        
        assert 'df' in params
        assert 'loc' in params
        assert 'scale' in params
        assert params['df'] > 0
        assert params['scale'] > 0
        
    def test_student_t_properties(self):
        """Test Student-t distribution properties."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)
        
        dist = StudentT(data)
        dist.fit()
        
        assert dist.df > 0
        assert dist.location is not None
        assert dist.scale_param > 0
        assert not np.isnan(dist.mean)  # df > 1, mean should exist
        assert not np.isnan(dist.variance)  # df > 2, variance should exist
        
    def test_student_t_undefined_mean(self):
        """Test that mean is undefined when df <= 1."""
        # Create data that would result in df close to 1
        np.random.seed(42)
        # Use heavy-tailed data
        data = np.random.standard_cauchy(size=1000)
        
        dist = StudentT(data)
        # Manually set df to test the property
        dist.fit()
        if dist.df <= 1:
            assert np.isnan(dist.mean)
        
    def test_student_t_with_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = [1.0, 2.0]
        
        with pytest.raises(InsufficientDataError):
            StudentT(data)
            
    def test_student_t_median_is_location(self):
        """Test that median equals location parameter."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)
        
        dist = StudentT(data)
        dist.fit()
        
        assert np.isclose(dist.median, dist.location, rtol=0.01)
        
    def test_student_t_approaches_normal(self):
        """Test that high df Student-t approaches normal distribution."""
        np.random.seed(42)
        # Generate data from normal distribution
        data = np.random.normal(loc=0, scale=1, size=1000)
        
        dist = StudentT(data)
        dist.fit()
        
        # With normal data, fitted df should be large
        assert dist.df > 10  # Should be much larger than 10
        
    def test_student_t_pdf_symmetric(self):
        """Test that PDF is symmetric around location."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)
        
        dist = StudentT(data)
        dist.fit()
        
        loc = dist.location
        offset = 1.0
        
        # PDF should be symmetric
        pdf_left = dist.pdf(loc - offset)
        pdf_right = dist.pdf(loc + offset)
        
        assert np.isclose(pdf_left, pdf_right, rtol=0.01)
        
    def test_student_t_goodness_of_fit(self):
        """Test goodness of fit test."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)
        
        dist = StudentT(data)
        dist.fit()
        
        ks_stat, p_value = dist.test_goodness_of_fit(method='ks')
        
        assert ks_stat >= 0
        assert 0 <= p_value <= 1

