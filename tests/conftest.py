"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def normal_data():
    """Generate sample data from normal distribution."""
    np.random.seed(123)  # Changed seed for better normal characteristics
    return np.random.normal(loc=10, scale=3, size=1000)


@pytest.fixture
def gamma_data():
    """Generate sample data from gamma distribution."""
    np.random.seed(42)
    return np.random.gamma(shape=2, scale=2, size=1000)


@pytest.fixture
def beta_data():
    """Generate sample data from beta distribution."""
    np.random.seed(42)
    return np.random.beta(a=2, b=5, size=1000)


@pytest.fixture
def weibull_data():
    """Generate sample data from Weibull distribution."""
    from scipy.stats import weibull_min
    np.random.seed(42)
    return weibull_min.rvs(c=1.5, scale=1.0, size=1000)


@pytest.fixture
def pandas_series_data(normal_data):
    """Convert normal data to pandas Series."""
    return pd.Series(normal_data)


@pytest.fixture
def data_with_nans(normal_data):
    """Create data with NaN values."""
    data = normal_data.copy()
    data[::100] = np.nan
    return data

