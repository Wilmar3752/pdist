# bestdist 📊

**Find the best probability distribution for your data**

`bestdist` is a Python package that helps you identify which probability distribution best fits your data using statistical tests and information criteria.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **Automatic Distribution Fitting**: Test multiple distributions at once
- 📈 **Statistical Tests**: Kolmogorov-Smirnov, Anderson-Darling, Chi-square
- 📊 **Information Criteria**: AIC and BIC for model selection
- 🎨 **Visualization**: Built-in plotting for fit assessment
- 🔧 **Extensible**: Easy to add custom distributions
- 🐼 **Pandas Integration**: Works seamlessly with pandas DataFrames
- ✅ **Type Hints**: Full type annotation support
- 🧪 **Well Tested**: Comprehensive test suite

## Installation

### From PyPI (when published)
```bash
pip install bestdist
```

### From source
```bash
git clone https://github.com/Wilmar3752/pdist.git
cd pdist
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Quick Start

### Continuous Data
```python
from bestdist import DistributionFitter
import numpy as np

# Your continuous data (e.g., measurements, prices)
data = np.random.gamma(2, 2, 1000)

# Create fitter (continuous is default)
fitter = DistributionFitter(data)
results = fitter.fit()

# Get best distribution
best = fitter.get_best_distribution()
print(f"Best fit: {best['distribution']}")
print(f"Parameters: {best['parameters']}")
print(f"P-value: {best['p_value']:.4f}")

# View summary of all fits
print(fitter.summary())

# Visualize the best fit
fitter.plot_best_fit()

# Compare all distributions
fitter.compare_distributions()
```

### Discrete Data
```python
from bestdist import DistributionFitter
import numpy as np

# Your discrete data (e.g., count data)
data = np.random.poisson(lam=3.5, size=1000)

# Create fitter for discrete distributions
fitter = DistributionFitter(data, dist_type='discrete')
results = fitter.fit()

# Get best distribution
best = fitter.get_best_distribution()
print(f"Best fit: {best['distribution']}")
print(f"Lambda: {best['parameters']}")
```

## Supported Distributions

### Continuous Distributions (9 distributions)
- **Normal** (Gaussian): Symmetric, bell-shaped distribution
- **Gamma**: Skewed distribution for positive values
- **Beta**: Bounded [0, 1], flexible shapes
- **Weibull**: Common in reliability engineering and lifetime analysis
- **Lognormal**: Right-skewed, for positive data (income, prices)
- **Exponential**: Memoryless distribution for waiting times
- **Uniform**: Equal probability across a range
- **Cauchy**: Heavy-tailed distribution (undefined mean/variance)
- **Student-t**: Robust to outliers, heavier tails than Normal

### Discrete Distributions (4 distributions)
- **Poisson**: Count data, number of events in fixed interval
- **Binomial**: Number of successes in fixed trials
- **Negative Binomial**: Overdispersed count data, failures before successes
- **Geometric**: Number of trials until first success

### Coming Soon
- Chi-square
- F-distribution
- Pareto

## Advanced Usage

### Custom Distribution List

```python
from bestdist import DistributionFitter
from bestdist.distributions.continuous import (
    Normal, Gamma, Lognormal, Exponential
)
from bestdist.distributions.discrete import (
    Poisson, Binomial, NegativeBinomial
)

# Continuous: only fit specific distributions
fitter = DistributionFitter(
    continuous_data,
    distributions=[Normal, Gamma, Lognormal, Exponential]
)
results = fitter.fit()

# Discrete: only fit specific distributions
fitter = DistributionFitter(
    count_data,
    dist_type='discrete',
    distributions=[Poisson, NegativeBinomial]
)
results = fitter.fit()
```

### Selection Criteria

```python
# Select best by different criteria
best_pvalue = fitter.get_best_distribution(criterion='p_value')
best_aic = fitter.get_best_distribution(criterion='aic')
best_bic = fitter.get_best_distribution(criterion='bic')
```

### Individual Distribution Usage

```python
# CONTINUOUS DISTRIBUTIONS
from bestdist.distributions.continuous import Normal, Lognormal, Exponential
import numpy as np

# Example 1: Normal distribution
data = np.random.normal(5, 2, 1000)
dist = Normal(data)
params = dist.fit()
print(f"Mean: {dist.mean:.2f}, Std: {dist.std:.2f}")

# Example 2: Lognormal (income data)
income_data = np.random.lognormal(mean=10.5, sigma=0.8, size=1000)
lognormal = Lognormal(income_data)
lognormal.fit()
print(f"Mean income: ${lognormal.mean:,.2f}")
print(f"Median income: ${lognormal.median:,.2f}")

# DISCRETE DISTRIBUTIONS
from bestdist.distributions.discrete import Poisson, Binomial

# Example 3: Poisson (count data)
count_data = np.random.poisson(lam=3.5, size=1000)
poisson = Poisson(count_data)
poisson.fit()
print(f"Lambda (rate): {poisson.mu:.4f}")
print(f"P(X=5) = {poisson.pmf(5):.4f}")

# Example 4: Binomial (success/failure)
trials_data = np.random.binomial(n=10, p=0.3, size=1000)
binomial = Binomial(trials_data)
binomial.fit()
print(f"n (trials): {binomial.n}, p (success): {binomial.p:.4f}")

# Generate samples, evaluate PDF/CDF/PMF
samples = dist.rvs(size=100, random_state=42)
x = np.linspace(0, 10, 100)
pdf_values = dist.pdf(x)  # For continuous
cdf_values = dist.cdf(x)
```

### Working with Pandas

```python
import pandas as pd
from bestdist import DistributionFitter

# Load data
df = pd.read_csv('data.csv')

# Fit distribution to a column
fitter = DistributionFitter(df['column_name'])
best = fitter.get_best_distribution()

# Get summary as DataFrame
summary_df = fitter.summary()
print(summary_df)
```

### Custom Distributions

```python
from bestdist.core.base import BaseDistribution
from scipy.stats import expon, rv_continuous
from typing import Tuple

class Exponential(BaseDistribution):
    """Custom exponential distribution."""
    
    def _get_scipy_dist(self) -> rv_continuous:
        return expon
    
    def _extract_params(self, fit_result: Tuple) -> dict:
        return {
            'loc': float(fit_result[0]),
            'scale': float(fit_result[1])
        }

# Use your custom distribution
fitter = DistributionFitter(data, distributions=[Exponential])
results = fitter.fit()
```

## API Reference

### DistributionFitter

Main class for fitting multiple distributions.

**Parameters:**
- `data`: Array-like data to fit
- `distributions`: List of distribution classes (default: all available)
- `method`: Goodness-of-fit test method ('ks', 'ad', 'chi2')

**Methods:**
- `fit(verbose=True)`: Fit all distributions
- `get_best_distribution(criterion='p_value')`: Get best fit
- `summary(top_n=None)`: Get summary DataFrame
- `plot_best_fit(bins=30)`: Plot best fit distribution
- `compare_distributions()`: Compare all fits

### BaseDistribution

Abstract base class for distributions.

**Methods:**
- `fit()`: Fit distribution to data
- `test_goodness_of_fit(method='ks')`: Perform GOF test
- `pdf(x)`: Probability density function
- `cdf(x)`: Cumulative distribution function
- `ppf(q)`: Percent point function (inverse CDF)
- `rvs(size, random_state)`: Generate random samples
- `get_info()`: Get distribution information

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pdist --cov-report=html

# Run specific test file
pytest tests/test_distributions/test_normal.py
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/pdist.git
cd pdist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint
flake8 src tests

# Type checking
mypy src
```

## Project Structure

```
pdist/
├── src/pdist/
│   ├── __init__.py
│   ├── core/
│   │   ├── base.py          # Abstract base class
│   │   └── fitter.py        # Main fitter
│   ├── distributions/
│   │   └── continuous/
│   │       ├── normal.py
│   │       ├── gamma.py
│   │       ├── beta.py
│   │       └── weibull.py
│   └── utils/
│       ├── exceptions.py
│       └── types.py
├── tests/
│   ├── test_distributions/
│   ├── test_core/
│   └── conftest.py
├── pyproject.toml
└── README.md
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{bestdist2024,
  author = {Sepulveda, Wilmar},
  title = {bestdist: Find the best probability distribution for your data},
  year = {2024},
  url = {https://github.com/Wilmar3752/pdist}
}
```

## Roadmap

- [ ] Add more distributions (lognormal, exponential, etc.)
- [ ] Support for discrete distributions
- [ ] Parallel fitting for large datasets
- [ ] GUI/Web interface
- [ ] Integration with scikit-learn
- [ ] Bayesian model selection
- [ ] Mixture distributions

## Acknowledgments

- Built with [scipy](https://scipy.org/) and [numpy](https://numpy.org/)
- Inspired by the need for easy distribution fitting in data science workflows

## Contact

- GitHub: [@Wilmar3752](https://github.com/Wilmar3752)
- Email: wilmar.sepulveda2@gmail.com

---

Made with ❤️
