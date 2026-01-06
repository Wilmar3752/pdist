# Quick Start Guide for pdist

## Installation & Setup

### Option 1: Development Installation (Recommended for Contributing)

```bash
# Clone the repository
git clone <your-repo-url>
cd pdist

# Create a virtual environment (highly recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import pdist; print(pdist.__version__)"
```

### Option 2: User Installation (When Published)

```bash
pip install pdist
```

### Option 3: Quick Test Without Installation

```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/macOS
# or
set PYTHONPATH=%PYTHONPATH%;%cd%\src  # Windows

python
>>> import sys
>>> sys.path.insert(0, 'src')
>>> from pdist import DistributionFitter
>>> import numpy as np
>>> data = np.random.normal(5, 2, 100)
>>> fitter = DistributionFitter(data)
>>> results = fitter.fit(verbose=False)
>>> best = fitter.get_best_distribution()
>>> print(f"Best: {best['distribution']}")
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=pdist --cov-report=html

# Run specific test file
pytest tests/test_distributions/test_normal.py

# Run specific test
pytest tests/test_distributions/test_normal.py::TestNormal::test_fit

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_fit"
```

## Code Quality Checks

```bash
# Format code (automatically fixes issues)
black src tests examples

# Sort imports (automatically fixes issues)
isort src tests examples

# Lint code (reports issues)
flake8 src tests --max-line-length=88 --exclude=__pycache__

# Type checking (reports type issues)
mypy src --ignore-missing-imports
```

## Quick Usage Examples

### Example 1: Find Best Distribution

```python
from pdist import DistributionFitter
import numpy as np

# Your data
data = np.random.gamma(2, 2, 1000)

# Fit all distributions
fitter = DistributionFitter(data)
results = fitter.fit()

# Get best distribution
best = fitter.get_best_distribution()
print(f"Best: {best['distribution']} (p={best['p_value']:.4f})")

# View summary
print(fitter.summary())

# Plot
fitter.plot_best_fit()
```

### Example 2: Use Specific Distribution

```python
from pdist.distributions.continuous import Normal
import numpy as np

# Generate data
data = np.random.normal(5, 2, 1000)

# Fit distribution
dist = Normal(data)
params = dist.fit()

print(f"Mean: {dist.mean:.2f}")
print(f"Std: {dist.std:.2f}")

# Test fit
ks_stat, p_value = dist.test_goodness_of_fit()
print(f"KS test p-value: {p_value:.4f}")
```

### Example 3: Pandas Integration

```python
import pandas as pd
from pdist import DistributionFitter

# Load data
df = pd.read_csv('data.csv')

# Fit distribution to column
fitter = DistributionFitter(df['column_name'])
best = fitter.get_best_distribution()

print(f"Best fit: {best['distribution']}")
```

## Project Structure Overview

```
pdist/
├── src/pdist/              # Source code
│   ├── core/              # Core functionality
│   ├── distributions/     # Distribution implementations
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── examples/              # Usage examples
├── pyproject.toml         # Package configuration
└── README.md              # Documentation
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'pdist'`

**Solutions**:
1. Install the package: `pip install -e .`
2. Or add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
3. Or use in Python: `sys.path.insert(0, 'src')`

### Test Failures

**Problem**: Tests fail with import errors

**Solutions**:
1. Install with dev dependencies: `pip install -e ".[dev]"`
2. Check Python version: `python --version` (requires 3.8+)

### Permission Issues (conda)

**Problem**: Permission denied when installing in conda environment

**Solutions**:
1. Use virtual environment instead of conda
2. Or install with `--user` flag: `pip install --user -e .`
3. Or create new conda environment: `conda create -n pdist python=3.11`

### scipy/numpy Issues

**Problem**: Segmentation fault or numpy errors

**Solutions**:
1. Update numpy/scipy: `pip install --upgrade numpy scipy`
2. Reinstall in fresh environment
3. Check Python and library compatibility

## Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes**
   - Edit code in `src/pdist/`
   - Add tests in `tests/`
   - Update documentation

3. **Test changes**
   ```bash
   pytest
   black src tests
   isort src tests
   flake8 src tests
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add my feature"
   git push origin feature/my-feature
   ```

5. **Create pull request**

## Next Steps

1. ✅ Package structure created
2. ✅ Core functionality implemented
3. ✅ Tests written
4. ⬜ Run tests: `pytest`
5. ⬜ Try examples: `python examples/basic_usage.py`
6. ⬜ Add more distributions
7. ⬜ Publish to PyPI

## Getting Help

- Read the full documentation: `README.md`
- Check architecture guide: `ARCHITECTURE.md`
- Look at examples: `examples/`
- Run example script: `python examples/basic_usage.py`

## Quick Reference

```bash
# Setup
pip install -e ".[dev]"

# Test
pytest

# Format
black src tests && isort src tests

# Lint
flake8 src tests --max-line-length=88

# Type check
mypy src --ignore-missing-imports

# Run example
python examples/basic_usage.py
```

---

**Need more help?** Check `ARCHITECTURE.md` for detailed architecture information.

