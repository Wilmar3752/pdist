# New Distributions Guide

This guide provides detailed information about the 5 new continuous distributions added to `bestdist`.

## 📋 Table of Contents

1. [Lognormal Distribution](#lognormal-distribution)
2. [Exponential Distribution](#exponential-distribution)
3. [Uniform Distribution](#uniform-distribution)
4. [Cauchy Distribution](#cauchy-distribution)
5. [Student-t Distribution](#student-t-distribution)
6. [Quick Reference](#quick-reference)

---

## 1. Lognormal Distribution

### When to Use
- **Income and wealth data**: Salaries, asset prices, household wealth
- **Biological measurements**: Cell sizes, gene expression levels
- **Internet/network data**: File sizes, web traffic
- **Financial data**: Stock prices, returns (positive only)

### Characteristics
- **Support**: x > 0 (positive values only)
- **Shape**: Right-skewed (long tail to the right)
- **Key property**: If X ~ Lognormal, then ln(X) ~ Normal

### Parameters
- `s` (sigma): Shape parameter (standard deviation of log-transformed data)
- `loc`: Location parameter (shift)
- `scale`: Scale parameter (median when loc=0)

### Example Usage

```python
from bestdist.distributions.continuous import Lognormal
import numpy as np

# Example: Annual salaries in thousands of dollars
salaries = np.random.lognormal(mean=10.5, sigma=0.8, size=1000)

# Fit distribution
dist = Lognormal(salaries)
params = dist.fit()

# Get statistics
print(f"Mean salary: ${dist.mean:,.2f}K")
print(f"Median salary: ${dist.median:,.2f}K")
print(f"Sigma (dispersion): {dist.sigma:.4f}")

# Calculate percentiles
p90 = dist.ppf(0.90)
print(f"90th percentile: ${p90:,.2f}K")

# Probability that salary exceeds $100K
prob = 1 - dist.cdf(100)
print(f"P(salary > $100K) = {prob:.2%}")
```

### Interpretation
- **High sigma** → More dispersed, greater inequality
- **Low sigma** → More concentrated around median
- **Mean > Median** → Always true for lognormal

---

## 2. Exponential Distribution

### When to Use
- **Time between events**: Customer arrivals, equipment failures
- **Lifetime data**: Product lifetimes, survival times
- **Queueing theory**: Service times, waiting times
- **Radioactive decay**: Time until decay

### Characteristics
- **Support**: x ≥ 0
- **Shape**: Decreasing (highest probability at x=0)
- **Memoryless property**: P(X > s+t | X > s) = P(X > t)

### Parameters
- `loc`: Location parameter (minimum value)
- `scale`: Scale parameter (mean = scale)
- `rate` (λ): Rate parameter = 1/scale

### Example Usage

```python
from bestdist.distributions.continuous import Exponential
import numpy as np

# Example: Time between customer arrivals (minutes)
wait_times = np.random.exponential(scale=5.0, size=1000)

# Fit distribution
dist = Exponential(wait_times)
dist.fit()

# Get statistics
print(f"Average wait time: {dist.mean:.2f} minutes")
print(f"Arrival rate (λ): {dist.rate:.4f} customers/minute")
print(f"Median wait time: {dist.median:.2f} minutes")

# Probability of waiting less than 10 minutes
prob = dist.cdf(10)
print(f"P(wait < 10 min) = {prob:.2%}")

# Time by which 95% of customers will arrive
t95 = dist.ppf(0.95)
print(f"95% arrive within {t95:.2f} minutes")
```

### Interpretation
- **High rate (λ)** → Events happen frequently, short waits
- **Low rate (λ)** → Events are rare, long waits
- **Memoryless** → Past doesn't affect future waiting time

---

## 3. Uniform Distribution

### When to Use
- **Random number generation**: Truly random processes
- **Lack of information**: When all outcomes equally likely
- **Manufacturing tolerances**: When values fall within a range
- **Baseline/null model**: For comparison with other distributions

### Characteristics
- **Support**: a ≤ x ≤ b
- **Shape**: Flat (constant probability)
- **All outcomes equally likely** within the range

### Parameters
- `loc`: Lower bound (a)
- `scale`: Range width (b - a)

### Example Usage

```python
from bestdist.distributions.continuous import Uniform
import numpy as np

# Example: Random sensor measurements
measurements = np.random.uniform(low=20, high=80, size=1000)

# Fit distribution
dist = Uniform(measurements)
dist.fit()

# Get statistics
print(f"Lower bound: {dist.lower_bound:.2f}")
print(f"Upper bound: {dist.upper_bound:.2f}")
print(f"Mean: {dist.mean:.2f}")
print(f"Variance: {dist.variance:.2f}")

# Probability in specific range
prob = dist.cdf(50) - dist.cdf(30)
print(f"P(30 < X < 50) = {prob:.2%}")
```

### Interpretation
- **Narrow range** → Low variance, precise measurements
- **Wide range** → High variance, imprecise measurements
- **Mean = Median = Mode** → Always at the center

---

## 4. Cauchy Distribution

### When to Use
- **Heavy-tailed data**: Data with extreme outliers
- **Robust statistics**: When standard methods fail
- **Physics**: Resonance phenomena, Lorentzian profiles
- **Finance**: Modeling crashes and extreme events

### Characteristics
- **Support**: -∞ < x < ∞
- **Shape**: Bell-shaped but with much heavier tails than Normal
- **Mean and variance are undefined** (do not exist mathematically)

### Parameters
- `loc`: Location parameter (median and mode)
- `scale`: Scale parameter (half-width at half-maximum)

### Example Usage

```python
from bestdist.distributions.continuous import Cauchy
import numpy as np

# Example: Physics measurements with outliers
measurements = np.random.standard_cauchy(size=1000)
# Clip extremes for practical analysis
measurements = measurements[(measurements > -10) & (measurements < 10)]

# Fit distribution
dist = Cauchy(measurements)
dist.fit()

# Get statistics
print(f"Location (median): {dist.location:.4f}")
print(f"Scale: {dist.scale_param:.4f}")
print(f"IQR: {dist.iqr:.4f}")

# Note: mean and variance are undefined
print(f"Mean: {dist.mean} (undefined)")
print(f"Variance: {dist.variance} (undefined)")

# Use quantiles instead
q25 = dist.ppf(0.25)
q75 = dist.ppf(0.75)
print(f"25th percentile: {q25:.4f}")
print(f"75th percentile: {q75:.4f}")
```

### Interpretation
- **Location** → Center of distribution (use instead of mean)
- **Scale** → Spread of distribution (use IQR instead of variance)
- **WARNING**: Do not use mean/variance for Cauchy!

---

## 5. Student-t Distribution

### When to Use
- **Small sample sizes**: When n < 30
- **Unknown population variance**: Estimating with sample variance
- **Robust estimation**: Resistant to outliers
- **Financial returns**: Stock returns with outliers

### Characteristics
- **Support**: -∞ < x < ∞
- **Shape**: Bell-shaped, heavier tails than Normal
- **Converges to Normal** as df → ∞

### Parameters
- `df`: Degrees of freedom (controls tail heaviness)
- `loc`: Location parameter (mean when df > 1)
- `scale`: Scale parameter

### Example Usage

```python
from bestdist.distributions.continuous import StudentT
import numpy as np

# Example: Stock returns with occasional extreme events
returns = np.random.standard_t(df=5, size=1000)

# Fit distribution
dist = StudentT(returns)
dist.fit()

# Get statistics
print(f"Degrees of freedom: {dist.df:.2f}")
print(f"Location: {dist.location:.4f}")
print(f"Mean: {dist.mean:.4f}")
print(f"Variance: {dist.variance:.4f}")

# Check tail behavior
if dist.df > 30:
    print("Close to Normal distribution")
elif dist.df > 10:
    print("Slightly heavier tails than Normal")
else:
    print("Much heavier tails than Normal - robust to outliers")

# Calculate risk metrics
var_95 = dist.ppf(0.05)  # 5% Value-at-Risk
print(f"95% VaR: {var_95:.4f}")
```

### Interpretation
- **df ≤ 2** → Very heavy tails, extreme outliers
- **2 < df < 30** → Heavier tails than Normal
- **df > 30** → Close to Normal distribution
- **Low df** → More robust to outliers

---

## 6. Quick Reference

### Distribution Selection Guide

| Data Characteristics | Recommended Distribution |
|---------------------|-------------------------|
| Positive, right-skewed (income, prices) | **Lognormal** |
| Time between events, memoryless | **Exponential** |
| Bounded range, all equally likely | **Uniform** |
| Extreme outliers, undefined mean | **Cauchy** |
| Small sample, unknown variance | **Student-t** |
| Symmetric, bell-shaped | Normal |
| Positive, flexible shape | Gamma, Weibull |

### Parameter Interpretation

| Distribution | Key Parameters | Interpretation |
|-------------|----------------|----------------|
| **Lognormal** | sigma (s) | Higher → more dispersion/inequality |
| **Exponential** | rate (λ) | Higher → more frequent events |
| **Uniform** | scale | Range width |
| **Cauchy** | scale | Spread (use IQR) |
| **Student-t** | df | Lower → heavier tails |

### Statistical Properties

| Distribution | Mean | Variance | Skewness |
|-------------|------|----------|----------|
| **Lognormal** | exp(μ + σ²/2) | Defined | Positive (right) |
| **Exponential** | 1/λ | 1/λ² | Positive |
| **Uniform** | (a+b)/2 | (b-a)²/12 | 0 |
| **Cauchy** | Undefined | Undefined | Undefined |
| **Student-t** | 0 (if df>1) | df/(df-2) (if df>2) | 0 |

### Code Examples

```python
from bestdist import DistributionFitter
from bestdist.distributions.continuous import (
    Lognormal, Exponential, Uniform, Cauchy, StudentT
)
import numpy as np

# Compare all distributions
data = np.random.lognormal(mean=2, sigma=0.5, size=1000)

fitter = DistributionFitter(
    data,
    distributions=[
        'normal', 'gamma', 'lognormal', 'exponential',
        'uniform', 'cauchy', 'studentt'
    ]
)

results = fitter.fit()
best = fitter.get_best_distribution()

print(f"Best fit: {best['distribution']}")
print(f"AIC: {best['aic']:.2f}")
print(f"BIC: {best['bic']:.2f}")

# View summary
summary = fitter.summary()
print(summary)
```

---

## Tips and Best Practices

1. **Start with visualization**: Plot your data histogram first
2. **Check data characteristics**: Positive? Bounded? Symmetric? Outliers?
3. **Use DistributionFitter**: Compare multiple distributions automatically
4. **Validate assumptions**: Check Q-Q plots and goodness-of-fit tests
5. **Consider domain knowledge**: Some distributions make more sense theoretically
6. **Be cautious with Cauchy**: Only use when mean/variance truly don't exist
7. **Student-t for robustness**: Good default when data has outliers

## Further Reading

- [Statistical Distributions in Python](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Choosing the Right Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366.htm)
- [Distribution Fitting Best Practices](https://www.researchgate.net/publication/267794804_Distribution_Fitting)

