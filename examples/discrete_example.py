"""
Example demonstrating discrete distributions in bestdist.

This script shows how to use discrete distributions for count data.
"""

import sys
sys.path.insert(0, '/Users/wilmarsepulvedaherrera/Desktop/personal_projects/pdist/src')

import numpy as np
from bestdist import DistributionFitter
from bestdist.distributions.discrete import Poisson, Binomial, NegativeBinomial, Geometric

np.random.seed(42)

print("=" * 80)
print("DISCRETE DISTRIBUTIONS EXAMPLE")
print("=" * 80)
print()

# ============================================================================
# Example 1: Poisson Distribution - Call Center Data
# ============================================================================
print("1. POISSON DISTRIBUTION - Number of calls per hour")
print("-" * 80)

# Generate call data (average 5 calls per hour)
calls_per_hour = np.random.poisson(lam=5, size=1000)

print(f"Sample data (first 20): {calls_per_hour[:20]}")
print(f"Mean: {calls_per_hour.mean():.2f}, Variance: {calls_per_hour.var():.2f}")
print()

# Fit with DistributionFitter
fitter = DistributionFitter(calls_per_hour, dist_type='discrete')
results = fitter.fit()

print("Fitted distributions:")
summary = fitter.summary()
print(summary.to_string())
print()

best = fitter.get_best_distribution()
if best:
    print(f"✅ Best fit: {best['distribution']}")
    print(f"   Parameters: {best['parameters']}")
    print(f"   P-value: {best['p_value']:.4f}")
    print()

# ============================================================================
# Example 2: Binomial Distribution - Quality Control
# ============================================================================
print("=" * 80)
print("2. BINOMIAL DISTRIBUTION - Defects in batches of 20 items")
print("-" * 80)

# Generate defect data (20 items per batch, 5% defect rate)
defects = np.random.binomial(n=20, p=0.05, size=1000)

print(f"Sample data (first 20): {defects[:20]}")
print(f"Mean: {defects.mean():.2f}, Expected: {20 * 0.05}")
print()

# Fit directly
binomial = Binomial(defects)
params = binomial.fit()

print(f"✅ Binomial fitted:")
print(f"   n (trials): {binomial.n}")
print(f"   p (defect rate): {binomial.p:.4f}")
print(f"   Mean: {binomial.mean:.2f}")
print(f"   Variance: {binomial.variance:.2f}")
print()

# ============================================================================
# Example 3: Compare All Discrete Distributions
# ============================================================================
print("=" * 80)
print("3. COMPARISON - Fit all discrete distributions to Poisson data")
print("-" * 80)

# Generate Poisson data
data = np.random.poisson(lam=3.5, size=1000)

# Fit all
fitter_all = DistributionFitter(data, dist_type='discrete')
results_all = fitter_all.fit()

print("\nAll distributions ranked by p-value:")
print(fitter_all.summary().to_string())
print()

print("=" * 80)
print("✅ EXAMPLES COMPLETE")
print("=" * 80)
print()
print("Key Takeaways:")
print("  • Use dist_type='discrete' for count data")
print("  • Poisson: For events in fixed interval (mean = variance)")
print("  • Binomial: For fixed number of trials")
print("  • Negative Binomial: For overdispersed counts (variance > mean)")
print("  • Geometric: For trials until first success")

