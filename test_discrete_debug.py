"""Test script for discrete distributions."""

import sys
sys.path.insert(0, '/Users/wilmarsepulvedaherrera/Desktop/personal_projects/pdist/src')

import numpy as np
from bestdist import DistributionFitter
from bestdist.distributions.discrete import Poisson

np.random.seed(42)

print("=" * 70)
print("TEST 1: Poisson Distribution Direct")
print("=" * 70)

# Generate Poisson data
data = np.random.poisson(lam=5, size=1000)
print(f"Data sample (first 20): {data[:20]}")
print(f"Data type: {data.dtype}")
print(f"Data min: {data.min()}, max: {data.max()}, mean: {data.mean():.2f}")
print()

# Fit directly
try:
    poisson = Poisson(data)
    params = poisson.fit()
    print(f"✅ Poisson fitted successfully!")
    print(f"   Lambda (mu): {params['mu']:.4f}")
    print(f"   Mean: {poisson.mean:.4f}")
    print()
    
    # Test PMF
    pmf_vals = poisson.pmf([0, 1, 2, 3, 4, 5])
    print(f"   PMF values: {pmf_vals}")
    print()
    
    # Test goodness of fit
    chi2_stat, p_value = poisson.test_goodness_of_fit(method='chi2')
    print(f"   Chi2 statistic: {chi2_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print()
except Exception as e:
    print(f"❌ Error fitting Poisson: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("TEST 2: DistributionFitter with Discrete")
print("=" * 70)

try:
    fitter = DistributionFitter(data, dist_type='discrete')
    print(f"Distributions to fit: {[d.__name__ for d in fitter.distributions]}")
    print(f"Method: {fitter.method}")
    print()
    
    results = fitter.fit(verbose=True, suppress_warnings=False)
    print(f"\n✅ Fitted {len(results)} distributions")
    print()
    
    if results:
        print("Results:")
        for r in results:
            print(f"  - {r['distribution']}: p-value={r['p_value']:.4f}, AIC={r['aic']:.2f}")
        print()
        
        best = fitter.get_best_distribution()
        if best:
            print(f"Best distribution: {best['distribution']}")
            print(f"Parameters: {best['parameters']}")
        else:
            print("❌ No best distribution found")
    else:
        print("❌ No results!")
        
except Exception as e:
    print(f"❌ Error with DistributionFitter: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST 3: Summary")
print("=" * 70)

try:
    summary = fitter.summary()
    print(summary)
except Exception as e:
    print(f"❌ Error getting summary: {e}")
    import traceback
    traceback.print_exc()

