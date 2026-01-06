"""
Simple test to verify the package structure works correctly.
Run this without installing to check imports.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("Testing pdist package structure...")
print("=" * 60)

# Test imports
try:
    from pdist import DistributionFitter
    print("✓ Successfully imported DistributionFitter")
    
    from pdist import Normal, Gamma, Beta, Weibull
    print("✓ Successfully imported all distributions")
    
    from pdist.core.base import BaseDistribution
    print("✓ Successfully imported BaseDistribution")
    
    from pdist.utils.exceptions import PdistException
    print("✓ Successfully imported exceptions")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test basic functionality
print("\n" + "=" * 60)
print("Testing basic functionality...")
print("=" * 60)

try:
    import numpy as np
    
    # Generate test data
    np.random.seed(42)
    data = np.random.normal(5, 2, 100)
    print(f"✓ Generated test data: {len(data)} points")
    
    # Test individual distribution
    dist = Normal(data)
    print(f"✓ Created Normal distribution object")
    
    params = dist.fit()
    print(f"✓ Fitted distribution: {params}")
    
    stat, pval = dist.test_goodness_of_fit()
    print(f"✓ Goodness-of-fit test: KS={stat:.4f}, p-value={pval:.4f}")
    
    # Test fitter
    fitter = DistributionFitter(data)
    print(f"✓ Created DistributionFitter")
    
    results = fitter.fit(verbose=False)
    print(f"✓ Fitted {len(results)} distributions")
    
    best = fitter.get_best_distribution()
    print(f"✓ Best distribution: {best['distribution']} (p={best['p_value']:.4f})")
    
    # Test summary
    summary = fitter.summary()
    print(f"✓ Generated summary DataFrame with {len(summary)} rows")
    
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
print("\nPackage structure is correct and ready to use!")
print("\nNext steps:")
print("1. Install in development mode: pip install -e .")
print("2. Run full test suite: pytest")
print("3. Try the examples: python examples/basic_usage.py")

