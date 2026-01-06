# bestdist v0.1.0 - Initial Release

🎉 First release of **bestdist** - Find the best probability distribution for your data!

## ✨ Highlights

- 🎯 Automatic fitting of Normal, Gamma, Beta, and Weibull distributions
- 📈 Statistical tests (KS, Anderson-Darling, Chi-square)
- 📊 Model selection (AIC, BIC)
- 🎨 Beautiful visualizations (PDF overlay, Q-Q plots)
- 🐼 Pandas integration
- ✅ Type hints and comprehensive tests
- 🚫 Automatic warning suppression for cleaner output

## 🚀 Quick Start

```bash
pip install bestdist
```

```python
from bestdist import DistributionFitter
import numpy as np

data = np.random.gamma(2, 2, 1000)
fitter = DistributionFitter(data)
results = fitter.fit()

best = fitter.get_best_distribution()
print(f"Best fit: {best['distribution']}")

fitter.plot_best_fit()
```

## 📚 Documentation

- [README](https://github.com/Wilmar3752/pdist#readme)
- [Quick Start Guide](https://github.com/Wilmar3752/pdist/blob/main/QUICKSTART.md)
- [Full Changelog](https://github.com/Wilmar3752/pdist/blob/main/CHANGELOG.md)

## 🙏 Thank You

Thank you for trying bestdist! If you find it useful, please ⭐ star the repo and share with colleagues.

Report bugs or suggest features on our [issue tracker](https://github.com/Wilmar3752/pdist/issues).

Happy fitting! 📊✨

