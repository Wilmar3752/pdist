# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-14

### Added
- **5 new continuous distributions**:
  - Lognormal
  - Exponential
  - Uniform
  - Cauchy
  - Student-t
- **4 discrete distributions** support:
  - Poisson
  - Binomial
  - Negative Binomial
  - Geometric
- `dist_type` parameter in `DistributionFitter` to select between 'continuous' (default) and 'discrete'
- Chi-square goodness-of-fit test for discrete distributions
- PMF (Probability Mass Function) plotting for discrete distributions
- Comprehensive test suite with 80 tests (81% coverage)

### Changed
- Improved distribution fitting accuracy
- Better handling of edge cases in goodness-of-fit tests
- Updated README with comprehensive examples for both continuous and discrete distributions
- Cleaned up project structure (removed redundant docs/ and examples/)

### Fixed
- Fixed Chi-square test for discrete distributions with proper count normalization
- Fixed AIC/BIC calculations for discrete distributions
- Fixed plotting methods to correctly handle PMF vs PDF

### Internal
- Created `BaseDiscreteDistribution` abstract class for discrete distributions
- Refactored `DistributionFitter` to support both continuous and discrete types
- Improved test fixtures for better distribution identification

## [0.1.0] - 2026-01-05

### Added
- Initial release with 4 continuous distributions:
  - Normal
  - Gamma
  - Beta
  - Weibull
- Core `DistributionFitter` class with automatic distribution selection
- Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit tests
- AIC and BIC information criteria
- Q-Q plots and visualization tools
- Comprehensive error handling
- Full type hinting support

