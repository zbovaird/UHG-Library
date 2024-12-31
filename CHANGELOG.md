# Changelog

All notable changes to this project will be documented in this file.

## [0.2.4] - 2024-03-19

### Fixed
- Corrected quadrance calculation to match UHG.pdf definition
- Fixed spread computation for better numerical stability
- Updated cross law implementation to handle all cases correctly
- Revised cross ratio calculation to use direct hyperbolic joins
- Improved handling of special cases in geometric operations
- Enhanced numerical stability across all core operations

## [0.2.0] - 2023-12-07

### Added
- UHG Metric Learning
  - Pure projective implementation
  - Cross-ratio preservation
  - Geometric pattern matching

- UHG Attention Mechanism
  - Multi-head attention in projective space
  - Cross-ratio preserving attention weights
  - Geometric relationship learning

- Threat Correlation
  - Indicator correlation using UHG
  - Cross-type pattern recognition
  - Network-System-Behavior relationships
  - Payload analysis integration

### Changed
- Updated core UHG operations for better numerical stability
- Improved projective transformation handling
- Enhanced cross-ratio computation

### Fixed
- Shape handling in projective operations
- Homogeneous coordinate normalization
- Cross-ratio preservation in transformations

## [0.1.0] - 2023-12-01

### Added
- Initial release
- Core UHG operations
- Basic feature extraction
- Anomaly detection

## [0.3.0] - 2024-03-19

### Added
- New UHG-compliant functional operations in `uhg.nn.functional`
- Specialized loss functions in `uhg.nn.losses` including UHGLoss and UHGAnomalyLoss
- Comprehensive metrics module in `uhg.utils.metrics` for geometric evaluations
- Detailed IMPLEMENTATION.md guide with best practices and examples
- Enhanced cross-ratio preservation checks and stability improvements
- Better numerical stability in geometric operations
- Improved error handling and validation

### Changed
- Optimized UHG operations for better performance
- Enhanced documentation with practical examples
- Improved type hints and error messages