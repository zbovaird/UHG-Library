# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.18] - 2024-03-19

### Added
- PEP 561 compliance with py.typed marker
- Improved type hints throughout codebase
- UHG feature extraction module with projective geometry operations
- Better numerical stability in geometric computations
- Cross-ratio preservation in feature space

### Changed
- Switched to setuptools_scm for version management
- Updated package metadata and dependencies
- Improved documentation and examples

### Fixed
- Version synchronization between setup.py and __init__.py
- PyPI packaging issues with missing files
- Numerical stability in cross-ratio computations

## [0.1.17] - 2024-03-18

### Added
- Initial release of UHG library
- Core projective geometry operations
- Basic hyperbolic neural network layers
- Graph neural network support
- Documentation and examples 