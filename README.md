# Universal Hyperbolic Geometry (UHG) Library

A pure implementation of Universal Hyperbolic Geometry using projective operations, optimized for cybersecurity applications.

## Features

### Core Features
- Pure projective geometry implementation
- No differential geometry or manifold assumptions
- Cross-ratio preservation throughout
- Numerically stable operations

### Advanced Features
- UHG Metric Learning
- Multi-head Attention Mechanism
- Threat Correlation Engine
- Pattern Recognition
- Anomaly Detection

### Cybersecurity Applications
- Network Traffic Analysis
- System Event Correlation
- Behavioral Pattern Detection
- Threat Intelligence Integration
- Zero-Day Attack Detection

## Installation

Basic installation:
```bash
pip install uhg
```

With security features:
```bash
pip install uhg[security]
```

With visualization tools:
```bash
pip install uhg[viz]
```

## Quick Start

```python
from uhg import (
    ProjectiveUHG,
    UHGMultiHeadAttention,
    ThreatCorrelation
)

# Initialize threat correlation
correlation = ThreatCorrelation(
    feature_dim=8,
    num_heads=4
)

# Create indicators
indicators = [
    ThreatIndicator(
        ThreatIndicatorType.NETWORK,
        value="suspicious_pattern",
        confidence=0.9,
        context={
            "port": 443,
            "protocol": "TCP",
            "bytes_out": 1024
        }
    ),
    ThreatIndicator(
        ThreatIndicatorType.SYSTEM,
        value="malicious_process",
        confidence=0.85,
        context={
            "pid": 1234,
            "memory_usage": 50000,
            "api_calls": 15
        }
    )
]

# Analyze relationships
groups = correlation.get_correlation_groups(indicators)
relationships = correlation.analyze_indicator_relationships(indicators)
```

## Documentation

Full documentation is available at [uhg.readthedocs.io](https://uhg.readthedocs.io).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
