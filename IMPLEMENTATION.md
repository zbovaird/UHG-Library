# UHG Library Implementation Guide

## Core Operations

### UHG Inner Product
**What it does**: Computes the inner product between points in UHG space.

**When to use**: For measuring relationships between points while preserving hyperbolic structure.

**Example**:
```python
import torch
from uhg.utils.metrics import uhg_inner_product

p1 = torch.tensor([1.0, 0.0, 1.0])  # [x, y, h]
p2 = torch.tensor([0.0, 1.0, 1.0])  # [x, y, h]
prod = uhg_inner_product(p1, p2)
```

### UHG Normalization
**What it does**: Normalizes points while preserving UHG structure.

**When to use**: After operations that might break hyperbolic relationships.

**Example**:
```python
from uhg.nn.functional import uhg_normalize

points = torch.randn(10, 3)  # [batch_size, features + 1]
normalized = uhg_normalize(points)
```

## Graph Operations

### UHG-Compliant Scatter Mean
**What it does**: Aggregates neighbor features in a UHG-preserving way.

**When to use**: In graph neural networks for message passing.

**Example**:
```python
from uhg.nn.functional import scatter_mean_custom

source = torch.randn(5, 3)  # [num_edges, features]
index = torch.tensor([0, 0, 1, 1, 2])
out = scatter_mean_custom(source, index)
```

## Neural Network Components

### UHGSAGEConv Layer
**What it does**: GraphSAGE convolution that preserves UHG structure.

**When to use**: For building hyperbolic graph neural networks.

**Example**:
```python
from uhg.nn.layers.sage import UHGSAGEConv

conv = UHGSAGEConv(in_channels=5, out_channels=3)
x = torch.randn(10, 5)
edge_index = torch.tensor([[0, 1], [1, 2]])
out = conv(x, edge_index)
```

## Loss Functions

### UHG Loss
**What it does**: Loss function that maintains hyperbolic structure.

**When to use**: For general training in hyperbolic space.

**Example**:
```python
from uhg.nn.losses import UHGLoss

criterion = UHGLoss(spread_weight=0.1)
loss = criterion(z, edge_index, batch_size=32)
```

### UHG Anomaly Loss
**What it does**: Loss function specialized for anomaly detection.

**When to use**: When detecting anomalies in hyperbolic space.

**Example**:
```python
from uhg.nn.losses import UHGAnomalyLoss

criterion = UHGAnomalyLoss(margin=1.0)
loss = criterion(z, edge_index, batch_size=32)
```

## Best Practices

1. **Always normalize after operations**:
```python
x = uhg_normalize(x)  # After any transformation
```

2. **Use UHG-compliant activations**:
```python
x = uhg_relu(x)  # Instead of F.relu
```

3. **Monitor cross-ratio preservation**:
```python
score = evaluate_cross_ratio_preservation(model, data)
print(f"CR Score: {score:.2f}")
```

4. **Handle homogeneous coordinates**:
```python
features = x[..., :-1]  # Regular features
h_coord = x[..., -1:]  # Homogeneous coordinate
```

## Common Issues

1. **NaN Loss**
   - Use proper clamping: `loss = torch.clamp(loss, max=100.0)`
   - Check normalization: `x = uhg_normalize(x)`

2. **Poor Cross-Ratio Preservation**
   - Add spread regularization: `spread_weight=0.1`
   - Use UHG-compliant operations

3. **Dimension Mismatch**
   - Account for homogeneous coordinate: `dim + 1`
   - Check tensor shapes carefully 