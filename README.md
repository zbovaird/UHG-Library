# UHG-Based Anomaly Detection

This repository contains an implementation of anomaly detection using Universal Hyperbolic Geometry (UHG) principles. The code leverages the UHG library to perform geometric operations in hyperbolic space.

## Overview

Universal Hyperbolic Geometry (UHG) provides a powerful framework for representing complex hierarchical data. This implementation uses UHG principles to detect anomalies in network traffic data by embedding the data in hyperbolic space and identifying points that deviate from the normal patterns.

## Key Features

- UHG-based graph neural network for anomaly detection
- Efficient implementation of UHG operations using the UHG library
- Support for large-scale datasets through batched processing
- Visualization tools for analyzing embeddings and anomaly scores

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/uhg-anomaly-detection.git
cd uhg-anomaly-detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from uhg_anomaly_detection_refactored import load_and_preprocess_data, create_graph_data, UHGGraphNN, UHGAnomalyLoss

# Load and preprocess data
features, labels, feature_info = load_and_preprocess_data("your_data.csv")

# Create graph data
graph_data = create_graph_data(features, labels, k=5)

# Create model
model = UHGGraphNN(
    in_channels=features.shape[1],
    hidden_channels=64,
    embedding_dim=32,
    num_layers=2,
    dropout=0.2
)

# Create loss function
criterion = UHGAnomalyLoss(spread_weight=0.1, quad_weight=1.0, margin=1.0)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Train the model
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    z = model(graph_data.x, graph_data.edge_index)
    loss = criterion(z, graph_data.edge_index)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Generate embeddings
model.eval()
with torch.no_grad():
    embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()
```

### Using UHG Utilities Directly

The `uhg_utils.py` file provides a clean interface to the UHG library. You can use these utilities in your own code:

```python
from uhg_utils import uhg_quadrance, uhg_spread, uhg_cross_ratio, to_uhg_space

# Convert points to UHG space
points = to_uhg_space(your_data)

# Compute quadrance (UHG distance) between points
distance = uhg_quadrance(points[0], points[1])

# Compute spread (UHG angle) between lines
spread = uhg_spread(line1, line2)

# Compute cross-ratio of four points
cr = uhg_cross_ratio(p1, p2, p3, p4)
```

## File Structure

- `uhg_utils.py`: Utility functions that provide a clean interface to the UHG library
- `uhg_anomaly_detection_refactored.py`: Main implementation of UHG-based anomaly detection
- `requirements.txt`: List of required dependencies

## UHG Library Integration

This implementation uses the UHG library for core geometric operations. The `uhg_utils.py` file serves as a bridge between the application code and the UHG library, providing a clean interface and handling edge cases.

Key UHG operations used:
- `quadrance`: Computes the squared distance between points in hyperbolic space
- `spread`: Computes the squared angle between lines in hyperbolic space
- `cross_ratio`: Computes the cross-ratio of four points, a projective invariant
- `normalize_points`: Normalizes points according to UHG conventions
- `join_points`: Computes the line joining two points
- `meet_line_point`: Computes the intersection of a line and a point

## References

- Norman J. Wildberger, "Universal Hyperbolic Geometry I: Trigonometry", Geometriae Dedicata, 2013
- Norman J. Wildberger, "Universal Hyperbolic Geometry II: A pictorial overview", KoG, 2013
- Norman J. Wildberger, "Universal Hyperbolic Geometry III: First Steps in Projective Triangle Geometry", KoG, 2014

## License

MIT License
