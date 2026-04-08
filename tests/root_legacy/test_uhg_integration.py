"""
Test script to verify UHG library integration.

This script tests the basic functionality of the UHG utilities and
ensures that they correctly interface with the UHG library.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from uhg_utils import (
    uhg_inner_product,
    uhg_norm,
    uhg_quadrance,
    uhg_spread,
    uhg_cross_ratio,
    to_uhg_space,
    normalize_points,
    get_uhg_instance
)

def test_uhg_operations():
    """Test basic UHG operations using the utility functions."""
    print("Testing UHG operations...")
    
    # Create some test points in Euclidean space
    points_euclidean = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.0]
    ], dtype=torch.float32)
    
    # Convert to UHG space
    points_uhg = to_uhg_space(points_euclidean)
    print(f"Points in UHG space:\n{points_uhg}")
    
    # Test inner product
    inner_prod = uhg_inner_product(points_uhg[0], points_uhg[1])
    print(f"Inner product: {inner_prod.item()}")
    
    # Test norm
    norm = uhg_norm(points_uhg[0])
    print(f"Norm: {norm.item()}")
    
    # Test quadrance
    quad = uhg_quadrance(points_uhg[0], points_uhg[1])
    print(f"Quadrance: {quad.item()}")
    
    # Test spread
    # Create lines by joining points
    uhg = get_uhg_instance()
    line1 = uhg.join(points_uhg[0], points_uhg[2])
    line2 = uhg.join(points_uhg[1], points_uhg[3])
    spread = uhg_spread(line1, line2)
    print(f"Spread: {spread.item()}")
    
    # Test cross ratio
    cr = uhg_cross_ratio(points_uhg[0], points_uhg[1], points_uhg[2], points_uhg[3])
    print(f"Cross ratio: {cr.item()}")
    
    print("UHG operations test completed.")
    
def test_cross_ratio_invariance():
    """Test that cross ratio is invariant under projective transformations."""
    print("\nTesting cross ratio invariance...")
    
    # Create some test points in UHG space
    points = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    # Compute initial cross ratio
    cr_initial = uhg_cross_ratio(points[0], points[1], points[2], points[3])
    print(f"Initial cross ratio: {cr_initial.item()}")
    
    # Create a projective transformation matrix
    uhg = get_uhg_instance()
    transform = uhg.get_projective_matrix(3)
    
    # Apply transformation
    transformed_points = torch.stack([
        uhg.transform(points[0], transform),
        uhg.transform(points[1], transform),
        uhg.transform(points[2], transform),
        uhg.transform(points[3], transform)
    ])
    
    # Compute cross ratio after transformation
    cr_transformed = uhg_cross_ratio(
        transformed_points[0], 
        transformed_points[1], 
        transformed_points[2], 
        transformed_points[3]
    )
    print(f"Transformed cross ratio: {cr_transformed.item()}")
    
    # Check invariance
    is_invariant = torch.abs(cr_initial - cr_transformed) < 1e-5
    print(f"Cross ratio is invariant: {is_invariant.item()}")
    
    print("Cross ratio invariance test completed.")
    
def visualize_uhg_distance():
    """Visualize UHG distance (quadrance) between points."""
    print("\nVisualizing UHG distance...")
    
    # Create a grid of points in Euclidean space
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Convert to PyTorch tensor
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # Convert to UHG space
    grid_uhg = to_uhg_space(grid_tensor)
    
    # Define a reference point
    ref_point = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    
    # Compute quadrance from reference point to all grid points
    quadrances = []
    batch_size = 1000
    for i in range(0, len(grid_uhg), batch_size):
        batch = grid_uhg[i:i+batch_size]
        batch_quad = uhg_quadrance(ref_point.expand_as(batch), batch)
        quadrances.append(batch_quad)
    
    quadrances = torch.cat(quadrances)
    
    # Reshape for plotting
    quadrance_grid = quadrances.reshape(100, 100).numpy()
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, quadrance_grid, 50, cmap='viridis')
    plt.colorbar(label='UHG Quadrance')
    plt.title('UHG Distance (Quadrance) from Origin')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig('uhg_distance_visualization.png')
    print("Visualization saved as 'uhg_distance_visualization.png'")
    
    print("UHG distance visualization completed.")

def main():
    """Run all tests."""
    test_uhg_operations()
    test_cross_ratio_invariance()
    visualize_uhg_distance()
    
if __name__ == "__main__":
    main() 