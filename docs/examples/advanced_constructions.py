"""
Advanced Geometric Constructions in UHG

This module demonstrates advanced geometric constructions using the UHG library.
It serves as both documentation and a practical guide for implementing complex
geometric operations in hyperbolic space.
"""

import torch
import matplotlib.pyplot as plt
from uhg.projective import ProjectiveUHG

def plot_hyperbolic_points(points, labels=None, title="Hyperbolic Points"):
    """Plot points in the hyperbolic disk model."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw unit circle (boundary of hyperbolic disk)
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(circle)
    
    # Convert projective coordinates to disk model
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    
    # Plot points
    ax.scatter(x, y, c='blue')
    
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]))
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    
    return fig, ax

def demonstrate_midpoints():
    """Demonstrate midpoint construction with visualization."""
    uhg = ProjectiveUHG()
    
    # Create two points in hyperbolic space
    A = torch.tensor([0.3, 0.0, 1.0])  # Point on x-axis
    B = torch.tensor([0.4, 0.2, 1.0])  # Point at angle to x-axis
    
    # Calculate midpoints
    m1, m2 = uhg.midpoints(A, B)
    
    # Verify midpoint properties
    assert uhg.verify_midpoints(A, B, m1, m2)
    
    # Plot points
    points = torch.stack([A, B, m1, m2])
    labels = ['A', 'B', 'm₁', 'm₂']
    fig, ax = plot_hyperbolic_points(points, labels, "Midpoint Construction")
    
    # Print properties
    print("Midpoint Properties:")
    print(f"Quadrance A to m₁: {uhg.quadrance(A, m1):.4f}")
    print(f"Quadrance B to m₁: {uhg.quadrance(B, m1):.4f}")
    print(f"Quadrance A to m₂: {uhg.quadrance(A, m2):.4f}")
    print(f"Quadrance B to m₂: {uhg.quadrance(B, m2):.4f}")
    print(f"m₁⊥m₂ dot product: {uhg.hyperbolic_dot(m1, m2):.4e}")
    print(f"Cross-ratio (A,B:m₁,m₂): {uhg.cross_ratio(A, B, m1, m2):.4f}")
    
    return fig

def demonstrate_projective_invariance():
    """Demonstrate invariance under projective transformations."""
    uhg = ProjectiveUHG()
    
    # Create original points
    A = torch.tensor([0.3, 0.0, 1.0])
    B = torch.tensor([0.4, 0.2, 1.0])
    m1, m2 = uhg.midpoints(A, B)
    
    # Create random projective transformation
    matrix = uhg.get_projective_matrix(2)
    
    # Transform all points
    A_trans = uhg.transform(A, matrix)
    B_trans = uhg.transform(B, matrix)
    m1_trans = uhg.transform(m1, matrix)
    m2_trans = uhg.transform(m2, matrix)
    
    # Calculate new midpoints directly
    m1_new, m2_new = uhg.midpoints(A_trans, B_trans)
    
    # Verify properties are preserved
    print("\nProjective Invariance Check:")
    print("Original cross-ratio:", uhg.cross_ratio(A, B, m1, m2).item())
    print("Transformed cross-ratio:", uhg.cross_ratio(A_trans, B_trans, m1_trans, m2_trans).item())
    print("New midpoints cross-ratio:", uhg.cross_ratio(A_trans, B_trans, m1_new, m2_new).item())
    
    # Plot original and transformed configurations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original points
    points = torch.stack([A, B, m1, m2])
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    ax1.scatter(x, y)
    ax1.set_title("Original Configuration")
    
    # Transformed points
    points_trans = torch.stack([A_trans, B_trans, m1_trans, m2_trans])
    x = points_trans[:, 0] / points_trans[:, 2]
    y = points_trans[:, 1] / points_trans[:, 2]
    ax2.scatter(x, y)
    ax2.set_title("Transformed Configuration")
    
    for ax in [ax1, ax2]:
        circle = plt.Circle((0, 0), 1, fill=False, color='black')
        ax.add_artist(circle)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True)
    
    return fig

def demonstrate_edge_cases():
    """Demonstrate handling of edge cases in geometric constructions."""
    uhg = ProjectiveUHG()
    
    print("\nEdge Cases in UHG:")
    
    # Case 1: Points too far apart
    print("\n1. Points too far apart:")
    A = torch.tensor([0.3, 0.0, 1.0])
    B = torch.tensor([2.0, 0.0, 1.0])  # Outside unit disk
    m1, m2 = uhg.midpoints(A, B)
    print("Midpoints exist:", m1 is not None and m2 is not None)
    
    # Case 2: Null point
    print("\n2. Null point:")
    null_point = torch.tensor([1.0, 0.0, 1.0])  # On boundary
    regular_point = torch.tensor([0.3, 0.0, 1.0])
    m1, m2 = uhg.midpoints(null_point, regular_point)
    print("First midpoint is null point:", torch.allclose(m1, null_point))
    print("Second midpoint exists:", m2 is not None)
    
    # Case 3: Same point
    print("\n3. Same point:")
    m1, m2 = uhg.midpoints(A, A)
    print("First midpoint is input point:", torch.allclose(m1, A))
    print("Second midpoint exists:", m2 is not None)
    
    # Case 4: Nearly coincident points
    print("\n4. Nearly coincident points:")
    B_close = A + torch.tensor([1e-6, 0.0, 0.0])
    m1, m2 = uhg.midpoints(A, B_close)
    print("Midpoints calculated:", m1 is not None and m2 is not None)
    if m1 is not None:
        print("First midpoint close to input:", torch.allclose(m1, A, rtol=1e-4))

if __name__ == "__main__":
    # Run demonstrations
    fig1 = demonstrate_midpoints()
    fig2 = demonstrate_projective_invariance()
    demonstrate_edge_cases()
    
    plt.show()
