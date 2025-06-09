#!/usr/bin/env python3
"""
UHG Examples - Demonstrates key functions from the Universal Hyperbolic Geometry library
"""

import torch
import uhg
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Initialize UHG model (using ProjectiveUHG class)
    model = uhg.ProjectiveUHG()
    print(f"UHG Version: {uhg.__version__}")
    
    # Set up device (CPU or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example 1: Quadrance between points
    print("\n=== Example 1: Quadrance between points ===")
    # Create two points in projective coordinates [x:y:z]
    p1 = torch.tensor([1.0, 0.0, 1.0], device=device)  # Point on the unit circle
    p2 = torch.tensor([0.0, 1.0, 1.0], device=device)  # Another point on the unit circle
    p3 = torch.tensor([2.0, 0.0, 2.0], device=device)  # Same as p1 (projectively equivalent)
    p4 = torch.tensor([0.0, 0.0, 1.0], device=device)  # Origin
    
    # Calculate quadrances
    q12 = model.quadrance(p1, p2)
    q13 = model.quadrance(p1, p3)
    q14 = model.quadrance(p1, p4)
    
    print(f"Point p1: {p1}")
    print(f"Point p2: {p2}")
    print(f"Point p3: {p3} (projectively equivalent to p1)")
    print(f"Point p4: {p4} (origin)")
    print(f"Quadrance q(p1,p2): {q12.item():.6f}")
    print(f"Quadrance q(p1,p3): {q13.item():.6f} (should be 0 as p1 and p3 are projectively equivalent)")
    print(f"Quadrance q(p1,p4): {q14.item():.6f}")
    
    # Example 2: Spread between lines
    print("\n=== Example 2: Spread between lines ===")
    # Create lines in projective coordinates [l:m:n]
    l1 = torch.tensor([1.0, 0.0, 0.0], device=device)  # x-axis
    l2 = torch.tensor([0.0, 1.0, 0.0], device=device)  # y-axis
    l3 = torch.tensor([1.0, 1.0, 0.0], device=device)  # Line y = -x
    
    # Calculate spreads
    s12 = model.spread(l1, l2)
    s13 = model.spread(l1, l3)
    s23 = model.spread(l2, l3)
    
    print(f"Line l1: {l1} (x-axis)")
    print(f"Line l2: {l2} (y-axis)")
    print(f"Line l3: {l3} (line y = -x)")
    print(f"Spread S(l1,l2): {s12.item():.6f} (should be 1 as lines are perpendicular)")
    print(f"Spread S(l1,l3): {s13.item():.6f}")
    print(f"Spread S(l2,l3): {s23.item():.6f}")
    
    # Example 3: Cross ratio
    print("\n=== Example 3: Cross ratio ===")
    # Create four points for cross ratio
    a = torch.tensor([1.0, 0.0, 1.0], device=device)
    b = torch.tensor([0.0, 1.0, 1.0], device=device)
    c = torch.tensor([1.0, 1.0, 1.0], device=device)
    d = torch.tensor([2.0, 1.0, 2.0], device=device)
    
    # Calculate cross ratio
    cr = model.cross_ratio(a, b, c, d)
    
    print(f"Point a: {a}")
    print(f"Point b: {b}")
    print(f"Point c: {c}")
    print(f"Point d: {d}")
    print(f"Cross ratio CR(a,b;c,d): {cr.item():.6f}")
    
    # Example 4: Hyperbolic Pythagoras theorem
    print("\n=== Example 4: Hyperbolic Pythagoras theorem ===")
    # Create a right triangle
    # First, create three points
    A = torch.tensor([0.0, 0.0, 1.0], device=device)  # Origin
    B = torch.tensor([1.0, 0.0, 1.0], device=device)  # Point on x-axis
    C = torch.tensor([0.0, 1.0, 1.0], device=device)  # Point on y-axis
    
    # Calculate the quadrances
    q_AB = model.quadrance(A, B)
    q_AC = model.quadrance(A, C)
    q_BC = model.quadrance(B, C)
    
    # Calculate the lines
    line_AB = model.join(A, B)
    line_AC = model.join(A, C)
    line_BC = model.join(B, C)
    
    # Calculate the spreads
    s_A = model.spread(line_AB, line_AC)
    s_B = model.spread(line_AB, line_BC)
    s_C = model.spread(line_AC, line_BC)
    
    print(f"Point A: {A} (origin)")
    print(f"Point B: {B} (on x-axis)")
    print(f"Point C: {C} (on y-axis)")
    print(f"Quadrance q(A,B): {q_AB.item():.6f}")
    print(f"Quadrance q(A,C): {q_AC.item():.6f}")
    print(f"Quadrance q(B,C): {q_BC.item():.6f}")
    print(f"Spread at A: {s_A.item():.6f}")
    print(f"Spread at B: {s_B.item():.6f}")
    print(f"Spread at C: {s_C.item():.6f}")
    
    # Verify Pythagoras theorem: q_BC = q_AB + q_AC - q_AB*q_AC
    expected_q_BC = q_AB + q_AC - q_AB * q_AC
    print(f"Expected q(B,C) from Pythagoras: {expected_q_BC.item():.6f}")
    print(f"Actual q(B,C): {q_BC.item():.6f}")
    print(f"Difference: {abs(expected_q_BC.item() - q_BC.item()):.10f}")
    print(f"Pythagoras theorem verified: {model.pythagoras(q_AB, q_AC, q_BC).item()}")
    
    # Example 5: Triple quad formula
    print("\n=== Example 5: Triple quad formula ===")
    # Use the quadrances from the previous example
    lhs = (q_AB + q_AC + q_BC)**2
    rhs = 2*(q_AB**2 + q_AC**2 + q_BC**2) + 4*q_AB*q_AC*q_BC
    
    print(f"Left side of triple quad formula: {lhs.item():.6f}")
    print(f"Right side of triple quad formula: {rhs.item():.6f}")
    print(f"Difference: {abs(lhs.item() - rhs.item()):.10f}")
    print(f"Triple quad formula verified: {model.triple_quad_formula(q_AB, q_AC, q_BC).item()}")
    
    # Example 6: Cross law
    print("\n=== Example 6: Cross law ===")
    # Use the quadrances and spreads from the previous example
    lhs_cross = q_AB * q_AC * q_BC * s_A * s_B * s_C
    inside_term = q_AB*q_AC*s_C + q_AC*q_BC*s_A + q_BC*q_AB*s_B - q_AB - q_AC - q_BC - s_A - s_B - s_C + 2
    rhs_cross = inside_term**2
    
    print(f"Left side of cross law: {lhs_cross.item():.6f}")
    print(f"Right side of cross law: {rhs_cross.item():.6f}")
    print(f"Difference: {abs(lhs_cross.item() - rhs_cross.item()):.10f}")
    print(f"Cross law verified: {model.cross_law(q_AB, q_AC, q_BC, s_A, s_B, s_C).item()}")
    
    # Example 7: Visualization of points in the Poincaré disk model
    print("\n=== Example 7: Visualization of points in the Poincaré disk model ===")
    print("Generating visualization (will save as 'uhg_visualization.png')...")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the unit circle (boundary of the Poincaré disk)
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(circle)
    
    # Convert projective coordinates to Poincaré disk coordinates
    def to_poincare(point):
        # Normalize to ensure z=1
        point = point / point[2]
        # Extract x and y
        x, y = point[0].item(), point[1].item()
        # Check if point is inside the disk
        if x**2 + y**2 < 1:
            return x, y
        else:
            # Project points outside the disk to the boundary
            norm = np.sqrt(x**2 + y**2)
            return x/norm * 0.99, y/norm * 0.99
    
    # Plot the points from our examples
    points = [p1, p2, p3, p4, A, B, C]
    labels = ['p1', 'p2', 'p3', 'p4', 'A', 'B', 'C']
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink']
    
    for point, label, color in zip(points, labels, colors):
        x, y = to_poincare(point.cpu().numpy())
        ax.plot(x, y, 'o', color=color, markersize=8)
        ax.text(x+0.05, y+0.05, label, fontsize=12)
    
    # Draw some hyperbolic lines (geodesics in the Poincaré disk)
    # In the Poincaré disk, hyperbolic lines are either:
    # 1. Diameters of the disk
    # 2. Circular arcs that intersect the boundary orthogonally
    
    # Draw the line between A and B (a diameter)
    ax.plot([0, 1], [0, 0], 'k-', linewidth=1)
    
    # Draw the line between A and C (a diameter)
    ax.plot([0, 0], [0, 1], 'k-', linewidth=1)
    
    # Draw the line between B and C (a circular arc)
    # For simplicity, we'll approximate with a straight line
    x1, y1 = to_poincare(B.cpu().numpy())
    x2, y2 = to_poincare(C.cpu().numpy())
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    
    # Set axis limits and labels
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Points in the Poincaré Disk Model')
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Save the figure
    plt.savefig('uhg_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'uhg_visualization.png'")

if __name__ == "__main__":
    main() 