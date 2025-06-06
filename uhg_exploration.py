#!/usr/bin/env python3
"""
UHG Exploration - Exploring the Universal Hyperbolic Geometry library from PyPI
"""

import torch
import uhg
from uhg.projective import ProjectiveUHG
import matplotlib.pyplot as plt
import numpy as np

def explore_quadrance():
    """Explore the quadrance function with various examples."""
    print("\n=== EXPLORING QUADRANCE ===")
    
    # Initialize UHG
    model = ProjectiveUHG()
    
    # Create points in projective coordinates [x:y:z]
    print("Creating test points...")
    
    # Points on the unit circle (null points)
    null_point1 = torch.tensor([1.0, 0.0, 1.0])  # x² + y² = z²
    null_point2 = torch.tensor([0.0, 1.0, 1.0])  # x² + y² = z²
    
    # Points inside the unit circle (proper hyperbolic points)
    hyp_point1 = torch.tensor([0.5, 0.0, 1.0])  # x² + y² < z²
    hyp_point2 = torch.tensor([0.0, 0.5, 1.0])  # x² + y² < z²
    
    # Points outside the unit circle
    ext_point1 = torch.tensor([2.0, 0.0, 1.0])  # x² + y² > z²
    ext_point2 = torch.tensor([0.0, 2.0, 1.0])  # x² + y² > z²
    
    # Check if points are null
    print(f"null_point1 is null: {model.is_null_point(null_point1)}")
    print(f"hyp_point1 is null: {model.is_null_point(hyp_point1)}")
    print(f"ext_point1 is null: {model.is_null_point(ext_point1)}")
    
    # Calculate quadrances between different types of points
    try:
        # Between hyperbolic points
        q_hyp = model.quadrance(hyp_point1, hyp_point2)
        print(f"Quadrance between hyperbolic points: {q_hyp.item():.6f}")
        
        # Between external points
        q_ext = model.quadrance(ext_point1, ext_point2)
        print(f"Quadrance between external points: {q_ext.item():.6f}")
        
        # Between hyperbolic and external points
        q_mixed = model.quadrance(hyp_point1, ext_point1)
        print(f"Quadrance between hyperbolic and external points: {q_mixed.item():.6f}")
        
        # Between null points (should raise an error)
        q_null = model.quadrance(null_point1, null_point2)
        print(f"Quadrance between null points: {q_null.item():.6f}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test perpendicular points
    perp1 = torch.tensor([1.0, 0.0, 2.0])  # Not normalized
    perp2 = torch.tensor([0.0, 1.0, 0.0])  # Not normalized
    
    # Calculate hyperbolic dot product to verify perpendicularity
    dot = model.hyperbolic_dot(perp1, perp2)
    print(f"Hyperbolic dot product of perpendicular points: {dot.item():.6f}")
    
    # Calculate quadrance between perpendicular points
    q_perp = model.quadrance(perp1, perp2)
    print(f"Quadrance between perpendicular points: {q_perp.item():.6f} (should be 1.0)")

def explore_spread():
    """Explore the spread function with various examples."""
    print("\n=== EXPLORING SPREAD ===")
    
    # Initialize UHG
    model = ProjectiveUHG()
    
    # Create lines in projective coordinates [l:m:n]
    print("Creating test lines...")
    
    # Standard lines
    x_axis = torch.tensor([1.0, 0.0, 0.0])  # x-axis: y = 0
    y_axis = torch.tensor([0.0, 1.0, 0.0])  # y-axis: x = 0
    diag_line = torch.tensor([1.0, 1.0, 0.0])  # Diagonal line: y = -x
    
    # Null lines (passing through the origin)
    null_line1 = torch.tensor([1.0, 0.0, 1.0])  # l² + m² = n²
    null_line2 = torch.tensor([0.0, 1.0, 1.0])  # l² + m² = n²
    
    # Check if lines are null
    print(f"x_axis is null: {model.is_null_line(x_axis)}")
    print(f"null_line1 is null: {model.is_null_line(null_line1)}")
    
    # Calculate spreads between different types of lines
    try:
        # Between standard lines
        s_std = model.spread(x_axis, y_axis)
        print(f"Spread between x-axis and y-axis: {s_std.item():.6f} (should be 1.0 for perpendicular lines)")
        
        s_diag = model.spread(x_axis, diag_line)
        print(f"Spread between x-axis and diagonal line: {s_diag.item():.6f}")
        
        # Between null lines (should raise an error)
        s_null = model.spread(null_line1, null_line2)
        print(f"Spread between null lines: {s_null.item():.6f}")
    except ValueError as e:
        print(f"Error: {e}")

def explore_cross_ratio():
    """Explore the cross ratio function with various examples."""
    print("\n=== EXPLORING CROSS RATIO ===")
    
    # Initialize UHG
    model = ProjectiveUHG()
    
    # Create points in projective coordinates [x:y:z]
    print("Creating test points...")
    
    # Create four collinear points
    a = torch.tensor([0.0, 0.0, 1.0])  # Origin
    b = torch.tensor([1.0, 0.0, 1.0])  # Point on x-axis
    c = torch.tensor([2.0, 0.0, 1.0])  # Another point on x-axis
    d = torch.tensor([3.0, 0.0, 1.0])  # Another point on x-axis
    
    # Calculate cross ratio
    cr = model.cross_ratio(a, b, c, d)
    print(f"Cross ratio of four collinear points: {cr.item():.6f}")
    
    # Test invariance under projective transformation
    # Create a projective transformation matrix
    matrix = model.get_projective_matrix(dim=2)
    
    # Transform the points
    a_t = model.transform(a, matrix)
    b_t = model.transform(b, matrix)
    c_t = model.transform(c, matrix)
    d_t = model.transform(d, matrix)
    
    # Calculate cross ratio of transformed points
    cr_t = model.cross_ratio(a_t, b_t, c_t, d_t)
    print(f"Cross ratio after projective transformation: {cr_t.item():.6f}")
    print(f"Difference: {abs(cr.item() - cr_t.item()):.10f}")
    
    # Test with non-collinear points
    p1 = torch.tensor([1.0, 0.0, 1.0])
    p2 = torch.tensor([0.0, 1.0, 1.0])
    p3 = torch.tensor([1.0, 1.0, 1.0])
    p4 = torch.tensor([2.0, 2.0, 1.0])
    
    cr_nc = model.cross_ratio(p1, p2, p3, p4)
    print(f"Cross ratio of non-collinear points: {cr_nc.item():.6f}")

def explore_geometric_theorems():
    """Explore the geometric theorems in UHG."""
    print("\n=== EXPLORING GEOMETRIC THEOREMS ===")
    
    # Initialize UHG
    model = ProjectiveUHG()
    
    # Create a proper right triangle in hyperbolic geometry
    # We'll create a triangle where one of the angles is a right angle (spread = 1)
    A = torch.tensor([0.3, 0.3, 1.0])  # Small offset from origin
    B = torch.tensor([0.7, 0.3, 1.0])  # Point on same "height" as A
    C = torch.tensor([0.3, 0.7, 1.0])  # Point above A
    
    # Calculate the lines
    line_AB = model.join(A, B)
    line_AC = model.join(A, C)
    line_BC = model.join(B, C)
    
    # Calculate the spreads
    s_A = model.spread(line_AB, line_AC)
    s_B = model.spread(line_AB, line_BC)
    s_C = model.spread(line_AC, line_BC)
    
    print(f"Triangle spreads: s_A={s_A.item():.6f}, s_B={s_B.item():.6f}, s_C={s_C.item():.6f}")
    
    # If s_A is not close to 1 (right angle), let's adjust the points to make it a right angle
    if abs(s_A.item() - 1.0) > 0.01:
        print("Adjusting triangle to create a right angle at A...")
        
        # Create a right angle at A by making AC perpendicular to AB
        # We'll keep A and B fixed, and adjust C
        
        # Get the direction of AB
        AB_dir = B - A
        
        # Create a perpendicular direction in the plane
        perp_dir = torch.tensor([-AB_dir[1], AB_dir[0], 0.0])
        
        # Normalize and scale
        perp_dir = perp_dir / torch.norm(perp_dir) * 0.4
        
        # Set C to be in the perpendicular direction from A
        C = A + perp_dir
        
        # Recalculate the lines and spreads
        line_AB = model.join(A, B)
        line_AC = model.join(A, C)
        line_BC = model.join(B, C)
        
        s_A = model.spread(line_AB, line_AC)
        s_B = model.spread(line_AB, line_BC)
        s_C = model.spread(line_AC, line_BC)
        
        print(f"Adjusted triangle spreads: s_A={s_A.item():.6f}, s_B={s_B.item():.6f}, s_C={s_C.item():.6f}")
    
    # Calculate the quadrances
    q_AB = model.quadrance(A, B)
    q_AC = model.quadrance(A, C)
    q_BC = model.quadrance(B, C)
    
    print(f"Triangle quadrances: q_AB={q_AB.item():.6f}, q_AC={q_AC.item():.6f}, q_BC={q_BC.item():.6f}")
    
    # Test Pythagoras theorem
    # For a right triangle (s_A = 1), q_BC = q_AB + q_AC - q_AB*q_AC
    expected_q_BC = q_AB + q_AC - q_AB * q_AC
    print(f"Expected q_BC from Pythagoras: {expected_q_BC.item():.6f}")
    print(f"Actual q_BC: {q_BC.item():.6f}")
    print(f"Difference: {abs(expected_q_BC.item() - q_BC.item()):.10f}")
    print(f"Pythagoras theorem verified: {abs(expected_q_BC.item() - q_BC.item()) < 1e-5}")
    
    # Test Triple Quad Formula
    # (q₁ + q₂ + q₃)² = 2(q₁² + q₂² + q₃²) + 4q₁q₂q₃
    lhs = (q_AB + q_AC + q_BC)**2
    rhs = 2*(q_AB**2 + q_AC**2 + q_BC**2) + 4*q_AB*q_AC*q_BC
    print(f"Triple Quad Formula: LHS={lhs.item():.6f}, RHS={rhs.item():.6f}")
    print(f"Difference: {abs(lhs.item() - rhs.item()):.10f}")
    print(f"Triple Quad Formula verified: {abs(lhs.item() - rhs.item()) < 1e-5}")
    
    # Test Triple Spread Formula
    # (S₁ + S₂ + S₃)² = 2(S₁² + S₂² + S₃²) + 4S₁S₂S₃
    lhs_s = (s_A + s_B + s_C)**2
    rhs_s = 2*(s_A**2 + s_B**2 + s_C**2) + 4*s_A*s_B*s_C
    print(f"Triple Spread Formula: LHS={lhs_s.item():.6f}, RHS={rhs_s.item():.6f}")
    print(f"Difference: {abs(lhs_s.item() - rhs_s.item()):.10f}")
    print(f"Triple Spread Formula verified: {abs(lhs_s.item() - rhs_s.item()) < 1e-5}")
    
    # Test Cross Law
    # q₁q₂q₃S₁S₂S₃ = (q₁q₂S₃ + q₂q₃S₁ + q₃q₁S₂ - q₁ - q₂ - q₃ - S₁ - S₂ - S₃ + 2)²
    lhs_cross = q_AB * q_AC * q_BC * s_A * s_B * s_C
    inside_term = q_AB*q_AC*s_C + q_AC*q_BC*s_A + q_BC*q_AB*s_B - q_AB - q_AC - q_BC - s_A - s_B - s_C + 2
    rhs_cross = inside_term**2
    print(f"Cross Law: LHS={lhs_cross.item():.6f}, RHS={rhs_cross.item():.6f}")
    print(f"Difference: {abs(lhs_cross.item() - rhs_cross.item()):.10f}")
    print(f"Cross Law verified: {abs(lhs_cross.item() - rhs_cross.item()) < 1e-5}")

def visualize_poincare_disk():
    """Visualize points and lines in the Poincaré disk model."""
    print("\n=== VISUALIZING POINCARÉ DISK MODEL ===")
    
    # Initialize UHG
    model = ProjectiveUHG()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw the unit circle (boundary of the Poincaré disk)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
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
    
    # Create points in projective coordinates
    points = [
        torch.tensor([0.0, 0.0, 1.0]),  # Origin
        torch.tensor([0.5, 0.0, 1.0]),  # Point on x-axis
        torch.tensor([0.0, 0.5, 1.0]),  # Point on y-axis
        torch.tensor([0.5, 0.5, 1.0]),  # Diagonal point
        torch.tensor([0.7, 0.0, 1.0]),  # Another point on x-axis
        torch.tensor([0.0, 0.7, 1.0]),  # Another point on y-axis
        torch.tensor([0.7, 0.7, 1.0]),  # Another diagonal point
    ]
    
    labels = ['O', 'A', 'B', 'C', 'D', 'E', 'F']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    # Plot the points
    for point, label, color in zip(points, labels, colors):
        x, y = to_poincare(point)
        ax.plot(x, y, 'o', color=color, markersize=8)
        ax.text(x+0.05, y+0.05, label, fontsize=12, color=color)
    
    # Draw hyperbolic lines (geodesics in the Poincaré disk)
    # In the Poincaré disk, hyperbolic lines are either:
    # 1. Diameters of the disk
    # 2. Circular arcs that intersect the boundary orthogonally
    
    # Draw some diameters
    ax.plot([-1, 1], [0, 0], 'k-', linewidth=1)  # x-axis
    ax.plot([0, 0], [-1, 1], 'k-', linewidth=1)  # y-axis
    
    # Draw some circular arcs
    # For simplicity, we'll use a function to draw arcs between points
    def draw_geodesic(p1, p2, color='black', linewidth=1):
        x1, y1 = to_poincare(p1)
        x2, y2 = to_poincare(p2)
        
        # If one point is the origin, the geodesic is a straight line
        if (abs(x1) < 1e-10 and abs(y1) < 1e-10) or (abs(x2) < 1e-10 and abs(y2) < 1e-10):
            ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=linewidth)
            return
        
        # If the points lie on a diameter, the geodesic is a straight line
        if abs(x1*y2 - x2*y1) < 1e-10:  # Cross product near zero means collinear with origin
            ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=linewidth)
            return
        
        # Otherwise, compute the circle that contains both points and is orthogonal to the boundary
        # We'll use a simplified approach to find the center and radius of this circle
        
        # Calculate the Euclidean distance between the points
        d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Calculate the midpoint
        mx, my = (x1+x2)/2, (y1+y2)/2
        
        # Calculate the perpendicular direction
        nx, ny = (y2-y1)/d, (x1-x2)/d
        
        # Calculate the distance from the midpoint to the center
        # This is a simplified formula for the Poincaré disk model
        h = (1 - (x1**2 + y1**2)) * (1 - (x2**2 + y2**2)) / (2 * d**2)
        h = np.sqrt(max(0, 1/4 + h))
        
        # Calculate the center of the circle
        cx, cy = mx + h*nx, my + h*ny
        
        # Calculate the radius
        r = np.sqrt((cx-x1)**2 + (cy-y1)**2)
        
        # Calculate the angles to the points from the center
        angle1 = np.arctan2(y1-cy, x1-cx)
        angle2 = np.arctan2(y2-cy, x2-cx)
        
        # Ensure we draw the shorter arc
        if abs(angle1-angle2) > np.pi:
            if angle1 > angle2:
                angle2 += 2*np.pi
            else:
                angle1 += 2*np.pi
                
        # Draw the arc
        angles = np.linspace(angle1, angle2, 100)
        arc_x = cx + r * np.cos(angles)
        arc_y = cy + r * np.sin(angles)
        ax.plot(arc_x, arc_y, '-', color=color, linewidth=linewidth)
    
    # Draw geodesics between some points
    draw_geodesic(points[0], points[1], 'blue')  # O to A
    draw_geodesic(points[0], points[2], 'green')  # O to B
    draw_geodesic(points[1], points[2], 'red')  # A to B
    draw_geodesic(points[3], points[6], 'purple')  # C to F
    
    # Set axis limits and labels
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Poincaré Disk Model of Hyperbolic Geometry')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig('poincare_disk.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'poincare_disk.png'")
    
    # Show the figure
    plt.close()

def main():
    """Main function to explore the UHG library."""
    print(f"UHG Version: {uhg.__version__}")
    
    # Explore the core functions
    explore_quadrance()
    explore_spread()
    explore_cross_ratio()
    explore_geometric_theorems()
    
    # Visualize the Poincaré disk model
    visualize_poincare_disk()

if __name__ == "__main__":
    main() 