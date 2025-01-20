# Universal Hyperbolic Geometry Reference

## Chapter 3: Projective Geometry

### Core Concepts
Projective geometry forms the foundation of Universal Hyperbolic Geometry (UHG). Key concepts include:

1. **Homogeneous Coordinates**: Points are represented using homogeneous coordinates [x:y:z], where:
   - Regular points satisfy x² + y² < z²
   - Null points satisfy x² + y² = z²
   - Points at infinity have z = 0

2. **Duality**: Every point has a dual line and vice versa:
   - Point [x:y:z] Line (x:y:z)
   - This duality is fundamental to UHG's elegant formulation

### Points and Lines
1. **Point Representation**:
   ```python
   # Regular point inside hyperbolic disk
   point = torch.tensor([0.3, 0.0, 1.0])  # x² + y² < z²
   
   # Null point on boundary
   null_point = torch.tensor([1.0, 0.0, 1.0])  # x² + y² = z²
   
   # Point at infinity
   infinity = torch.tensor([1.0, 0.0, 0.0])  # z = 0
   ```

2. **Line Representation**:
   ```python
   # Line through two points
   line = uhg.join(point1, point2)  # Returns (l:m:n)
   
   # Null line (contains its pole)
   null_line = torch.tensor([1.0, 0.0, 1.0])  # l² + m² = n²
   ```

3. **Homogeneous Coordinates**:
   - Scale invariant: [x:y:z] ≡ [λx:λy:λz] for λ ≠ 0
   - Normalization preserves projective properties
   - Used for both points and lines in dual roles

### Projective Transformations
1. **Definition**: 
   Linear transformations that preserve:
   - Cross-ratios
   - Incidence relations
   - Projective invariants

2. **Properties**:
   - Form a group under composition
   - Preserve null points and lines
   - Maintain hyperbolic structure

3. **Matrix Form**:
   ```python
   # Generate random projective transformation
   matrix = uhg.get_projective_matrix(2)  # 2D projective matrix
   
   # Apply transformation to points
   transformed_points = uhg.transform(points, matrix)
   ```

## Chapter 4: Cross-ratios and Invariants

### Cross-ratio
1. **Definition**:
   For four points A, B, C, D:
   CR(A,B;C,D) = |AC|·|BD| / |AD|·|BC|
   where |PQ| is the determinant of points P and Q.

2. **Properties**:
   - Projectively invariant
   - Key to defining distance and angle
   - Fundamental to UHG constructions

3. **Example**:
   ```python
   # Calculate cross-ratio of four points
   cr = uhg.cross_ratio(A, B, C, D)
   
   # Special values:
   # cr = -1: C,D are midpoints of AB
   # cr = 1: C,D are perpendicular
   # cr = 0: Points are coincident
   ```

### Quadrance and Spread
1. **Quadrance**:
   - Measures squared distance between points
   - q(A,B) = 1 - <A,B>²/(<A,A><B,B>)
   ```python
   # Calculate quadrance between points
   q = uhg.quadrance(A, B)
   ```

2. **Spread**:
   - Measures squared angle between lines
   - S(l,m) = (l₁m₁ + l₂m₂ - l₃m₃)²/((l₁² + l₂² - l₃²)(m₁² + m₂² - m₃²))
   ```python
   # Calculate spread between lines
   s = uhg.spread(line1, line2)
   ```

## Chapter 5: Advanced Constructions

### Midpoint Construction
1. **Theory**:
   - Midpoints m₁,m₂ of side AB exist when p = 1-q is a square
   - After normalizing to equal hyperbolic norms:
     m₁ = [x₁+x₂ : y₁+y₂ : z₁+z₂]
     m₂ = [x₁-x₂ : y₁-y₂ : z₁-z₂]

2. **Properties**:
   - Equal quadrances to endpoints
   - Perpendicular to each other
   - Cross-ratio (A,B:m₁,m₂) = -1

3. **Edge Cases**:
   - Null points: Return the null point as midpoint
   - Points too far apart: No midpoints exist
   - Identical points: Return point as single midpoint

### Numerical Considerations
1. **Stability**:
   - Normalize points to prevent overflow/underflow
   - Handle near-null points carefully
   - Check discriminants before square roots

2. **Edge Cases**:
   - Points at infinity require special handling
   - Null lines may have undefined operations
   - Nearly coincident points need careful treatment

3. **Best Practices**:
   ```python
   # Always normalize points
   points = uhg.normalize_points(points)
   
   # Check for null points
   if uhg.is_null_point(point):
       # Handle null case
   
   # Use epsilon for comparisons
   if torch.allclose(a, b, rtol=1e-5):
       # Points are effectively equal
   ```

## Applications

### Geometric Constructions
1. **Perpendicular Lines**:
   ```python
   # Get perpendicular through point to line
   perp = uhg.wedge(point, line)
   ```

2. **Midpoint Finding**:
   ```python
   # Find midpoints of two points
   m1, m2 = uhg.midpoints(A, B)
   
   # Verify properties
   assert uhg.verify_midpoints(A, B, m1, m2)
   ```

3. **Circle Construction**:
   ```python
   # Define circle by center and radius
   center = torch.tensor([0.0, 0.0, 1.0])
   radius = 0.5
   
   # Points on circle satisfy:
   # quadrance(point, center) = radius²
   ```

### Cybersecurity Applications
1. **Network Topology**:
   - Model network structure in hyperbolic space
   - Use distance for routing decisions
   - Exploit projective invariance for scaling

2. **Threat Analysis**:
   - Map threats to hyperbolic points
   - Use spreads for angular correlation
   - Apply projective transformations for perspective changes

3. **Example: Threat Correlation**:
   ```python
   # Map threats to hyperbolic points
   threats = uhg.normalize_points(threat_vectors)
   
   # Calculate pairwise correlations
   correlations = uhg.quadrance(threats[:, None], threats[None, :])
   
   # Find clusters using hyperbolic distance
   clusters = hyperbolic_clustering(threats, max_distance=0.5)
   ```

## Important Notes and Warnings
[Insert any critical warnings or notes from UHG.pdf]

## Reference Formulas Quick Sheet
[Insert a quick reference of the most important formulas]
