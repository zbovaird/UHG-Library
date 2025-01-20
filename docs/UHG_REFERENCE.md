# Universal Hyperbolic Geometry Reference

## Chapter 3: Projective Geometry

### Core Concepts
[Insert core concepts about projective geometry from Chapter 3]

### Points and Lines
- Point Representation: [Insert how points are represented]
- Line Representation: [Insert how lines are represented]
- Homogeneous Coordinates: [Insert explanation of homogeneous coordinates]

### Projective Transformations
- Definition: [Insert mathematical definition]
- Properties: [Insert key properties]
- Matrix Form: [Insert matrix representation]

## Chapter 4: Cross-ratios and Invariants

### Null Points and Lines

#### Definition
A point a ≡ [x:y:z] is null precisely when it lies on its dual line:
```
x² + y² - z² = 0
```

#### Theorem 21: Parametrization of Null Points
Any null point can be represented in the form:
```
α(t:u) = [t² - u² : 2tu : t² + u²]
```
where [t:u] are homogeneous parameters.

#### Theorem 23: Join of Null Points
For two null points α(t₁:u₁) and α(t₂:u₂), their join is:
```
[t₁t₂ - u₁u₂ : t₁u₂ + t₂u₁ : t₁t₂ + u₁u₂]
```

#### Key Properties
1. **Null Circle Structure**:
   - Null points form the fundamental "null circle" that defines the geometry
   - Every line intersects this circle in at most two points

2. **Perpendicularity**:
   - If a₁ is null and a₂ distinct, then a₁a₂ is a null line ⟺ a₁ ⟂ a₂
   - This provides a key link between null points and perpendicularity

3. **Triangle Geometry**:
   - Null points are essential in triangle geometry
   - Special case: "nil triangles" have at least one null vertex
   - "Triply nil triangles" have all vertices null

### Cross-ratio Definition

#### Fundamental Concept
The cross-ratio is an affine quantity that extends to be projectively invariant. For four non-zero vectors v₁, v₂, u₁, u₂ lying in a two-dimensional subspace spanned by vectors p and q:

#### Mathematical Definition
Given:
- v₁ = x₁p + y₁q
- v₂ = x₂p + y₂q
- u₁ = z₁p + w₁q
- u₂ = z₂p + w₂q

The cross-ratio is defined as:

```
(v₁,v₂:u₁,u₂) ≡ (x₁w₁ - y₁z₁)(x₁w₂ - y₁z₂) / (x₂w₁ - y₂z₁)(x₂w₂ - y₂z₂)
```

Which can be expressed in determinant form:
```
|x₁ y₁|  |x₁ y₁|
|z₁ w₁|  |z₂ w₂|
─────── / ───────
|x₂ y₂|  |x₂ y₂|
|z₁ w₁|  |z₂ w₂|
```

#### Key Properties

1. **Basis Independence**: 
   - Under change of basis p = ap′ + bq′, q = cp′ + dq′
   - Cross-ratio remains unchanged (multiplied by det(ad-bc) in both numerator and denominator)

2. **Projective Invariance**:
   - Only depends on the central lines or (hyperbolic) points [v₁], [v₂], [u₁], [u₂]
   - We write: ([v₁],[v₂]:[u₁],[u₂]) ≡ (v₁,v₂:u₁,u₂)

3. **Fundamental Identity**:
   ```
   ([v₁],[v₂]:[u₁],[u₂]) + ([v₁],[u₁]:[v₂],[u₂]) = 1
   ```

#### Special Case Formula
When p = v₁ and q = v₂, with u₁ = z₁v₁ + w₁v₂, and u₂ = z₂v₁ + w₂v₂:
```
(v₁,v₂:u₁,u₂) = w₁/w₂ / z₁/z₂ = w₁z₂/w₂z₁ = (w₁/z₁)/(w₂/z₂)
```

#### Quadrance Cross-ratio Theorem
For a non-null, non-nil side a₁a₂ with opposite points:
- o₁ ≡ (a₁a₂)a₁⊥
- o₂ ≡ (a₁a₂)a₂⊥

The quadrance is related to the cross-ratio:
```
q(a₁,a₂) = (a₁,o₂:a₂,o₁)
```

#### Implementation Notes
1. The quadrance between points and spread between lines can be framed entirely within projective geometry
2. Uses quadratic form x² + y² - z²
3. This approach differs from classical real number treatment:
   - No requirement for line a₁a₂ to pass through null points
   - Immediately dualizes to spreads between lines
4. Works in vector space F³, with vector v ≡ (a,b,c) and associated point [v] = [a:b:c]

### Key Invariants
- List and define each invariant
- Explain their significance
- Include formulas

### Higher Dimensions
- How cross-ratios extend to higher dimensions
- Any special considerations
- Formulas for n-dimensional space

## Chapter 5: Fundamental Operations

### Basic Operations
1. Join Operation
   - Definition: [Insert definition]
   - Formula: [Insert formula]
   - Properties: [Insert properties]

2. Meet Operation
   - Definition: [Insert definition]
   - Formula: [Insert formula]
   - Properties: [Insert properties]

### Hyperbolic Operations
### Fundamental Measurements in UHG

#### Quadrance
The quadrance between two points is defined using cross-ratio and opposite points:

1. **Definition via Cross-ratio**:
   ```
   q(a₁,a₂) = (a₁,o₂:a₂,o₁)
   ```
   where o₁,o₂ are opposite points:
   ```
   o₁ = (a₁·a₂)a₁ - (a₁·a₁)a₂
   o₂ = (a₂·a₂)a₁ - (a₁·a₂)a₂
   ```

2. **Properties**:
   - Undefined when either point is null (x² + y² - z² = 0)
   - Equal to 1 when points are perpendicular
   - Equal to 0 when points are same or form a null line
   - Projectively invariant

3. **Alternative Form**:
   ```
   q(a₁,a₂) = 1 - (a₁·a₂)²/((a₁·a₁)(a₂·a₂))
   ```
   where (·) is the hyperbolic dot product

#### Spread
The spread between two lines is the dual concept to quadrance:

1. **Definition**:
   ```
   S(l,m) = (l₁m₁ + l₂m₂ - l₃m₃)² / ((l₁² + l₂² - l₃²)(m₁² + m₂² - m₃²))
   ```
   where l = [l₁:l₂:l₃] and m = [m₁:m₂:m₃] are lines in projective form

2. **Properties**:
   - Undefined for null lines
   - Equal to 1 when lines are perpendicular
   - Equal to 0 when lines are parallel or coincident
   - Projectively invariant
   - Dual to quadrance under projective duality

### Important Note on Distance and Angle
UHG does not use classical notions of distance and angle. Instead:
- Quadrance replaces the concept of squared distance
- Spread replaces the concept of squared sine of angle
- All measurements are purely projective and algebraic
- No transcendental functions (like sin, cos, log) are needed
- All computations are rational functions in the coordinates

### Numerical Considerations
- Handling Special Cases: [Insert guidelines]
- Numerical Stability: [Insert recommendations]
- Error Bounds: [Insert acceptable tolerances]

## Implementation Guidelines

### Core Requirements
1. Cross-ratio Preservation
   - Exact conditions: [Insert conditions]
   - Error tolerances: [Insert tolerances]

2. Projective Structure
   - What must be preserved: [Insert requirements]
   - How to verify: [Insert verification methods]

### Neural Network Extensions
1. Layer Operations
   - How to extend to neural networks: [Insert guidelines]
   - Required properties: [Insert properties]

2. Attention Mechanisms
   - How to compute attention scores: [Insert method]
   - Cross-ratio preservation requirements: [Insert requirements]

3. Aggregation Operations
   - How to combine points: [Insert method]
   - Preservation of structure: [Insert requirements]

## Important Notes and Warnings
[Insert any critical warnings or notes from UHG.pdf]

## Reference Formulas Quick Sheet
[Insert a quick reference of the most important formulas]
