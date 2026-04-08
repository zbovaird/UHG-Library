# Universal Hyperbolic Geometry (UHG) Functions Summary

This document provides a summary of the key functions in the UHG library (version 0.3.2) and their implementations.

## Core Geometric Functions

### Quadrance

**Mathematical Definition**: The quadrance between two points a=[x₁:y₁:z₁] and b=[x₂:y₂:z₂] is defined as:

```
q(a,b) = 1 - (x₁x₂ + y₁y₂ - z₁z₂)²/((x₁² + y₁² - z₁²)(x₂² + y₂² - z₂²))
```

**Implementation Details**:
- Normalizes input points
- Checks for null points (undefined for null points)
- Computes hyperbolic dot products for numerator and denominator
- Handles numerical stability for the denominator
- Returns 1 for perpendicular points (when hyperbolic dot product is 0)

### Spread

**Mathematical Definition**: The spread between two lines L1=[l₁:l₂:l₃] and L2=[m₁:m₂:m₃] is defined as:

```
S(L1,L2) = (l₁m₁ + l₂m₂ - l₃m₃)² / ((l₁² + l₂² - l₃²)(m₁² + m₂² - m₃²))
```

**Implementation Details**:
- Normalizes input lines
- Checks for null lines (undefined for null lines)
- Computes hyperbolic dot products for numerator and denominator
- Handles numerical stability for the denominator
- Returns 1 for perpendicular lines (when hyperbolic dot product is 0)

### Cross Ratio

**Mathematical Definition**: The cross-ratio of four vectors in projective space is defined as:

```
CR(A,B;C,D) = |x₁ y₁|  |x₁ y₁|
              |z₁ w₁|  |z₂ w₂|
              ─────── / ───────
              |x₂ y₂|  |x₂ y₂|
              |z₁ w₁|  |z₂ w₂|
```

For the special case where v1,v2 are used as basis:
```
CR(v₁,v₂:u₁,u₂) = w₁/w₂ / z₁/z₂ = w₁z₂/w₂z₁
```

**Implementation Details**:
- First attempts a special case where v1,v2 can be used as basis (more numerically stable)
- Falls back to general case using determinant form if special case fails
- Projects to 2D if needed by taking first two coordinates (preserves cross-ratio due to projective invariance)
- Computes the four 2x2 determinants and their ratio
- Handles numerical stability while preserving sign

## Supporting Functions

### Hyperbolic Dot Product

**Mathematical Definition**: For points [x₁:y₁:z₁] and [x₂:y₂:z₂]:
```
x₁x₂ + y₁y₂ - z₁z₂
```

**Implementation Details**:
- Splits inputs into spatial and time components
- Computes dot product with correct signature (+ for spatial, - for time)
- Handles both 3D and higher dimensional points

### Null Point/Line Detection

**Mathematical Definition**:
- A point [x:y:z] is null when x² + y² = z²
- A line (l:m:n) is null when l² + m² = n²

**Implementation Details**:
- Normalizes inputs for numerical stability
- Computes the norm x² + y² - z²
- Returns true if norm is close to 0 (within epsilon)

### Join and Meet Operations

**Join**: Computes the line through two points using cross product
**Meet**: Computes the intersection point of a line and a point using cross product

### Point Normalization

**Implementation Details**:
- For null points (x² + y² = z²), normalizes so z = 1
- For non-null points, normalizes so largest component is ±1
- Ensures first non-zero component is positive
- Handles both single points and batched tensors

## Geometric Theorems and Formulas

### Triple Quad Formula

**Mathematical Definition**:
```
(q₁ + q₂ + q₃)² = 2(q₁² + q₂² + q₃²) + 4q₁q₂q₃
```

### Triple Spread Formula

**Mathematical Definition**:
```
(S₁ + S₂ + S₃)² = 2(S₁² + S₂² + S₃²) + 4S₁S₂S₃
```

### Pythagoras Theorem

**Mathematical Definition**: For a right triangle (S₃ = 1):
```
q₃ = q₁ + q₂ - q₁q₂
```

### Dual Pythagoras Theorem

**Mathematical Definition**: For a right triangle (q₃ = 1):
```
S₃ = S₁ + S₂ - S₁S₂
```

### Cross Law

**Mathematical Definition**: Relates the three quadrances and three spreads of a triangle:
```
q₁q₂q₃S₁S₂S₃ = (q₁q₂S₃ + q₂q₃S₁ + q₃q₁S₂ - q₁ - q₂ - q₃ - S₁ - S₂ - S₃ + 2)²
``` 