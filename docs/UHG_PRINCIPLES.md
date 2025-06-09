# Universal Hyperbolic Geometry (UHG) Library Principles

## Core Purpose
The UHG library implements machine learning operations in hyperbolic space using **purely projective geometry** techniques, adhering to Universal Hyperbolic Geometry principles. This approach avoids differential geometry entirely, offering a more fundamental and computationally elegant way to handle hyperbolic spaces.

## Key Principles

### 1. Pure Projective Geometry
- All operations MUST be implemented using projective geometry only
- NO differential geometry concepts or operations allowed
- All points are represented in projective coordinates (homogeneous coordinates)
- Cross-ratio preservation is fundamental and must be maintained through all transformations

### 2. Core Operations
- All transformations must preserve the projective structure
- Key operations include:
  - Projective transformations
  - Cross-ratio calculations
  - Point normalization
  - Distance calculations via cross-ratio
  - Angle calculations via cross-ratio
  - Projective joins and meets

### 3. Neural Network Integration
- All neural network operations must be reformulated in projective terms
- Common operations to support:
  - Linear transformations → Projective transformations
  - Activation functions → Cross-ratio preserving operations
  - Attention mechanisms → Cross-ratio based similarity
  - Aggregation operations → Projective averaging
  - Dropout → Projective dropout preserving structure

### 4. Architecture Independence
- The library must be usable with any neural network architecture
- Core operations should be architecture-agnostic
- Implementation examples (e.g., GraphSAGE) serve as references but not limitations
- Easy integration with popular deep learning frameworks

### 5. Computational Requirements
- Maintain numerical stability in projective calculations
- Ensure proper handling of homogeneous coordinates
- Preserve cross-ratios within acceptable numerical tolerance
- Efficient implementation of core operations

### 6. Implementation Guidelines

#### Point Representation
- Points must always include homogeneous coordinate
- Format: [x₁, x₂, ..., xₙ, w] where w is the homogeneous coordinate
- Normalization should maintain projective structure

#### Transformations
- Must preserve cross-ratios
- Should handle both finite and ideal points correctly
- Must maintain projective invariants

#### Neural Network Layers
- Input/output must maintain projective structure
- Layer operations must preserve geometric meaning
- Should support both Euclidean and hyperbolic architectures

### 7. Testing Requirements
- Verify cross-ratio preservation
- Check projective invariants
- Ensure numerical stability
- Test with both finite and ideal points
- Verify geometric properties preservation

## Implementation Priorities

1. Core Projective Operations
   - Cross-ratio calculations
   - Projective transformations
   - Point normalization
   - Distance metrics

2. Basic Neural Operations
   - Projective linear transformations
   - Structure-preserving activation functions
   - Projective aggregation methods

3. Advanced Neural Components
   - Attention mechanisms
   - Convolution operations
   - Pooling layers
   - Dropout implementations

4. Architecture-Specific Implementations
   - Graph neural networks
   - Feedforward networks
   - Recurrent networks
   - Transformer-style architectures

## Code Organization

### Core Components
```
uhg/
├── projective.py      # Core projective geometry operations
├── transforms.py      # Projective transformations
├── metrics.py        # Distance and similarity measures
└── nn/
    ├── layers/       # Basic neural network layers
    ├── models/       # Complete model implementations
    └── functional/   # Neural network operations
```

## Quality Standards
1. All operations must maintain projective structure
2. Cross-ratios must be preserved within numerical tolerance
3. Operations must be numerically stable
4. Code must be well-documented with mathematical foundations
5. Tests must verify geometric properties
6. Performance optimizations must not compromise geometric accuracy
