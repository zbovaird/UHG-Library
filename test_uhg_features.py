"""
Tests for UHG Feature Extraction module.
"""

import torch
import numpy as np
import pytest
from uhg_features import UHGFeatureExtractor, UHGFeatureConfig, create_feature_extractor

def test_feature_extractor_initialization():
    """Test proper initialization of feature extractor."""
    config = UHGFeatureConfig(input_dim=10, output_dim=5)
    extractor = UHGFeatureExtractor(config)
    
    assert extractor.config.input_dim == 10
    assert extractor.config.output_dim == 5
    assert extractor.config.preserve_cross_ratio == True
    assert extractor.config.use_homogeneous_coords == True

def test_homogeneous_coordinate_addition():
    """Test addition of homogeneous coordinates."""
    config = UHGFeatureConfig(input_dim=3, output_dim=2)
    extractor = UHGFeatureExtractor(config)
    
    x = torch.randn(5, 3)  # 5 samples, 3 dimensions
    x_homogeneous = extractor.add_homogeneous_coordinate(x)
    
    assert x_homogeneous.shape == (5, 4)
    assert torch.all(x_homogeneous[:, -1] == 1)

def test_cross_ratio_preservation():
    """Test if cross-ratio is preserved in feature extraction."""
    config = UHGFeatureConfig(input_dim=3, output_dim=2)
    extractor = UHGFeatureExtractor(config)
    
    # Create four points in general position
    p1 = torch.tensor([[1.0, 0.0, 0.0]])
    p2 = torch.tensor([[0.0, 1.0, 0.0]])
    p3 = torch.tensor([[0.0, 0.0, 1.0]])
    p4 = torch.tensor([[1.0, 1.0, 1.0]])
    
    # Extract features
    f1 = extractor.compute_uhg_features(p1)
    f2 = extractor.compute_uhg_features(p2)
    f3 = extractor.compute_uhg_features(p3)
    f4 = extractor.compute_uhg_features(p4)
    
    # Compute cross-ratios in original and feature space
    cr_original = extractor._compute_cross_ratio(
        extractor.add_homogeneous_coordinate(p1),
        extractor.add_homogeneous_coordinate(p2),
        extractor.add_homogeneous_coordinate(p3),
        extractor.add_homogeneous_coordinate(p4)
    )
    
    cr_features = extractor._compute_cross_ratio(
        extractor.add_homogeneous_coordinate(f1),
        extractor.add_homogeneous_coordinate(f2),
        extractor.add_homogeneous_coordinate(f3),
        extractor.add_homogeneous_coordinate(f4)
    )
    
    assert torch.allclose(cr_original, cr_features, rtol=1e-5)

def test_feature_extraction_output_shape():
    """Test if feature extraction produces correct output shapes."""
    input_dim = 10
    output_dim = 5
    batch_size = 32
    
    extractor = create_feature_extractor(input_dim, output_dim)
    data = torch.randn(batch_size, input_dim)
    
    results = extractor.extract_features(data)
    features = results['features']
    
    assert features.shape == (batch_size, output_dim)

def test_intermediate_results():
    """Test if intermediate results are correctly computed and returned."""
    input_dim = 5
    output_dim = 3
    batch_size = 16
    
    extractor = create_feature_extractor(input_dim, output_dim)
    data = torch.randn(batch_size, input_dim)
    
    results = extractor.extract_features(data, return_intermediate=True)
    
    assert 'features' in results
    assert 'homogeneous' in results
    assert 'cross_ratio_features' in results
    assert 'geometric_features' in results
    
    assert results['homogeneous'].shape == (batch_size, input_dim + 1)

def test_geometric_feature_computation():
    """Test computation of geometric features."""
    config = UHGFeatureConfig(input_dim=3, output_dim=2)
    extractor = UHGFeatureExtractor(config)
    
    x = torch.randn(5, 3)
    x_homogeneous = extractor.add_homogeneous_coordinate(x)
    
    geom_features = extractor._compute_geometric_features(x_homogeneous)
    
    # Should have features for each axis plus invariants
    assert len(geom_features.shape) == 2
    assert geom_features.shape[0] == 5  # batch size

def test_feature_extractor_with_real_data():
    """Test feature extraction with realistic network data."""
    # Simulate network traffic features
    batch_size = 64
    input_features = 10  # e.g., packet size, timing, ports, etc.
    output_features = 5
    
    # Create sample data with more realistic patterns
    data = torch.rand(batch_size, input_features)
    
    # Create more distinct patterns
    # Pattern 1: High values in first two features
    data[:10, 0] = 0.9 + 0.1 * torch.rand(10)
    data[:10, 1] = 0.8 + 0.1 * torch.rand(10)
    data[:10, 2:] *= 0.3  # Lower other values
    
    # Pattern 2: High values in features 3-4, low elsewhere
    data[10:20, 2] = 0.9 + 0.1 * torch.rand(10)
    data[10:20, 3] = 0.8 + 0.1 * torch.rand(10)
    data[10:20, :2] *= 0.3
    data[10:20, 4:] *= 0.3
    
    extractor = create_feature_extractor(
        input_dim=input_features,
        output_dim=output_features
    )
    
    results = extractor.extract_features(data)
    features = results['features']
    
    # Check if similar input patterns produce similar features
    pattern1_features = features[:10]
    pattern2_features = features[10:20]
    
    # Compute within-pattern and between-pattern distances
    within_pattern1 = torch.pdist(pattern1_features).mean()
    within_pattern2 = torch.pdist(pattern2_features).mean()
    between_patterns = torch.cdist(pattern1_features, pattern2_features).mean()
    
    # Within-pattern distances should be smaller than between-pattern distances
    assert within_pattern1 < between_patterns
    assert within_pattern2 < between_patterns

if __name__ == "__main__":
    pytest.main([__file__]) 