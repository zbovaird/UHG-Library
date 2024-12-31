"""Tests for pattern correlation module."""

import pytest
import torch
import numpy as np
from uhg.patterns.correlation import PatternCorrelator, CorrelationPattern

def create_test_patterns():
    """Create test patterns for correlation analysis."""
    access_patterns = [
        {
            "id": "a1",
            "type": "escalation",
            "strength": 0.8,
            "from_level": "user",
            "to_level": "admin"
        },
        {
            "id": "a2",
            "type": "normal",
            "strength": 0.4,
            "from_level": "user",
            "to_level": "viewer"
        }
    ]
    
    permission_patterns = [
        {
            "id": "p1",
            "type": "unusual_combination",
            "strength": 0.7,
            "permissions": {"read", "write", "execute", "delete"}
        },
        {
            "id": "p2",
            "type": "normal",
            "strength": 0.3,
            "permissions": {"read", "write"}
        }
    ]
    
    temporal_patterns = [
        {
            "id": "t1",
            "type": "rapid",
            "strength": 0.9,
            "time_span": 30.0,  # 30 seconds
            "start_time": 1000.0,
            "end_time": 1030.0
        },
        {
            "id": "t2",
            "type": "normal",
            "strength": 0.4,
            "time_span": 600.0,  # 10 minutes
            "start_time": 2000.0,
            "end_time": 2600.0
        }
    ]
    
    relationship_patterns = [
        {
            "id": "r1",
            "type": "close",
            "strength": 0.6,
            "distance": 0.2
        },
        {
            "id": "r2",
            "type": "distant",
            "strength": 0.3,
            "distance": 0.8
        }
    ]
    
    return (
        access_patterns,
        permission_patterns,
        temporal_patterns,
        relationship_patterns
    )

@pytest.fixture
def correlator():
    """Initialize pattern correlator."""
    return PatternCorrelator(
        feature_dim=64,
        num_heads=4,
        correlation_threshold=0.7
    )

def test_pattern_encoding(correlator):
    """Test basic pattern encoding functionality."""
    patterns = create_test_patterns()[0]  # Only test access patterns
    
    encoded = correlator.encode_patterns(patterns, pattern_type="access")
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dim() == 2  # [num_patterns, embedding_dim]
    assert encoded.size(0) == len(patterns)  # Number of patterns
    assert encoded.size(1) == correlator.feature_dim  # Embedding dimension

def test_risk_score_computation(correlator):
    """Test risk score computation."""
    pattern = CorrelationPattern(
        pattern_type="access_violation",
        strength=0.8,
        components={
            "access": {"from_level": "user", "to_level": "admin"},
            "permission": {"permissions": {"read", "write", "execute"}},
            "temporal": {"time_span": 60.0}
        },
        temporal_span=(1000.0, 1060.0),
        risk_score=None,  # Will be computed
        related_patterns=["a1", "p1", "t1"]
    )
    
    score = correlator.compute_risk_score(pattern)
    assert 0.0 <= score <= 1.0

def test_pattern_similarity(correlator):
    """Test pattern similarity computation."""
    pattern1 = CorrelationPattern(
        pattern_type="access_violation",
        strength=0.8,
        components={
            "access": {"from_level": "user", "to_level": "admin"},
            "permission": {"permissions": {"read", "write"}},
            "temporal": {"time_span": 300.0}
        },
        temporal_span=(1000.0, 1300.0),
        risk_score=0.7,
        related_patterns=["a1", "p1", "t1"]
    )
    
    pattern2 = CorrelationPattern(
        pattern_type="access_violation",
        strength=0.7,
        components={
            "access": {"from_level": "user", "to_level": "admin"},
            "permission": {"permissions": {"read"}},
            "temporal": {"time_span": 200.0}
        },
        temporal_span=(2000.0, 2200.0),
        risk_score=0.6,
        related_patterns=["a2", "p2", "t2"]
    )
    
    similarity = correlator.compute_pattern_similarity(pattern1, pattern2)
    assert 0.0 <= similarity <= 1.0 