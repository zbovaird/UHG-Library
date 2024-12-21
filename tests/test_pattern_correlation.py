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
    """Test pattern encoding into hyperbolic space."""
    access_pattern = {
        "strength": 0.8,
        "from_level": "user",
        "to_level": "admin"
    }
    
    features = correlator._encode_pattern(access_pattern, "access")
    
    assert isinstance(features, torch.Tensor)
    assert features.shape == (correlator.feature_dim,)
    assert features[0] == 0.8  # Strength should be directly encoded
    assert torch.sum(features > 0) > 0  # Should have some non-zero features

def test_pattern_correlation(correlator):
    """Test pattern correlation across dimensions."""
    (
        access_patterns,
        permission_patterns,
        temporal_patterns,
        relationship_patterns
    ) = create_test_patterns()
    
    correlated_patterns = correlator.correlate_patterns(
        access_patterns,
        permission_patterns,
        temporal_patterns,
        relationship_patterns
    )
    
    assert isinstance(correlated_patterns, list)
    assert all(isinstance(p, CorrelationPattern) for p in correlated_patterns)
    
    # High-risk patterns should be detected
    high_risk_patterns = [p for p in correlated_patterns if p.risk_score > 0.7]
    assert len(high_risk_patterns) > 0
    
    # Check pattern components
    for pattern in correlated_patterns:
        assert "access" in pattern.components
        assert "permission" in pattern.components
        assert "temporal" in pattern.components
        assert len(pattern.related_patterns) == 3

def test_risk_score_computation(correlator):
    """Test risk score computation for correlated patterns."""
    access_pattern = {
        "type": "escalation",
        "strength": 0.8
    }
    
    perm_pattern = {
        "type": "unusual_combination",
        "strength": 0.7
    }
    
    temp_pattern = {
        "strength": 0.9,
        "time_span": 30.0  # 30 seconds
    }
    
    risk_score = correlator._compute_risk_score(
        access_pattern,
        perm_pattern,
        temp_pattern
    )
    
    assert 0.0 <= risk_score <= 1.0
    # High risk due to escalation, unusual permissions, and rapid sequence
    assert risk_score > 0.7

def test_pattern_evolution_analysis(correlator):
    """Test pattern evolution analysis."""
    # Create a sequence of patterns over time
    patterns = [
        CorrelationPattern(
            pattern_type="complex_violation",
            strength=0.7,
            components={
                "access": {"from_level": "user", "to_level": "viewer"},
                "permission": {"permissions": {"read", "write"}},
                "temporal": {"time_span": 300.0}
            },
            temporal_span=(1000.0, 1300.0),
            risk_score=0.6,
            related_patterns=["a1", "p1", "t1"]
        ),
        CorrelationPattern(
            pattern_type="complex_violation",
            strength=0.8,
            components={
                "access": {"from_level": "user", "to_level": "admin"},
                "permission": {"permissions": {"read", "write", "execute"}},
                "temporal": {"time_span": 60.0}
            },
            temporal_span=(1400.0, 1460.0),
            risk_score=0.8,
            related_patterns=["a2", "p2", "t2"]
        ),
        CorrelationPattern(
            pattern_type="complex_violation",
            strength=0.9,
            components={
                "access": {"from_level": "user", "to_level": "admin"},
                "permission": {"permissions": {"read", "write", "execute", "delete"}},
                "temporal": {"time_span": 30.0}
            },
            temporal_span=(1500.0, 1530.0),
            risk_score=0.9,
            related_patterns=["a3", "p3", "t3"]
        )
    ]
    
    evolution = correlator.analyze_pattern_evolution(patterns, time_window=3600)
    
    assert isinstance(evolution, dict)
    assert all(k in evolution for k in [
        "escalating", "persistent", "cascading", "cyclical"
    ])
    
    # Should detect escalating pattern
    assert len(evolution["escalating"]) > 0
    
    # Should detect cascading pattern (3 patterns in sequence)
    assert len(evolution["cascading"]) > 0

def test_pattern_similarity(correlator):
    """Test pattern similarity comparison."""
    p1 = CorrelationPattern(
        pattern_type="complex_violation",
        strength=0.8,
        components={
            "access": {"from_level": "user", "to_level": "admin"},
            "permission": {"permissions": {"read", "write"}},
            "temporal": {"time_span": 60.0}
        },
        temporal_span=(1000.0, 1060.0),
        risk_score=0.7,
        related_patterns=["a1", "p1", "t1"]
    )
    
    # Similar pattern
    p2 = CorrelationPattern(
        pattern_type="complex_violation",
        strength=0.75,
        components={
            "access": {"from_level": "user", "to_level": "admin"},
            "permission": {"permissions": {"read", "write"}},
            "temporal": {"time_span": 90.0}
        },
        temporal_span=(2000.0, 2090.0),
        risk_score=0.65,
        related_patterns=["a2", "p2", "t2"]
    )
    
    # Different pattern
    p3 = CorrelationPattern(
        pattern_type="complex_violation",
        strength=0.6,
        components={
            "access": {"from_level": "user", "to_level": "viewer"},
            "permission": {"permissions": {"read"}},
            "temporal": {"time_span": 300.0}
        },
        temporal_span=(3000.0, 3300.0),
        risk_score=0.5,
        related_patterns=["a3", "p3", "t3"]
    )
    
    assert correlator._are_patterns_similar(p1, p2)
    assert not correlator._are_patterns_similar(p1, p3)
    assert not correlator._are_patterns_similar(p2, p3) 