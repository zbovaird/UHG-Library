"""UHG-based threat indicator processing and correlation."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from uhg.projective import ProjectiveUHG
from uhg.attention import UHGMultiHeadAttention, UHGAttentionConfig

class ThreatIndicatorType:
    """Types of threat indicators supported."""
    NETWORK = "network"  # Network-based indicators (IPs, domains, etc.)
    SYSTEM = "system"    # System-based indicators (files, registry, etc.)
    BEHAVIOR = "behavior"  # Behavioral indicators (process patterns, etc.)
    PAYLOAD = "payload"   # Payload-based indicators (signatures, etc.)

class ThreatIndicator:
    """Representation of a threat indicator in projective space."""
    
    def __init__(
        self,
        indicator_type: str,
        value: str,
        confidence: float,
        context: Dict[str, float],
        feature_dim: int = 8
    ):
        """Initialize threat indicator.
        
        Args:
            indicator_type: Type of indicator (network, system, etc.)
            value: The indicator value (e.g. IP, hash, etc.)
            confidence: Confidence score [0,1]
            context: Additional context features
            feature_dim: Dimension of feature space
        """
        self.type = indicator_type
        self.value = value
        self.confidence = confidence
        self.context = context
        self.feature_dim = feature_dim
        
    def to_projective(self, uhg: ProjectiveUHG) -> torch.Tensor:
        """Convert indicator to projective coordinates."""
        # Create feature vector
        features = torch.zeros(self.feature_dim + 1)  # +1 for homogeneous coordinate
        
        # Encode type using projective coordinates
        if self.type == ThreatIndicatorType.NETWORK:
            features[0] = 1.0
        elif self.type == ThreatIndicatorType.SYSTEM:
            features[1] = 1.0
        elif self.type == ThreatIndicatorType.BEHAVIOR:
            features[2] = 1.0
        elif self.type == ThreatIndicatorType.PAYLOAD:
            features[3] = 1.0
            
        # Encode confidence
        features[4] = self.confidence
        
        # Encode context features
        for i, value in enumerate(self.context.values()):
            if i + 5 < self.feature_dim:
                features[i + 5] = value
                
        # Set homogeneous coordinate
        features[-1] = 1.0
        
        # Normalize using cross-ratio
        e1 = torch.zeros_like(features)
        e1[0] = 1.0
        e1[-1] = 1.0
        
        e2 = torch.zeros_like(features)
        e2[1] = 1.0
        e2[-1] = 1.0
        
        e3 = torch.zeros_like(features)
        e3[2] = 1.0
        e3[-1] = 1.0
        
        cr = uhg.cross_ratio(features, e1, e2, e3)
        features = features / (cr + 1e-6)
        
        return features

class ThreatCorrelation:
    """Correlate threat indicators using UHG attention."""
    
    def __init__(
        self,
        feature_dim: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """Initialize threat correlation.
        
        Args:
            feature_dim: Dimension of feature space
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        self.uhg = ProjectiveUHG()
        self.feature_dim = feature_dim
        
        # Initialize attention
        config = UHGAttentionConfig(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_cross_ratio=True
        )
        self.attention = UHGMultiHeadAttention(config)
        
    def correlate(
        self,
        indicators: List[ThreatIndicator],
        query_indicators: Optional[List[ThreatIndicator]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Correlate threat indicators using UHG attention.
        
        Args:
            indicators: List of threat indicators to correlate
            query_indicators: Optional query indicators (if None, use indicators)
            
        Returns:
            Tuple of (correlation matrix, attention weights)
        """
        # Convert indicators to projective coordinates
        proj_indicators = torch.stack([
            ind.to_projective(self.uhg) for ind in indicators
        ])
        
        # Add batch dimension
        proj_indicators = proj_indicators.unsqueeze(0)
        
        if query_indicators is None:
            query_indicators = proj_indicators
        else:
            query_indicators = torch.stack([
                ind.to_projective(self.uhg) for ind in query_indicators
            ]).unsqueeze(0)
            
        # Compute correlations using attention
        correlations, weights = self.attention(
            query_indicators,
            proj_indicators,
            proj_indicators
        )
        
        return correlations.squeeze(0), weights.squeeze(0)
    
    def get_correlation_groups(
        self,
        indicators: List[ThreatIndicator],
        threshold: float = 0.7
    ) -> List[List[ThreatIndicator]]:
        """Group correlated threat indicators.
        
        Args:
            indicators: List of threat indicators to group
            threshold: Correlation threshold for grouping
            
        Returns:
            List of correlated indicator groups
        """
        # Get correlation matrix
        correlations, _ = self.correlate(indicators)
        
        # Convert to adjacency matrix
        adj_matrix = (correlations > threshold).float()
        
        # Find connected components (groups)
        groups = []
        visited = set()
        
        for i in range(len(indicators)):
            if i in visited:
                continue
                
            # Find connected indicators
            group = []
            stack = [i]
            
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    group.append(indicators[node])
                    
                    # Add connected indicators
                    for j in range(len(indicators)):
                        if adj_matrix[node, j] > 0 and j not in visited:
                            stack.append(j)
                            
            groups.append(group)
            
        return groups
    
    def analyze_indicator_relationships(
        self,
        indicators: List[ThreatIndicator]
    ) -> Dict[str, List[Tuple[ThreatIndicator, ThreatIndicator, float]]]:
        """Analyze relationships between different types of indicators.
        
        Args:
            indicators: List of threat indicators to analyze
            
        Returns:
            Dictionary mapping relationship types to indicator pairs and scores
        """
        correlations, weights = self.correlate(indicators)
        
        # Initialize relationship types
        relationships = {
            "network_system": [],    # Network-system relationships
            "network_behavior": [],  # Network-behavior relationships
            "system_behavior": [],   # System-behavior relationships
            "payload_related": []    # Payload-related relationships
        }
        
        # Analyze correlations
        for i in range(len(indicators)):
            for j in range(i + 1, len(indicators)):
                score = correlations[i, j].item()
                pair = (indicators[i], indicators[j], score)
                
                # Categorize relationship
                if indicators[i].type == ThreatIndicatorType.NETWORK:
                    if indicators[j].type == ThreatIndicatorType.SYSTEM:
                        relationships["network_system"].append(pair)
                    elif indicators[j].type == ThreatIndicatorType.BEHAVIOR:
                        relationships["network_behavior"].append(pair)
                elif indicators[i].type == ThreatIndicatorType.SYSTEM:
                    if indicators[j].type == ThreatIndicatorType.BEHAVIOR:
                        relationships["system_behavior"].append(pair)
                
                # Check payload relationships
                if (indicators[i].type == ThreatIndicatorType.PAYLOAD or
                    indicators[j].type == ThreatIndicatorType.PAYLOAD):
                    relationships["payload_related"].append(pair)
                    
        return relationships 