"""
Advanced Pattern Correlation in Hyperbolic Space.

This module implements advanced pattern correlation techniques using UHG principles,
focusing on detecting complex relationships between different types of patterns
in authorization hierarchies.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from ..projective import ProjectiveUHG

@dataclass
class CorrelationPattern:
    """Represents a correlated pattern across different dimensions."""
    pattern_type: str  # Type of correlation pattern
    strength: float  # Correlation strength
    components: Dict[str, any]  # Component patterns
    temporal_span: Tuple[float, float]  # Time span of pattern
    risk_score: float  # Risk assessment score
    related_patterns: List[str]  # IDs of related patterns

class PatternCorrelator:
    """Correlates patterns across different dimensions in hyperbolic space."""
    
    def __init__(
        self,
        feature_dim: int = 64,
        num_heads: int = 4,
        correlation_threshold: float = 0.7,
        device: Optional[torch.device] = None
    ):
        self.feature_dim = feature_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.uhg = ProjectiveUHG()
        self.correlation_threshold = correlation_threshold
        
    def _encode_pattern(
        self,
        pattern: Dict[str, any],
        pattern_type: str
    ) -> torch.Tensor:
        """Encode a pattern into hyperbolic space."""
        features = torch.zeros(self.feature_dim, device=self.device)
        
        if pattern_type == "access":
            # Encode access pattern features
            features[0] = float(pattern.get("strength", 0.0))
            features[1] = hash(pattern.get("from_level", "")) % (self.feature_dim // 4)
            features[2] = hash(pattern.get("to_level", "")) % (self.feature_dim // 4)
            
        elif pattern_type == "permission":
            # Encode permission pattern features
            features[0] = float(pattern.get("strength", 0.0))
            perms = pattern.get("permissions", set())
            for i, perm in enumerate(sorted(perms)):
                if i < self.feature_dim // 4:
                    features[3 + i] = 1.0
                    
        elif pattern_type == "temporal":
            # Encode temporal pattern features
            features[0] = float(pattern.get("strength", 0.0))
            features[1] = float(pattern.get("time_span", 0.0))
            
        elif pattern_type == "relationship":
            # Encode relationship pattern features
            features[0] = float(pattern.get("strength", 0.0))
            features[1] = float(pattern.get("distance", 0.0))
            
        return features
        
    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention scores."""
        # Compute attention scores using dot product
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale scores
        scores = scores / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
        
    def correlate_patterns(
        self,
        access_patterns: List[Dict[str, any]],
        permission_patterns: List[Dict[str, any]],
        temporal_patterns: List[Dict[str, any]],
        relationship_patterns: List[Dict[str, any]]
    ) -> List[CorrelationPattern]:
        """
        Correlate patterns across different dimensions.
        
        Args:
            access_patterns: List of access patterns
            permission_patterns: List of permission patterns
            temporal_patterns: List of temporal patterns
            relationship_patterns: List of relationship patterns
            
        Returns:
            List of correlated patterns
        """
        # Handle empty pattern lists
        if not access_patterns or not permission_patterns or not temporal_patterns:
            return []
            
        # Encode patterns into tensors
        access_features = torch.stack([
            self._encode_pattern(p, "access") for p in access_patterns
        ]).unsqueeze(0).to(self.device)  # Add batch dimension
        
        perm_features = torch.stack([
            self._encode_pattern(p, "permission") for p in permission_patterns
        ]).unsqueeze(0).to(self.device)
        
        temp_features = torch.stack([
            self._encode_pattern(p, "temporal") for p in temporal_patterns
        ]).unsqueeze(0).to(self.device)
        
        # Only process relationship patterns if they exist
        if relationship_patterns:
            rel_features = torch.stack([
                self._encode_pattern(p, "relationship") for p in relationship_patterns
            ]).unsqueeze(0).to(self.device)
        else:
            rel_features = None
        
        # Compute cross-attention between pattern types
        correlated_patterns = []
        
        # Access-Permission correlations
        _, access_perm_attn = self._compute_attention(
            access_features, perm_features, perm_features
        )
        access_perm_attn = access_perm_attn.squeeze(0)
        
        # Access-Temporal correlations
        _, access_temp_attn = self._compute_attention(
            access_features, temp_features, temp_features
        )
        access_temp_attn = access_temp_attn.squeeze(0)
        
        # Permission-Temporal correlations
        _, perm_temp_attn = self._compute_attention(
            perm_features, temp_features, temp_features
        )
        perm_temp_attn = perm_temp_attn.squeeze(0)
        
        # Find significant correlations
        for i, access_pattern in enumerate(access_patterns):
            for j, perm_pattern in enumerate(permission_patterns):
                for k, temp_pattern in enumerate(temporal_patterns):
                    # Compute correlation strength using softmax
                    correlation_strength = F.softmax(
                        torch.tensor([
                            access_perm_attn[i, j],
                            access_temp_attn[i, k],
                            perm_temp_attn[j, k]
                        ], device=self.device),
                        dim=0
                    ).mean().item()
                    
                    if correlation_strength > self.correlation_threshold:
                        # Create correlated pattern
                        pattern = CorrelationPattern(
                            pattern_type="complex_violation",
                            strength=correlation_strength,
                            components={
                                "access": access_pattern,
                                "permission": perm_pattern,
                                "temporal": temp_pattern
                            },
                            temporal_span=(
                                temp_pattern.get("start_time", 0),
                                temp_pattern.get("end_time", 0)
                            ),
                            risk_score=self._compute_risk_score(
                                access_pattern,
                                perm_pattern,
                                temp_pattern
                            ),
                            related_patterns=[
                                access_pattern.get("id", ""),
                                perm_pattern.get("id", ""),
                                temp_pattern.get("id", "")
                            ]
                        )
                        correlated_patterns.append(pattern)
                        
        return correlated_patterns
        
    def _compute_risk_score(
        self,
        access_pattern: Dict[str, any],
        perm_pattern: Dict[str, any],
        temp_pattern: Dict[str, any]
    ) -> float:
        """Compute risk score for a correlated pattern."""
        # Base risk from pattern strengths
        risk = (
            float(access_pattern.get("strength", 0.0)) +
            float(perm_pattern.get("strength", 0.0)) +
            float(temp_pattern.get("strength", 0.0))
        ) / 3.0
        
        # Increase risk for rapid sequences
        time_span = float(temp_pattern.get("time_span", float("inf")))
        if time_span < 60:  # Less than 1 minute
            risk *= 1.5
        elif time_span < 300:  # Less than 5 minutes
            risk *= 1.2
            
        # Increase risk for privilege escalation
        if access_pattern.get("type") == "escalation":
            risk *= 1.3
            
        # Increase risk for unusual permission combinations
        if perm_pattern.get("type") == "unusual_combination":
            risk *= 1.2
            
        return min(risk, 1.0)  # Cap risk at 1.0
        
    def analyze_pattern_evolution(
        self,
        patterns: List[CorrelationPattern],
        time_window: float = 3600  # 1 hour
    ) -> Dict[str, List[CorrelationPattern]]:
        """
        Analyze how patterns evolve over time.
        
        Args:
            patterns: List of correlated patterns
            time_window: Time window for evolution analysis
            
        Returns:
            Dictionary mapping evolution types to pattern sequences
        """
        # Sort patterns by time
        sorted_patterns = sorted(
            patterns,
            key=lambda p: p.temporal_span[0]
        )
        
        evolution_types = {
            "escalating": [],  # Patterns showing increasing severity
            "persistent": [],  # Patterns that persist over time
            "cascading": [],  # Patterns that trigger other patterns
            "cyclical": []    # Patterns that repeat periodically
        }
        
        # Analyze pattern evolution
        for i, pattern in enumerate(sorted_patterns[:-1]):
            # Look at subsequent patterns within time window
            next_patterns = [
                p for p in sorted_patterns[i+1:]
                if p.temporal_span[0] - pattern.temporal_span[1] <= time_window
            ]
            
            if not next_patterns:
                continue
                
            # Check for escalation
            if any(p.risk_score > pattern.risk_score * 1.2 for p in next_patterns):
                evolution_types["escalating"].append(pattern)
                
            # Check for persistence
            if any(
                self._are_patterns_similar(pattern, p) and
                p.temporal_span[0] - pattern.temporal_span[1] < 300  # 5 minutes
                for p in next_patterns
            ):
                evolution_types["persistent"].append(pattern)
                
            # Check for cascading effects
            if len(next_patterns) >= 3:  # Multiple patterns triggered
                evolution_types["cascading"].append(pattern)
                
            # Check for cyclical patterns
            if any(
                self._are_patterns_similar(pattern, p) and
                abs(p.temporal_span[0] - pattern.temporal_span[0] - 3600) < 300  # ~1 hour cycle
                for p in next_patterns
            ):
                evolution_types["cyclical"].append(pattern)
                
        return evolution_types
        
    def _are_patterns_similar(
        self,
        p1: CorrelationPattern,
        p2: CorrelationPattern
    ) -> bool:
        """Check if two patterns are similar."""
        # Compare components
        access_similar = (
            p1.components["access"].get("from_level") ==
            p2.components["access"].get("from_level") and
            p1.components["access"].get("to_level") ==
            p2.components["access"].get("to_level")
        )
        
        perm_similar = (
            set(p1.components["permission"].get("permissions", [])) ==
            set(p2.components["permission"].get("permissions", []))
        )
        
        # Compare strengths
        strength_diff = abs(p1.strength - p2.strength)
        
        return access_similar and perm_similar and strength_diff < 0.2

    def encode_patterns(self, patterns: List[Dict[str, any]], pattern_type: str) -> torch.Tensor:
        """
        Encode a list of patterns into hyperbolic space.
        
        Args:
            patterns: List of patterns to encode
            pattern_type: Type of patterns (access, permission, temporal, etc.)
            
        Returns:
            Tensor of encoded patterns
        """
        return torch.stack([
            self._encode_pattern(p, pattern_type) for p in patterns
        ]).to(self.device)

    def compute_risk_score(self, pattern: CorrelationPattern) -> float:
        """
        Compute risk score for a correlation pattern.
        
        Args:
            pattern: The correlation pattern to score
            
        Returns:
            Risk score between 0 and 1
        """
        return self._compute_risk_score(
            pattern.components.get("access", {}),
            pattern.components.get("permission", {}),
            pattern.components.get("temporal", {})
        )

    def compute_pattern_similarity(
        self,
        pattern1: CorrelationPattern,
        pattern2: CorrelationPattern
    ) -> float:
        """
        Compute similarity between two correlation patterns.
        
        Args:
            pattern1: First correlation pattern
            pattern2: Second correlation pattern
            
        Returns:
            Similarity score between 0 and 1
        """
        # Encode components
        p1_features = []
        p2_features = []
        
        for component in ["access", "permission", "temporal"]:
            if component in pattern1.components and component in pattern2.components:
                f1 = self._encode_pattern(pattern1.components[component], component)
                f2 = self._encode_pattern(pattern2.components[component], component)
                p1_features.append(f1)
                p2_features.append(f2)
        
        if not p1_features:
            return 0.0
        
        # Stack features
        p1_tensor = torch.stack(p1_features)
        p2_tensor = torch.stack(p2_features)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(p1_tensor, p2_tensor, dim=-1).mean()
        
        return similarity.item()