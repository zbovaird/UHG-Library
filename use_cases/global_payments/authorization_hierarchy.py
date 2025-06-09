# Authorization Hierarchy Violation Detection using UHG
# Requires: pip install uhg torch numpy

import sys
import subprocess
import pkg_resources
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, DefaultDict
import torch.amp

def check_dependencies():
    """Check and install required packages."""
    required = {'uhg', 'torch', 'numpy'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print(f"Installing missing packages: {missing}")
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install'] + list(missing), 
                            stdout=subprocess.DEVNULL)
        print("Dependencies installed successfully.")
        print("Please restart the script for changes to take effect.")
        sys.exit(0)

# Check dependencies before imports
check_dependencies()

import torch
import numpy as np
from uhg.projective import ProjectiveUHG
from uhg.attention import UHGMultiHeadAttention, UHGAttentionConfig

@dataclass
class AccessEvent:
    """Represents a single access event in time."""
    timestamp: float
    level_from: str
    level_to: str
    permissions_used: Set[str]
    success: bool

@dataclass
class PatternFeatures:
    """Features for pattern detection."""
    access_sequence: torch.Tensor  # Sequence of access attempts
    permission_sequence: torch.Tensor  # Sequence of permission changes
    temporal_features: torch.Tensor  # Temporal patterns
    relationship_features: torch.Tensor  # Relationship patterns

class AuthLevel:
    """Represents an authorization level in hyperbolic space."""
    
    def __init__(self, level_id: str, permissions: Set[str], coords: Optional[torch.Tensor] = None):
        self.level_id = level_id
        self.permissions = permissions.copy()  # Make a copy of permissions
        self.parent: Optional['AuthLevel'] = None
        self.children: List['AuthLevel'] = []
        # Initialize in hyperbolic space if coords not provided
        if coords is None:
            self.coords = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        else:
            self.coords = coords.double()
        
        # Temporal tracking
        self.permission_history: List[Tuple[float, Set[str]]] = [(time.time(), permissions.copy())]
        self.access_attempts: List[AccessEvent] = []
        self.last_access_time: DefaultDict[str, float] = defaultdict(float)
        self.access_count: DefaultDict[str, int] = defaultdict(int)
            
    def add_child(self, child: 'AuthLevel'):
        """Add a child authorization level."""
        self.children.append(child)
        child.parent = self
        # Don't inherit permissions automatically
        
    def update_permissions(self, new_permissions: Set[str]):
        """Update permissions and record the change."""
        self.permissions = new_permissions.copy()
        self.permission_history.append((time.time(), new_permissions.copy()))
        
    def record_access_attempt(self, from_level: str, permissions_used: Set[str], success: bool):
        """Record an access attempt to this level."""
        event = AccessEvent(
            timestamp=time.time(),
            level_from=from_level,
            level_to=self.level_id,
            permissions_used=permissions_used,
            success=success
        )
        self.access_attempts.append(event)
        self.last_access_time[from_level] = event.timestamp
        self.access_count[from_level] += 1

class PerformanceMetrics:
    """Track performance metrics."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.batch_times = []
        self.memory_usage = []
        self.total_time = 0
        self.start_time = None
        
    def start_batch(self):
        """Start timing a batch."""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def end_batch(self):
        """End timing a batch and record metrics."""
        if self.start_time is not None:
            batch_time = time.time() - self.start_time
            self.batch_times.append(batch_time)
            
            if torch.cuda.is_available():
                memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                self.memory_usage.append(memory)
                
            self.start_time = None
            
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        if not self.batch_times:
            return {}
            
        summary = {
            "avg_batch_time": sum(self.batch_times) / len(self.batch_times),
            "total_time": sum(self.batch_times),
            "num_batches": len(self.batch_times)
        }
        
        if self.memory_usage:
            summary["max_memory_mb"] = max(self.memory_usage)
            summary["avg_memory_mb"] = sum(self.memory_usage) / len(self.memory_usage)
            
        return summary

class AdvancedPatternDetector:
    """Detects complex patterns using UHG attention mechanisms."""
    
    def __init__(self, feature_dim: int = 64, num_heads: int = 4, batch_size: int = 32, device: str = None):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        
        # Determine device (GPU if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize attention with config
        config = UHGAttentionConfig(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            use_cross_ratio=True
        )
        self.attention = UHGMultiHeadAttention(config).to(self.device)
        
        # Initialize performance tracking
        self.metrics = PerformanceMetrics()
        
        # Enable automatic mixed precision for better performance
        if self.device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize memory layout optimization
        self.optimize_memory_layout()
        
    def optimize_memory_layout(self):
        """Optimize memory layout for better performance."""
        if self.device == 'cuda':
            # Set optimal memory allocator
            torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 20% for system
            torch.cuda.empty_cache()
            
            # Enable memory pinning for faster CPU->GPU transfer
            torch.cuda.set_device(self.device)
            
            # Set optimal tensor layout
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_set_profiling_executor(True)
            
    def build_features(self, events: List[AccessEvent], window_size: float) -> PatternFeatures:
        """Build feature tensors from access events with batching."""
        if not events:
            return None
            
        self.metrics.reset()
        
        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)
        base_time = events[0].timestamp
        
        # Process in batches for memory efficiency
        seq_len = len(events)
        num_batches = (seq_len + self.batch_size - 1) // self.batch_size
        
        # Pre-allocate tensors with optimal memory layout
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        access_seq = torch.zeros((seq_len, self.feature_dim), device=self.device, dtype=dtype)
        perm_seq = torch.zeros((seq_len, self.feature_dim), device=self.device, dtype=dtype)
        temp_seq = torch.zeros((seq_len, self.feature_dim), device=self.device, dtype=dtype)
        rel_seq = torch.zeros((seq_len, self.feature_dim), device=self.device, dtype=dtype)
        
        # Pin memory for faster transfer if using CUDA
        if self.device == 'cuda':
            access_seq = access_seq.pin_memory()
            perm_seq = perm_seq.pin_memory()
            temp_seq = temp_seq.pin_memory()
            rel_seq = rel_seq.pin_memory()
        
        for batch_idx in range(num_batches):
            self.metrics.start_batch()
            
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, seq_len)
            batch_events = events[start_idx:end_idx]
            
            # Process batch with mixed precision if available
            if self.device == 'cuda':
                with torch.amp.autocast('cuda'):
                    self._process_feature_batch(
                        batch_events, start_idx, access_seq, perm_seq, 
                        temp_seq, rel_seq, base_time, window_size
                    )
                    torch.cuda.synchronize()
            else:
                self._process_feature_batch(
                    batch_events, start_idx, access_seq, perm_seq, 
                    temp_seq, rel_seq, base_time, window_size
                )
                
            self.metrics.end_batch()
            
        # Convert back to float32 for compatibility
        return PatternFeatures(
            access_sequence=access_seq.float(),
            permission_sequence=perm_seq.float(),
            temporal_features=temp_seq.float(),
            relationship_features=rel_seq.float()
        )
        
    def _process_feature_batch(
        self, batch_events: List[AccessEvent], start_idx: int,
        access_seq: torch.Tensor, perm_seq: torch.Tensor,
        temp_seq: torch.Tensor, rel_seq: torch.Tensor,
        base_time: float, window_size: float
    ):
        """Process a batch of events to build feature tensors."""
        for i, event in enumerate(batch_events):
            idx = start_idx + i
            # Access features
            access_seq[idx] = self._encode_access_event(event)
            
            # Permission features
            perm_seq[idx] = self._encode_permissions(event.permissions_used)
            
            # Temporal features
            temp_seq[idx] = self._encode_temporal(event.timestamp - base_time, window_size)
            
            # Relationship features
            rel_seq[idx] = self._encode_relationship(event.level_from, event.level_to)
        
    def _encode_access_event(self, event: AccessEvent) -> torch.Tensor:
        """Encode access event into feature vector."""
        features = torch.zeros(self.feature_dim, device=self.device)
        features[0] = float(event.success)
        return features
        
    def _encode_permissions(self, permissions: Set[str]) -> torch.Tensor:
        """Encode permissions into feature vector."""
        features = torch.zeros(self.feature_dim, device=self.device)
        # Map common permissions to feature indices
        perm_map = {"read": 0, "write": 1, "execute": 2, "all": 3}
        for perm in permissions:
            if perm in perm_map:
                features[perm_map[perm]] = 1.0
        return features
        
    def _encode_temporal(self, time_delta: float, window_size: float) -> torch.Tensor:
        """Encode temporal information into feature vector."""
        features = torch.zeros(self.feature_dim, device=self.device)
        # Normalize time within window
        norm_time = time_delta / window_size
        features[0] = norm_time
        return features
        
    def _encode_relationship(self, from_level: str, to_level: str) -> torch.Tensor:
        """Encode level relationship into feature vector."""
        features = torch.zeros(self.feature_dim, device=self.device)
        # Simple encoding for prototype - will be enhanced
        features[0] = hash(from_level) % (self.feature_dim // 2)
        features[1] = hash(to_level) % (self.feature_dim // 2)
        return features
        
    def detect_patterns(self, features: PatternFeatures) -> Dict[str, any]:
        """Detect patterns using attention mechanisms with batching."""
        self.metrics.reset()
        
        # Process in batches for memory efficiency
        seq_len = features.access_sequence.shape[0]
        num_batches = (seq_len + self.batch_size - 1) // self.batch_size
        
        # Pre-allocate output tensors with optimal layout
        all_access_patterns = []
        all_perm_patterns = []
        all_temp_patterns = []
        all_rel_patterns = []
        
        for batch_idx in range(num_batches):
            self.metrics.start_batch()
            
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, seq_len)
            
            # Process batch with mixed precision if available
            if self.device == 'cuda':
                with torch.amp.autocast('cuda'):
                    patterns = self._process_attention_batch(
                        features, start_idx, end_idx
                    )
                    torch.cuda.synchronize()
            else:
                patterns = self._process_attention_batch(
                    features, start_idx, end_idx
                )
            
            # Store batch results
            all_access_patterns.append(patterns[0])
            all_perm_patterns.append(patterns[1])
            all_temp_patterns.append(patterns[2])
            all_rel_patterns.append(patterns[3])
            
            self.metrics.end_batch()
        
        # Combine batch results efficiently
        if self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                patterns = self._combine_attention_patterns(
                    all_access_patterns, all_perm_patterns,
                    all_temp_patterns, all_rel_patterns
                )
        else:
            patterns = self._combine_attention_patterns(
                all_access_patterns, all_perm_patterns,
                all_temp_patterns, all_rel_patterns
            )
        
        # Add performance metrics
        patterns["performance"] = self.metrics.get_summary()
        
        return patterns
        
    def _process_attention_batch(
        self, features: PatternFeatures, 
        start_idx: int, end_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a batch through attention mechanism."""
        # Extract batch sequences
        access_batch = features.access_sequence[start_idx:end_idx].unsqueeze(0)
        perm_batch = features.permission_sequence[start_idx:end_idx].unsqueeze(0)
        temp_batch = features.temporal_features[start_idx:end_idx].unsqueeze(0)
        rel_batch = features.relationship_features[start_idx:end_idx].unsqueeze(0)
        
        # Process batches with attention
        access_patterns = self.attention(access_batch, access_batch, access_batch)
        perm_patterns = self.attention(perm_batch, perm_batch, perm_batch)
        temp_patterns = self.attention(temp_batch, temp_batch, temp_batch)
        rel_patterns = self.attention(rel_batch, rel_batch, rel_batch)
        
        return (
            access_patterns[0], perm_patterns[0],
            temp_patterns[0], rel_patterns[0]
        )
        
    def _combine_attention_patterns(
        self, access_patterns: List[torch.Tensor],
        perm_patterns: List[torch.Tensor],
        temp_patterns: List[torch.Tensor],
        rel_patterns: List[torch.Tensor]
    ) -> Dict[str, List[Dict[str, any]]]:
        """Combine and analyze attention patterns."""
        # Combine patterns
        access_combined = torch.cat(access_patterns, dim=0)
        perm_combined = torch.cat(perm_patterns, dim=0)
        temp_combined = torch.cat(temp_patterns, dim=0)
        rel_combined = torch.cat(rel_patterns, dim=0)
        
        # Analyze patterns
        return {
            "access_patterns": self._analyze_attention(access_combined),
            "permission_patterns": self._analyze_attention(perm_combined),
            "temporal_patterns": self._analyze_attention(temp_combined),
            "relationship_patterns": self._analyze_attention(rel_combined)
        }
        
    def _analyze_attention(self, attention_output: torch.Tensor) -> List[Dict[str, any]]:
        """Analyze attention output for patterns with GPU support."""
        patterns = []
        
        # Process with mixed precision if available
        if self.device == 'cuda':
            with torch.amp.autocast('cuda'):
                patterns = self._compute_attention_patterns(attention_output)
        else:
            patterns = self._compute_attention_patterns(attention_output)
            
        return patterns
        
    def _compute_attention_patterns(self, attention_output: torch.Tensor) -> List[Dict[str, any]]:
        """Compute attention patterns from output."""
        # Get attention weights
        weights = attention_output.mean(dim=0)
        
        # Find significant attention patterns
        threshold = weights.mean() + weights.std()
        significant_indices = torch.where(weights > threshold)
        
        # Convert to CPU for list creation
        weights = weights.cpu()
        significant_indices = [idx.cpu() for idx in significant_indices]
        
        patterns = []
        for i, j in zip(significant_indices[0], significant_indices[1]):
            patterns.append({
                "start_idx": i.item(),
                "end_idx": j.item(),
                "strength": weights[i, j].item()
            })
            
        return patterns

class AuthHierarchy:
    """Models authorization hierarchy using UHG principles."""
    
    def __init__(self):
        self.levels: Dict[str, AuthLevel] = {}
        self.uhg = ProjectiveUHG()
        self.temporal_window = 3600  # Default 1-hour window for temporal analysis
        self.pattern_detector = AdvancedPatternDetector()
        
    def add_level(self, level_id: str, permissions: Set[str], 
                  parent_id: Optional[str] = None) -> AuthLevel:
        """Add a new authorization level."""
        # Create level with appropriate coordinates
        if parent_id is None:
            # Root level on null cone
            coords = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        else:
            parent = self.levels[parent_id]
            # Place child in relation to parent using UHG
            parent_coords = parent.coords
            # Generate child coordinates using UHG principles
            child_offset = torch.tensor([0.0, len(parent.children) + 1, 1.0], dtype=torch.float64)
            coords = self.uhg.join(parent_coords, child_offset)
            coords = coords / torch.norm(coords)  # Normalize
            
        level = AuthLevel(level_id, permissions, coords)
        self.levels[level_id] = level
        
        # Set up parent-child relationship
        if parent_id:
            self.levels[parent_id].add_child(level)
            
        return level
    
    def check_valid_transition(self, from_id: str, to_id: str, permissions_used: Optional[Set[str]] = None) -> Tuple[bool, float]:
        """Check if transition between levels is valid and record the attempt."""
        if from_id not in self.levels or to_id not in self.levels:
            return False, float('inf')
            
        from_level = self.levels[from_id]
        to_level = self.levels[to_id]
        
        # Calculate UHG distance using cross-ratio
        line = self.uhg.join(from_level.coords, to_level.coords)
        ideal1, ideal2 = self.uhg.get_ideal_points(line)
        distance = self.uhg.cross_ratio(from_level.coords, to_level.coords, ideal1, ideal2)
        
        # Check if transition follows hierarchy
        is_valid = False
        current = to_level
        while current:
            if current == from_level:
                is_valid = True
                break
            current = current.parent
        
        # Record the access attempt
        if to_id in self.levels:
            self.levels[to_id].record_access_attempt(
                from_level=from_id,
                permissions_used=permissions_used or set(),
                success=is_valid
            )
            
        return is_valid, float(distance)
    
    def get_valid_paths(self, from_id: str, to_id: str) -> List[List[str]]:
        """Get all valid paths between two levels."""
        def find_paths(current: AuthLevel, target: AuthLevel, 
                      path: List[str], paths: List[List[str]], visited: Set[str]):
            if current.level_id in visited:
                return
            
            visited.add(current.level_id)
            path.append(current.level_id)
            
            if current == target:
                paths.append(path.copy())
            else:
                # Try parent path
                if current.parent and current.parent.level_id not in visited:
                    find_paths(current.parent, target, path, paths, visited)
                # Try children paths
                for child in current.children:
                    if child.level_id not in visited:
                        find_paths(child, target, path, paths, visited)
                
            path.pop()
            visited.remove(current.level_id)
            
        if from_id not in self.levels or to_id not in self.levels:
            return []
            
        paths: List[List[str]] = []
        find_paths(self.levels[from_id], self.levels[to_id], [], paths, set())
        return paths
    
    def get_inherited_permissions(self, level_id: str) -> Set[str]:
        """Get all permissions including inherited ones."""
        if level_id not in self.levels:
            return set()
            
        level = self.levels[level_id]
        permissions = level.permissions.copy()
        
        # Only inherit permissions that are explicitly allowed at this level
        if level.parent:
            parent_perms = self.get_inherited_permissions(level.parent.level_id)
            # Only inherit read permission by default
            if "read" in level.permissions:
                permissions.add("read")
            # Only inherit write if explicitly granted
            if "write" in level.permissions and "write" in parent_perms:
                permissions.add("write")
            # Only inherit admin permissions if explicitly granted
            if "all" in level.permissions and "all" in parent_perms:
                permissions.add("all")
                
        return permissions
    
    def to_tensor(self) -> torch.Tensor:
        """Convert hierarchy to tensor representation for ML."""
        coords = []
        for level in self.levels.values():
            coords.append(level.coords)
        return torch.stack(coords)
    
    def analyze_temporal_patterns(self, level_id: str, window_size: Optional[float] = None) -> Dict[str, any]:
        """Analyze temporal access patterns for a level."""
        if level_id not in self.levels:
            return {}
            
        level = self.levels[level_id]
        window = window_size or self.temporal_window
        current_time = time.time()
        
        # Filter recent events
        recent_events = [
            event for event in level.access_attempts 
            if current_time - event.timestamp <= window
        ]
        
        if not recent_events:
            return {"status": "no_recent_activity"}
            
        # Analyze patterns
        analysis = {
            "access_frequency": defaultdict(int),
            "success_rate": defaultdict(lambda: {"success": 0, "total": 0}),
            "permission_changes": [],
            "unusual_patterns": []
        }
        
        # Analyze access frequency and success rates
        for event in recent_events:
            analysis["access_frequency"][event.level_from] += 1
            analysis["success_rate"][event.level_from]["total"] += 1
            if event.success:
                analysis["success_rate"][event.level_from]["success"] += 1
                
        # Check for permission changes
        for timestamp, perms in level.permission_history:
            if current_time - timestamp <= window:
                analysis["permission_changes"].append({
                    "timestamp": timestamp,
                    "permissions": perms
                })
                
        # Detect unusual patterns
        for from_level, freq in analysis["access_frequency"].items():
            # High frequency access
            if freq > 10:  # Threshold to be tuned
                analysis["unusual_patterns"].append({
                    "type": "high_frequency",
                    "from_level": from_level,
                    "frequency": freq
                })
                
            # Sudden change in success rate
            if analysis["success_rate"][from_level]["total"] > 0:
                success_rate = (
                    analysis["success_rate"][from_level]["success"] /
                    analysis["success_rate"][from_level]["total"]
                )
                if success_rate < 0.5:  # Threshold to be tuned
                    analysis["unusual_patterns"].append({
                        "type": "low_success_rate",
                        "from_level": from_level,
                        "rate": success_rate
                    })
                    
        return analysis
        
    def detect_privilege_escalation(self, level_id: str, window_size: Optional[float] = None) -> List[Dict[str, any]]:
        """Detect potential privilege escalation patterns."""
        if level_id not in self.levels:
            return []
            
        level = self.levels[level_id]
        window = window_size or self.temporal_window
        current_time = time.time()
        
        escalation_patterns = []
        recent_events = [
            event for event in level.access_attempts 
            if current_time - event.timestamp <= window
        ]
        
        if not recent_events:
            return []
            
        # Track permission accumulation over time
        permission_timeline = defaultdict(set)
        for event in recent_events:
            if event.success:
                permission_timeline[event.level_from].update(event.permissions_used)
                
                # Check for unusual permission accumulation
                inherited = self.get_inherited_permissions(event.level_from)
                extra_perms = permission_timeline[event.level_from] - inherited
                if extra_perms:
                    escalation_patterns.append({
                        "type": "permission_accumulation",
                        "from_level": event.level_from,
                        "extra_permissions": extra_perms,
                        "timestamp": event.timestamp
                    })
                    
        # Check for rapid level transitions
        level_transitions = defaultdict(list)
        for event in recent_events:
            if event.success:
                level_transitions[event.level_from].append(event.timestamp)
                
        for from_level, timestamps in level_transitions.items():
            if len(timestamps) > 1:
                min_interval = min(t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]))
                if min_interval < 60:  # Threshold: transitions faster than 1 minute
                    escalation_patterns.append({
                        "type": "rapid_transitions",
                        "from_level": from_level,
                        "interval": min_interval,
                        "timestamp": timestamps[-1]
                    })
                    
        return escalation_patterns
    
    def analyze_patterns(self, level_id: str, window_size: Optional[float] = None) -> Dict[str, any]:
        """Analyze patterns using advanced detection."""
        if level_id not in self.levels:
            return {}
            
        level = self.levels[level_id]
        window = window_size or self.temporal_window
        current_time = time.time()
        
        # Get recent events
        recent_events = [
            event for event in level.access_attempts 
            if current_time - event.timestamp <= window
        ]
        
        if not recent_events:
            return {"status": "no_recent_activity"}
            
        # Build features
        features = self.pattern_detector.build_features(recent_events, window)
        if features is None:
            return {"status": "insufficient_data"}
            
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(features)
        
        # Analyze results
        analysis = {
            "patterns": patterns,
            "summary": self._summarize_patterns(patterns, recent_events)
        }
        
        return analysis
        
    def _summarize_patterns(self, patterns: Dict[str, any], events: List[AccessEvent]) -> Dict[str, any]:
        """Summarize detected patterns."""
        summary = {
            "high_risk_patterns": [],
            "unusual_sequences": [],
            "temporal_anomalies": []
        }
        
        num_events = len(events)
        
        # Analyze access patterns
        for pattern in patterns["access_patterns"]:
            if pattern["strength"] > 0.8:  # High attention strength
                start_idx = min(pattern["start_idx"], num_events - 1)
                end_idx = min(pattern["end_idx"], num_events - 1)
                start_event = events[start_idx]
                end_event = events[end_idx]
                summary["high_risk_patterns"].append({
                    "type": "access_sequence",
                    "from_level": start_event.level_from,
                    "to_level": end_event.level_to,
                    "strength": pattern["strength"]
                })
                
        # Analyze permission patterns
        for pattern in patterns["permission_patterns"]:
            if pattern["strength"] > 0.7:
                start_idx = min(pattern["start_idx"], num_events - 1)
                end_idx = min(pattern["end_idx"], num_events - 1)
                start_event = events[start_idx]
                end_event = events[end_idx]
                summary["unusual_sequences"].append({
                    "type": "permission_change",
                    "start_permissions": start_event.permissions_used,
                    "end_permissions": end_event.permissions_used,
                    "strength": pattern["strength"]
                })
                
        # Analyze temporal patterns
        for pattern in patterns["temporal_patterns"]:
            if pattern["strength"] > 0.9:
                start_idx = min(pattern["start_idx"], num_events - 1)
                end_idx = min(pattern["end_idx"], num_events - 1)
                start_time = events[start_idx].timestamp
                end_time = events[end_idx].timestamp
                summary["temporal_anomalies"].append({
                    "type": "temporal_sequence",
                    "time_span": end_time - start_time,
                    "strength": pattern["strength"]
                })
                
        return summary

def test_hierarchy():
    """Test authorization hierarchy implementation."""
    print("\nTesting Authorization Hierarchy Implementation")
    print("--------------------------------------------")
    
    # Create hierarchy
    print("\n1. Creating hierarchy...")
    hierarchy = AuthHierarchy()
    
    # Add levels
    print("2. Adding authorization levels...")
    hierarchy.add_level("admin", {"all", "read", "write"})
    hierarchy.add_level("manager", {"read", "write", "approve"}, "admin")
    hierarchy.add_level("user", {"read", "write"}, "manager")
    hierarchy.add_level("guest", {"read"}, "user")
    print("   Created: admin -> manager -> user -> guest")
    
    # Test permission inheritance
    print("\n3. Testing permission inheritance...")
    assert "all" in hierarchy.get_inherited_permissions("admin"), "Admin should have all permissions"
    assert "write" in hierarchy.get_inherited_permissions("manager"), "Manager should have write permission"
    assert "read" in hierarchy.get_inherited_permissions("user"), "User should have read permission"
    assert "write" in hierarchy.get_inherited_permissions("user"), "User should have write permission"
    assert "read" in hierarchy.get_inherited_permissions("guest"), "Guest should have read permission"
    assert "write" not in hierarchy.get_inherited_permissions("guest"), "Guest should not have write permission"
    print("   ✓ Permission inheritance working correctly")
    
    # Test valid transitions
    print("\n4. Testing level transitions...")
    valid, distance = hierarchy.check_valid_transition("admin", "manager")
    assert valid and distance < 1.0
    print(f"   ✓ Admin -> Manager transition: valid={valid}, distance={distance:.4f}")
    
    valid, distance = hierarchy.check_valid_transition("guest", "admin")
    assert not valid
    print(f"   ✓ Guest -> Admin transition: valid={valid}, distance={distance:.4f}")
    
    # Test path finding
    print("\n5. Testing path finding...")
    paths = hierarchy.get_valid_paths("guest", "admin")
    assert len(paths) > 0
    assert paths[0] == ["guest", "user", "manager", "admin"]
    print(f"   ✓ Found valid path: {' -> '.join(paths[0])}")
    
    # Test UHG properties
    print("\n6. Testing UHG properties...")
    coords = hierarchy.to_tensor()
    assert coords.shape == (4, 3)  # 4 levels, 3D coordinates
    print(f"   ✓ Hierarchy embedded in 3D hyperbolic space: shape={coords.shape}")
    
    print("\nAll tests passed successfully!")

def test_temporal_analysis():
    """Test temporal analysis functionality."""
    print("\nTesting Temporal Analysis")
    print("------------------------")
    
    # Create hierarchy
    hierarchy = AuthHierarchy()
    
    # Add levels
    hierarchy.add_level("admin", {"all", "read", "write"})
    hierarchy.add_level("manager", {"read", "write", "approve"}, "admin")
    hierarchy.add_level("user", {"read", "write"}, "manager")
    hierarchy.add_level("guest", {"read"}, "user")
    
    # Simulate access patterns
    print("\n1. Simulating normal access patterns...")
    # Normal access
    hierarchy.check_valid_transition("user", "guest", {"read"})
    hierarchy.check_valid_transition("manager", "user", {"read", "write"})
    time.sleep(1)  # Small delay to create time difference
    
    # Simulate permission change
    print("2. Testing permission changes...")
    user_level = hierarchy.levels["user"]
    user_level.update_permissions({"read", "write", "execute"})
    
    # Simulate suspicious patterns
    print("3. Simulating suspicious patterns...")
    # Rapid transitions
    for _ in range(5):
        hierarchy.check_valid_transition("guest", "user", {"read"})
        time.sleep(0.1)
    
    # Permission accumulation attempt
    hierarchy.check_valid_transition("guest", "admin", {"all", "read", "write"})
    
    # Analyze patterns
    print("\n4. Analyzing temporal patterns...")
    user_analysis = hierarchy.analyze_temporal_patterns("user")
    print("\nUser level analysis:")
    print(f"Access frequency: {dict(user_analysis['access_frequency'])}")
    print(f"Unusual patterns: {user_analysis['unusual_patterns']}")
    
    # Check privilege escalation
    print("\n5. Checking privilege escalation...")
    escalation = hierarchy.detect_privilege_escalation("admin")
    if escalation:
        print("Detected privilege escalation attempts:")
        for pattern in escalation:
            print(f"- {pattern['type']} from {pattern['from_level']}")
    else:
        print("No privilege escalation detected")
    
    print("\nTemporal analysis tests completed!")

def test_advanced_patterns():
    """Test advanced pattern detection."""
    print("\nTesting Advanced Pattern Detection")
    print("--------------------------------")
    
    # Create hierarchy
    hierarchy = AuthHierarchy()
    
    # Add levels
    hierarchy.add_level("admin", {"all", "read", "write"})
    hierarchy.add_level("manager", {"read", "write", "approve"}, "admin")
    hierarchy.add_level("user", {"read", "write"}, "manager")
    hierarchy.add_level("guest", {"read"}, "user")
    
    # Simulate complex patterns
    print("\n1. Simulating complex access patterns...")
    
    # Pattern 1: Rapid escalation attempt
    for _ in range(3):
        hierarchy.check_valid_transition("guest", "user", {"read"})
        time.sleep(0.1)
        hierarchy.check_valid_transition("user", "manager", {"write"})
        time.sleep(0.1)
        hierarchy.check_valid_transition("manager", "admin", {"all"})
        time.sleep(0.1)
    
    # Pattern 2: Permission accumulation
    print("2. Simulating permission accumulation...")
    hierarchy.check_valid_transition("user", "manager", {"read"})
    time.sleep(0.2)
    hierarchy.check_valid_transition("user", "manager", {"read", "write"})
    time.sleep(0.2)
    hierarchy.check_valid_transition("user", "manager", {"read", "write", "approve"})
    
    # Analyze patterns
    print("\n3. Analyzing patterns...")
    manager_analysis = hierarchy.analyze_patterns("manager")
    
    print("\nDetected Patterns:")
    if "high_risk_patterns" in manager_analysis["summary"]:
        print("\nHigh Risk Patterns:")
        for pattern in manager_analysis["summary"]["high_risk_patterns"]:
            print(f"- {pattern['type']} from {pattern['from_level']} to {pattern['to_level']} (strength: {pattern['strength']:.2f})")
    
    if "unusual_sequences" in manager_analysis["summary"]:
        print("\nUnusual Sequences:")
        for pattern in manager_analysis["summary"]["unusual_sequences"]:
            print(f"- {pattern['type']} (strength: {pattern['strength']:.2f})")
    
    if "temporal_anomalies" in manager_analysis["summary"]:
        print("\nTemporal Anomalies:")
        for pattern in manager_analysis["summary"]["temporal_anomalies"]:
            print(f"- {pattern['type']} spanning {pattern['time_span']:.2f}s (strength: {pattern['strength']:.2f})")
    
    print("\nAdvanced pattern detection tests completed!")

if __name__ == "__main__":
    test_hierarchy()
    test_temporal_analysis()
    test_advanced_patterns() 