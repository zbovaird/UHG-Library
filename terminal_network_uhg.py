"""
Terminal Network Security Analysis using Universal Hyperbolic Geometry
Implements UHG theorems for POS terminal network monitoring with exact equations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from uhg.projective import ProjectiveUHG

class TerminalNode:
    """Represents a POS terminal in hyperbolic space."""
    
    def __init__(self, terminal_id: str, coords: torch.Tensor):
        """Initialize terminal with projective coordinates [x:y:z]."""
        self.terminal_id = terminal_id
        self.coords = coords  # Projective coordinates [x:y:z]
        self.children: List[TerminalNode] = []
        self.parent: Optional[TerminalNode] = None
        
    def add_child(self, child: 'TerminalNode'):
        """Add child terminal to hierarchy."""
        self.children.append(child)
        child.parent = self

class TerminalNetwork:
    """Models POS terminal network in hyperbolic space using UHG theorems."""
    
    def __init__(self, dim: int = 3):
        self.dim = dim
        self.terminals: Dict[str, TerminalNode] = {}
        
    def join_points(self, a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        """Theorem 1: Join of points - Connect terminals.
        
        For points a1=[x1:y1:z1] and a2=[x2:y2:z2], returns line
        L = (y1z2-y2z1 : z1x2-z2x1 : x2y1-x1y2)
        """
        x1, y1, z1 = a1
        x2, y2, z2 = a2
        
        return torch.tensor([
            y1*z2 - y2*z1,  # First component
            z1*x2 - z2*x1,  # Second component
            x2*y1 - x1*y2   # Third component
        ])
    
    def check_collinear(self, a1: torch.Tensor, a2: torch.Tensor, 
                       a3: torch.Tensor) -> bool:
        """Theorem 3: Check if three terminals are collinear.
        
        Points a1=[x1:y1:z1], a2=[x2:y2:z2], a3=[x3:y3:z3] are collinear when:
        x1y2z3 - x1y3z2 + x2y3z1 - x3y2z1 + x3y1z2 - x2y1z3 = 0
        """
        x1, y1, z1 = a1
        x2, y2, z2 = a2
        x3, y3, z3 = a3
        
        det = (x1*y2*z3 - x1*y3*z2 + x2*y3*z1 - 
               x3*y2*z1 + x3*y1*z2 - x2*y1*z3)
        
        return torch.abs(det) < 1e-6
    
    def point_on_line(self, point: torch.Tensor, line: torch.Tensor) -> bool:
        """Theorem 6: Check if terminal lies on connection line.
        
        A point lies on at most two null lines.
        Uses inner product test in projective space.
        """
        # Inner product in projective space should be zero
        return torch.abs(torch.dot(point, line)) < 1e-6
    
    def quadrance(self, a1: torch.Tensor, a2: torch.Tensor) -> float:
        """Theorem 32: Measure relationship distance between terminals.
        
        For points a1=[x1:y1:z1] and a2=[x2:y2:z2]:
        q(a1,a2) = 1 - (x1x2 + y1y2 - z1z2)² / [(x1²+y1²-z1²)(x2²+y2²-z2²)]
        """
        x1, y1, z1 = a1
        x2, y2, z2 = a2
        
        # Numerator terms
        dot_product = x1*x2 + y1*y2 - z1*z2
        
        # Denominator terms
        a1_term = x1*x1 + y1*y1 - z1*z1
        a2_term = x2*x2 + y2*y2 - z2*z2
        
        # Full quadrance formula
        return 1.0 - (dot_product * dot_product) / (a1_term * a2_term)

class TerminalSecurityModel(nn.Module):
    """Neural network for terminal security analysis using UHG embeddings."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.uhg = ProjectiveUHG(
            in_channels=input_dim,
            hidden_channels=64,
            num_layers=3,
            dropout=0.1
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through UHG and MLP layers."""
        # Get hyperbolic embeddings
        h = self.uhg(x, edge_index)
        # Classify security state
        return self.mlp(h)

class TerminalSecurityAnalyzer:
    """Main class for terminal network security analysis."""
    
    def __init__(self, network: TerminalNetwork):
        self.network = network
        self.model = None
        
    def build_features(self, terminal: TerminalNode) -> List[float]:
        """Extract terminal features using UHG metrics."""
        features = []
        
        # Basic topology features
        features.extend([
            len(terminal.children),  # Number of child terminals
            1 if terminal.parent else 0,  # Is leaf node
            
            # Quadrance to parent (Theorem 32)
            self.network.quadrance(
                terminal.coords, 
                terminal.parent.coords if terminal.parent else 
                torch.zeros_like(terminal.coords)
            )
        ])
        
        # Collinearity features (Theorem 3)
        if terminal.parent and terminal.children:
            for child in terminal.children:
                # Check parent-terminal-child collinearity
                features.append(
                    self.network.check_collinear(
                        terminal.parent.coords,
                        terminal.coords,
                        child.coords
                    )
                )
                
                # Get connection line (Theorem 1)
                connection = self.network.join_points(terminal.coords, child.coords)
                
                # Check if connection is valid (Theorem 6)
                features.append(
                    self.network.point_on_line(child.coords, connection)
                )
        
        return features
    
    def detect_anomalies(self, threshold: float = 0.5) -> List[str]:
        """Detect anomalous terminals using UHG-based features."""
        anomalies = []
        
        # Build feature matrix
        features = []
        edge_list = []
        terminal_ids = list(self.network.terminals.keys())
        
        for i, terminal_id in enumerate(terminal_ids):
            terminal = self.network.terminals[terminal_id]
            features.append(self.build_features(terminal))
            
            # Add edges based on join operation (Theorem 1)
            if terminal.parent:
                parent_idx = terminal_ids.index(terminal.parent.terminal_id)
                edge_list.append([parent_idx, i])
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        edge_index = torch.LongTensor(edge_list).t()
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            scores = self.model(X, edge_index)
            
        # Identify anomalies
        for i, score in enumerate(scores):
            if score > threshold:
                anomalies.append(terminal_ids[i])
        
        return anomalies

def create_test_network() -> TerminalNetwork:
    """Create sample terminal network for testing."""
    network = TerminalNetwork()
    
    # Create root terminal in projective space
    root = TerminalNode("ROOT", torch.tensor([1.0, 0.0, 1.0]))  # On null cone
    network.terminals[root.terminal_id] = root
    
    # Add child terminals with proper projective coordinates
    for i in range(3):
        # Child coordinates ensuring proper hyperbolic relationships
        child = TerminalNode(
            f"TERM_{i}",
            torch.tensor([1.0, float(i+1), 
                         np.sqrt(1 + (i+1)**2)])  # Maintains hyperbolic constraint
        )
        root.add_child(child)
        network.terminals[child.terminal_id] = child
        
        # Add grandchildren
        for j in range(2):
            # Grandchild coordinates
            x = 1.0
            y = float(i+1)
            z = float(j+1)
            norm = np.sqrt(x*x + y*y - z*z)  # Normalize to maintain hyperbolic structure
            
            grandchild = TerminalNode(
                f"TERM_{i}_{j}",
                torch.tensor([x/norm, y/norm, z/norm])
            )
            child.add_child(grandchild)
            network.terminals[grandchild.terminal_id] = grandchild
    
    return network

if __name__ == "__main__":
    # Create test network
    network = create_test_network()
    
    # Initialize analyzer
    analyzer = TerminalSecurityAnalyzer(network)
    
    # Test anomaly detection
    anomalies = analyzer.detect_anomalies()
    print(f"Detected anomalies: {anomalies}")