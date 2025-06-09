# Terminal Network Security Analysis for Splunk SOAR
# Implements UHG-based detection of terminal compromise patterns

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class TerminalNode:
    """Represents a POS terminal in hyperbolic space."""
    
    def __init__(self, terminal_id: str, coords: torch.Tensor, metadata: dict = None):
        """Initialize terminal with coordinates and metadata."""
        self.terminal_id = terminal_id
        self.coords = coords.double()
        self.metadata = metadata or {}
        self.related_terminals: List[TerminalNode] = []
        self.parent: Optional[TerminalNode] = None
        self.last_seen = self.metadata.get('last_seen', 0)
        self.transaction_count = self.metadata.get('transaction_count', 0)
        self.auth_failures = self.metadata.get('auth_failures', 0)
        
    def add_related(self, related: 'TerminalNode'):
        """Add related terminal to network."""
        self.related_terminals.append(related)
        related.parent = self

class TerminalNetwork:
    """Models terminal network using UHG theorems."""
    
    def __init__(self):
        self.terminals: Dict[str, TerminalNode] = {}
        
    def join_points(self, a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
        """UHG Theorem 1: Join of terminals."""
        x1, y1, z1 = a1.double()
        x2, y2, z2 = a2.double()
        return torch.tensor([
            y1*z2 - y2*z1,
            z1*x2 - z2*x1,
            x2*y1 - x1*y2
        ], dtype=torch.float64)
    
    def check_collinear(self, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor) -> bool:
        """UHG Theorem 3: Check terminal alignment."""
        x1, y1, z1 = a1.double()
        x2, y2, z2 = a2.double()
        x3, y3, z3 = a3.double()
        det = (x1*y2*z3 - x1*y3*z2 + x2*y3*z1 - x3*y2*z1 + x3*y1*z2 - x2*y1*z3)
        return torch.abs(det) < 1e-6
    
    def quadrance(self, a1: torch.Tensor, a2: torch.Tensor) -> float:
        """UHG Theorem 32: Measure terminal relationship distance."""
        x1, y1, z1 = a1.double()
        x2, y2, z2 = a2.double()
        
        # Normalize coordinates to prevent numerical issues
        norm1 = torch.sqrt(x1*x1 + y1*y1 + z1*z1)
        norm2 = torch.sqrt(x2*x2 + y2*y2 + z2*z2)
        
        if norm1 > 1e-6 and norm2 > 1e-6:
            x1, y1, z1 = x1/norm1, y1/norm1, z1/norm1
            x2, y2, z2 = x2/norm2, y2/norm2, z2/norm2
        
        # Calculate quadrance with normalized coordinates
        dot_product = x1*x2 + y1*y2 - z1*z2
        a1_term = x1*x1 + y1*y1 - z1*z1
        a2_term = x2*x2 + y2*y2 - z2*z2
        
        # Handle numerical stability
        if abs(a1_term) < 1e-6 or abs(a2_term) < 1e-6:
            return 1.0  # Maximum distance
        
        q = 1.0 - (dot_product * dot_product) / (a1_term * a2_term)
        return min(max(q, 0.0), 1.0)  # Clamp between 0 and 1

class TerminalAnalyzer:
    """Analyzes terminal patterns for Splunk SOAR."""
    
    def __init__(self, network: TerminalNetwork):
        self.network = network
        self.feature_dim = 8  # Increased for terminal-specific features
    
    def build_features(self, terminal: TerminalNode) -> List[float]:
        """Extract terminal features for security analysis."""
        features = []
        
        # 1. Network topology features
        features.extend([
            len(terminal.related_terminals) / 10.0,
            terminal.transaction_count / 1000.0,
            terminal.auth_failures / 100.0
        ])
        
        # 2. Parent terminal relationship
        if terminal.parent:
            parent_quad = self.network.quadrance(
                terminal.coords, 
                terminal.parent.coords
            )
            features.append(parent_quad)
            
            # Time since last parent interaction
            time_diff = abs(terminal.last_seen - terminal.parent.last_seen)
            features.append(min(time_diff / 3600.0, 24.0))  # Cap at 24 hours
        else:
            features.extend([0.0, 0.0])
        
        # 3. Terminal behavior patterns
        if terminal.parent and terminal.related_terminals:
            related = terminal.related_terminals
            
            # Geometric pattern detection
            pattern_count = sum(
                1 for r1 in related for r2 in related if r1 != r2 and
                self.network.check_collinear(
                    terminal.parent.coords,
                    r1.coords,
                    r2.coords
                )
            )
            features.append(pattern_count / max(1, len(related) * (len(related) - 1)))
            
            # Transaction pattern analysis
            tx_counts = [r.transaction_count for r in related]
            features.append(np.std(tx_counts) / (np.mean(tx_counts) + 1e-6))
            
            # Auth failure analysis
            auth_fails = [r.auth_failures for r in related]
            features.append(np.mean(auth_fails) / 100.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        assert len(features) == self.feature_dim
        return features
    
    def analyze_terminal(self, terminal_id: str) -> dict:
        """Analyze single terminal for Splunk SOAR."""
        terminal = self.network.terminals.get(terminal_id)
        if not terminal:
            return {"error": "Terminal not found"}
            
        features = self.build_features(terminal)
        alerts = []
        
        # Check for suspicious patterns
        if terminal.parent:
            quad = self.network.quadrance(terminal.coords, terminal.parent.coords)
            
            # 1. Unusual network distance (adjusted thresholds)
            if quad > 0.6:  # Reduced from 0.8
                alerts.append({
                    "type": "NETWORK_DISTANCE",
                    "severity": "high",
                    "details": f"Unusual network distance from parent: {quad:.4f}"
                })
            
            # 2. Auth failure spikes
            if terminal.auth_failures > 10:
                alerts.append({
                    "type": "AUTH_FAILURES",
                    "severity": "medium",
                    "details": f"High number of auth failures: {terminal.auth_failures}"
                })
            
            # 3. Transaction pattern anomalies
            if len(terminal.related_terminals) > 0:
                tx_counts = [t.transaction_count for t in terminal.related_terminals]
                mean_tx = np.mean(tx_counts)
                if terminal.transaction_count > mean_tx * 3:
                    alerts.append({
                        "type": "TRANSACTION_VOLUME",
                        "severity": "medium",
                        "details": f"Unusual transaction volume: {terminal.transaction_count} vs mean {mean_tx:.2f}"
                    })
            
                # 4. Add coordinate validation alert
                coords = terminal.coords
                norm = torch.sqrt(torch.sum(coords * coords))
                if norm > 5.0:  # Check for unusually large coordinates
                    alerts.append({
                        "type": "COORDINATE_SCALE",
                        "severity": "medium",
                        "details": f"Unusual coordinate scale: {norm:.4f}"
                    })
        
        return {
            "terminal_id": terminal_id,
            "features": features,
            "alerts": alerts,
            "metadata": terminal.metadata,
            "coordinates": terminal.coords.tolist(),  # Add coordinates for debugging
            "parent_id": terminal.parent.terminal_id if terminal.parent else None
        }
    
    def analyze_network(self) -> List[dict]:
        """Analyze entire terminal network for Splunk SOAR."""
        results = []
        for terminal_id in self.network.terminals:
            analysis = self.analyze_terminal(terminal_id)
            if analysis.get("alerts"):  # Only include terminals with alerts
                results.append(analysis)
        return results

def process_splunk_event(event_data: dict) -> dict:
    """Process Splunk SOAR event and return analysis results."""
    # Create network from event data
    network = TerminalNetwork()
    
    # Parse terminal data from event
    for terminal_data in event_data.get("terminals", []):
        terminal = TerminalNode(
            terminal_id=terminal_data["id"],
            coords=torch.tensor(terminal_data["coords"]),
            metadata=terminal_data.get("metadata", {})
        )
        network.terminals[terminal.terminal_id] = terminal
    
    # Build relationships
    for terminal_data in event_data.get("terminals", []):
        if "parent_id" in terminal_data:
            parent = network.terminals.get(terminal_data["parent_id"])
            terminal = network.terminals.get(terminal_data["id"])
            if parent and terminal:
                parent.add_related(terminal)
    
    # Analyze network
    analyzer = TerminalAnalyzer(network)
    results = analyzer.analyze_network()
    
    return {
        "timestamp": event_data.get("timestamp"),
        "analysis_results": results,
        "alert_count": sum(len(r["alerts"]) for r in results)
    }

# Example usage for Splunk SOAR
def test_splunk_integration():
    """Test the Splunk SOAR integration."""
    # Sample event data with corrected coordinates
    event_data = {
        "timestamp": "2024-03-19T12:00:00Z",
        "terminals": [
            {
                "id": "STORE_1",
                "coords": [1.0, 0.0, 1.0],  # Root node on null cone
                "metadata": {
                    "last_seen": 1710854400,
                    "transaction_count": 1000,
                    "auth_failures": 2
                }
            },
            {
                "id": "POS_1_1",
                "parent_id": "STORE_1",
                "coords": [1.0, 0.5, 1.118],  # Normal distance
                "metadata": {
                    "last_seen": 1710854400,
                    "transaction_count": 500,
                    "auth_failures": 15
                }
            },
            {
                "id": "POS_1_2",
                "parent_id": "STORE_1",
                "coords": [2.0, 3.0, 4.0],  # Unusual coordinates (should trigger alert)
                "metadata": {
                    "last_seen": 1710854400,
                    "transaction_count": 2000,
                    "auth_failures": 1
                }
            }
        ]
    }
    
    # Process event
    results = process_splunk_event(event_data)
    
    # Print results with more detail
    print("\nSplunk SOAR Analysis Results:")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Alerts: {results['alert_count']}")
    
    for terminal in results["analysis_results"]:
        print(f"\nTerminal: {terminal['terminal_id']}")
        print(f"Coordinates: {terminal['coordinates']}")
        print(f"Parent: {terminal['parent_id']}")
        for alert in terminal["alerts"]:
            print(f"- {alert['type']} ({alert['severity']}): {alert['details']}")
        print(f"Features: {terminal['features']}")

if __name__ == "__main__":
    test_splunk_integration() 