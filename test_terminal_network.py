"""
Test cases for Terminal Network Security Analysis
Implements attack scenarios to validate UHG-based detection
"""

import torch
import numpy as np
from terminal_network_uhg import (
    TerminalNode, TerminalNetwork, 
    TerminalSecurityAnalyzer, TerminalSecurityModel
)

def test_lateral_movement_attack():
    """Test detection of lateral movement attack.
    
    Scenario:
    1. Create legitimate terminal hierarchy
    2. Attacker compromises a terminal
    3. Attacker creates unauthorized connection to another branch
    4. System should detect the anomalous connection
    """
    # Create legitimate network
    network = TerminalNetwork()
    
    # Create root (regional processor)
    root = TerminalNode(
        "REGION_1", 
        torch.tensor([1.0, 0.0, 1.0])  # On null cone
    )
    network.terminals[root.terminal_id] = root
    
    # Create two store branches
    stores = []
    for i in range(2):
        store = TerminalNode(
            f"STORE_{i}",
            torch.tensor([1.0, float(i+1), 
                         np.sqrt(1 + (i+1)**2)])
        )
        root.add_child(store)
        network.terminals[store.terminal_id] = store
        stores.append(store)
    
    # Add legitimate terminals to each store
    for i, store in enumerate(stores):
        for j in range(2):
            term = TerminalNode(
                f"POS_{i}_{j}",
                torch.tensor([
                    1.0,
                    float(i+1),
                    float(j+1)
                ])
            )
            store.add_child(term)
            network.terminals[term.terminal_id] = term
    
    # Initialize analyzer and model
    analyzer = TerminalSecurityAnalyzer(network)
    analyzer.model = TerminalSecurityModel(input_dim=5)  # Basic features
    
    # Test legitimate network
    print("\nTesting legitimate network...")
    anomalies = analyzer.detect_anomalies()
    assert len(anomalies) == 0, "False positives in legitimate network"
    print("No anomalies detected in legitimate network")
    
    # Simulate attack: Create unauthorized connection
    print("\nSimulating lateral movement attack...")
    
    # Compromised terminal tries to connect to another branch
    compromised = network.terminals["POS_0_0"]
    target = network.terminals["POS_1_1"]
    
    # Create malicious connection by modifying coordinates
    attack_coords = network.join_points(
        compromised.coords,
        target.coords
    )
    compromised.coords = attack_coords
    
    # Test attack detection
    anomalies = analyzer.detect_anomalies()
    print(f"Detected anomalies: {anomalies}")
    
    # Verify detection
    assert "POS_0_0" in anomalies, "Failed to detect compromised terminal"
    print("Successfully detected lateral movement attack")
    
    # Analyze detection metrics
    print("\nAnalyzing attack patterns:")
    
    # Check quadrance (distance) violation
    q = network.quadrance(compromised.coords, target.coords)
    print(f"Attack path quadrance: {q:.4f}")
    
    # Check collinearity violation
    col = network.check_collinear(
        compromised.parent.coords,
        compromised.coords,
        target.coords
    )
    print(f"Collinearity preserved: {col}")
    
    # Check connection validity
    connection = network.join_points(compromised.coords, target.coords)
    valid = network.point_on_line(target.coords, connection)
    print(f"Connection validity: {valid}")

if __name__ == "__main__":
    test_lateral_movement_attack() 