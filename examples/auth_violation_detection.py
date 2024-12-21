"""
Example of Authorization Hierarchy Violation Detection using Pattern Correlation.

This example demonstrates how to use the UHG pattern correlation module
to detect complex authorization hierarchy violations.
"""

import time
from typing import List, Dict
from uhg.patterns.correlation import PatternCorrelator

def generate_synthetic_access_logs() -> List[Dict]:
    """Generate synthetic access logs for demonstration."""
    base_time = time.time()
    return [
        {
            "timestamp": base_time,
            "user_id": "user123",
            "action": "login",
            "from_level": "user",
            "to_level": "user",
            "success": True
        },
        {
            "timestamp": base_time + 30,
            "user_id": "user123",
            "action": "elevate",
            "from_level": "user",
            "to_level": "power_user",
            "success": True
        },
        {
            "timestamp": base_time + 45,
            "user_id": "user123",
            "action": "elevate",
            "from_level": "power_user",
            "to_level": "admin",
            "success": True
        }
    ]

def generate_synthetic_permission_logs() -> List[Dict]:
    """Generate synthetic permission change logs."""
    base_time = time.time()
    return [
        {
            "timestamp": base_time + 15,
            "user_id": "user123",
            "permissions_added": {"read", "write"},
            "permissions_removed": set(),
            "level": "user"
        },
        {
            "timestamp": base_time + 35,
            "user_id": "user123",
            "permissions_added": {"execute"},
            "permissions_removed": set(),
            "level": "power_user"
        },
        {
            "timestamp": base_time + 50,
            "user_id": "user123",
            "permissions_added": {"delete", "modify_acl"},
            "permissions_removed": set(),
            "level": "admin"
        }
    ]

def extract_access_patterns(logs: List[Dict]) -> List[Dict]:
    """Extract access patterns from logs."""
    patterns = []
    
    for i in range(len(logs) - 1):
        current = logs[i]
        next_log = logs[i + 1]
        
        # Look for privilege escalation patterns
        if (
            current["user_id"] == next_log["user_id"] and
            current["to_level"] != next_log["to_level"]
        ):
            time_diff = next_log["timestamp"] - current["timestamp"]
            strength = 0.5
            
            # Increase strength for rapid escalations
            if time_diff < 60:  # Less than 1 minute
                strength = 0.9
            elif time_diff < 300:  # Less than 5 minutes
                strength = 0.7
                
            # Increase strength for big jumps in privilege
            if (
                current["to_level"] == "user" and
                next_log["to_level"] == "admin"
            ):
                strength *= 1.3
                
            patterns.append({
                "id": f"access_{i}",
                "type": "escalation",
                "strength": strength,
                "from_level": current["to_level"],
                "to_level": next_log["to_level"],
                "time_diff": time_diff
            })
            
    return patterns

def extract_permission_patterns(logs: List[Dict]) -> List[Dict]:
    """Extract permission patterns from logs."""
    patterns = []
    
    for i in range(len(logs) - 1):
        current = logs[i]
        next_log = logs[i + 1]
        
        if current["user_id"] == next_log["user_id"]:
            time_diff = next_log["timestamp"] - current["timestamp"]
            combined_perms = (
                current["permissions_added"] |
                next_log["permissions_added"]
            )
            
            # Detect unusual permission combinations
            high_risk_combinations = [
                {"write", "execute", "delete"},
                {"modify_acl", "delete"},
                {"read", "write", "execute", "delete"}
            ]
            
            pattern_type = "normal"
            strength = 0.5
            
            for risk_combo in high_risk_combinations:
                if risk_combo.issubset(combined_perms):
                    pattern_type = "unusual_combination"
                    strength = 0.8
                    break
                    
            patterns.append({
                "id": f"perm_{i}",
                "type": pattern_type,
                "strength": strength,
                "permissions": combined_perms,
                "time_diff": time_diff
            })
            
    return patterns

def extract_temporal_patterns(
    access_logs: List[Dict],
    perm_logs: List[Dict]
) -> List[Dict]:
    """Extract temporal patterns from combined logs."""
    # Combine and sort logs by timestamp
    combined_logs = (
        [(log, "access") for log in access_logs] +
        [(log, "permission") for log in perm_logs]
    )
    combined_logs.sort(key=lambda x: x[0]["timestamp"])
    
    patterns = []
    window_size = 300  # 5 minutes
    
    for i in range(len(combined_logs)):
        window_logs = [
            log for log in combined_logs[i:]
            if (
                log[0]["timestamp"] -
                combined_logs[i][0]["timestamp"]
            ) <= window_size
        ]
        
        if len(window_logs) < 2:
            continue
            
        # Compute time span and activity density
        time_span = (
            window_logs[-1][0]["timestamp"] -
            window_logs[0][0]["timestamp"]
        )
        activity_density = len(window_logs) / time_span if time_span > 0 else 1.0
        
        # Determine pattern strength based on density and types
        strength = min(0.4 + activity_density * 0.3, 0.9)
        
        pattern_type = "normal"
        if time_span < 60 and len(window_logs) >= 3:
            pattern_type = "rapid"
            strength = min(strength * 1.3, 0.9)
            
        patterns.append({
            "id": f"temporal_{i}",
            "type": pattern_type,
            "strength": strength,
            "time_span": time_span,
            "start_time": window_logs[0][0]["timestamp"],
            "end_time": window_logs[-1][0]["timestamp"],
            "activity_count": len(window_logs)
        })
        
    return patterns

def main():
    """Run authorization hierarchy violation detection example."""
    # Generate synthetic logs
    access_logs = generate_synthetic_access_logs()
    permission_logs = generate_synthetic_permission_logs()
    
    print("\nGenerated Logs:")
    print("-" * 50)
    print("\nAccess Logs:")
    for log in access_logs:
        print(f"  {log['action']}: {log['from_level']} -> {log['to_level']}")
    
    print("\nPermission Logs:")
    for log in permission_logs:
        print(f"  Added: {log['permissions_added']}, Level: {log['level']}")
    
    # Extract patterns
    access_patterns = extract_access_patterns(access_logs)
    permission_patterns = extract_permission_patterns(permission_logs)
    temporal_patterns = extract_temporal_patterns(
        access_logs,
        permission_logs
    )
    
    print("\nExtracted Patterns:")
    print("-" * 50)
    print("\nAccess Patterns:")
    for pattern in access_patterns:
        print(f"  Type: {pattern['type']}, Strength: {pattern['strength']:.2f}")
        print(f"  From: {pattern['from_level']} -> {pattern['to_level']}")
    
    print("\nPermission Patterns:")
    for pattern in permission_patterns:
        print(f"  Type: {pattern['type']}, Strength: {pattern['strength']:.2f}")
        print(f"  Permissions: {pattern['permissions']}")
    
    print("\nTemporal Patterns:")
    for pattern in temporal_patterns:
        print(f"  Type: {pattern['type']}, Strength: {pattern['strength']:.2f}")
        print(f"  Time Span: {pattern['time_span']:.2f}s")
    
    # Initialize pattern correlator with lower threshold
    correlator = PatternCorrelator(
        feature_dim=64,
        num_heads=4,
        correlation_threshold=0.3  # Even lower threshold for demonstration
    )
    
    # Correlate patterns
    correlated_patterns = correlator.correlate_patterns(
        access_patterns,
        permission_patterns,
        temporal_patterns,
        []  # No relationship patterns in this example
    )
    
    # Analyze pattern evolution
    evolution = correlator.analyze_pattern_evolution(
        correlated_patterns,
        time_window=3600
    )
    
    # Print results
    print("\nDetected Violations:")
    print("-" * 50)
    
    for pattern in correlated_patterns:
        print(f"\nPattern Type: {pattern.pattern_type}")
        print(f"Strength: {pattern.strength:.2f}")
        print(f"Risk Score: {pattern.risk_score:.2f}")
        print("\nComponents:")
        print("  Access:", pattern.components["access"])
        print("  Permissions:", pattern.components["permission"])
        print("  Temporal:", pattern.components["temporal"])
        
    print("\nPattern Evolution:")
    print("-" * 50)
    
    for evolution_type, patterns in evolution.items():
        if patterns:
            print(f"\n{evolution_type.title()} Patterns:")
            for pattern in patterns:
                print(f"  - {pattern.pattern_type} "
                      f"(Risk: {pattern.risk_score:.2f})")

if __name__ == "__main__":
    main() 