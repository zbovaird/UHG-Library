"""Example of threat indicator correlation using UHG."""

from uhg.threat_indicators import (
    ThreatIndicator,
    ThreatIndicatorType,
    ThreatCorrelation
)

def main():
    """Run threat correlation example."""
    # Create some example threat indicators
    indicators = [
        # Network indicators
        ThreatIndicator(
            ThreatIndicatorType.NETWORK,
            "192.168.1.100",
            confidence=0.9,
            context={
                "port": 443,
                "protocol": 6,  # TCP
                "bytes_out": 1024
            }
        ),
        ThreatIndicator(
            ThreatIndicatorType.NETWORK,
            "evil.example.com",
            confidence=0.8,
            context={
                "dns_queries": 5,
                "ttl": 300,
                "response_size": 512
            }
        ),
        
        # System indicators
        ThreatIndicator(
            ThreatIndicatorType.SYSTEM,
            "malware.exe",
            confidence=0.95,
            context={
                "file_size": 2048,
                "entropy": 7.8,
                "imports": 24
            }
        ),
        ThreatIndicator(
            ThreatIndicatorType.SYSTEM,
            "HKEY_LOCAL_MACHINE\\Software\\Evil",
            confidence=0.85,
            context={
                "reg_type": 1,
                "data_size": 128,
                "persistence": 1
            }
        ),
        
        # Behavioral indicators
        ThreatIndicator(
            ThreatIndicatorType.BEHAVIOR,
            "process_injection",
            confidence=0.9,
            context={
                "target_pid": 1234,
                "memory_allocated": 4096,
                "api_calls": 15
            }
        ),
        ThreatIndicator(
            ThreatIndicatorType.BEHAVIOR,
            "network_scan",
            confidence=0.7,
            context={
                "scan_rate": 100,
                "ports_tried": 20,
                "duration": 30
            }
        ),
        
        # Payload indicators
        ThreatIndicator(
            ThreatIndicatorType.PAYLOAD,
            "shellcode_pattern",
            confidence=0.95,
            context={
                "pattern_size": 64,
                "xor_key": 0xFF,
                "obfuscation": 1
            }
        )
    ]
    
    # Initialize correlation engine
    correlation = ThreatCorrelation(
        feature_dim=8,
        num_heads=4
    )
    
    # Get correlation groups
    print("\nFinding correlated indicator groups...")
    groups = correlation.get_correlation_groups(indicators, threshold=0.7)
    
    for i, group in enumerate(groups):
        print(f"\nGroup {i + 1}:")
        for indicator in group:
            print(f"- {indicator.type}: {indicator.value} (confidence: {indicator.confidence})")
            
    # Analyze relationships
    print("\nAnalyzing indicator relationships...")
    relationships = correlation.analyze_indicator_relationships(indicators)
    
    # Print network-system relationships
    print("\nNetwork-System Relationships:")
    for ind1, ind2, score in relationships["network_system"]:
        print(f"- {ind1.value} <-> {ind2.value}: {score:.3f}")
        
    # Print network-behavior relationships
    print("\nNetwork-Behavior Relationships:")
    for ind1, ind2, score in relationships["network_behavior"]:
        print(f"- {ind1.value} <-> {ind2.value}: {score:.3f}")
        
    # Print system-behavior relationships
    print("\nSystem-Behavior Relationships:")
    for ind1, ind2, score in relationships["system_behavior"]:
        print(f"- {ind1.value} <-> {ind2.value}: {score:.3f}")
        
    # Print payload relationships
    print("\nPayload-Related Relationships:")
    for ind1, ind2, score in relationships["payload_related"]:
        print(f"- {ind1.value} <-> {ind2.value}: {score:.3f}")

if __name__ == "__main__":
    main() 