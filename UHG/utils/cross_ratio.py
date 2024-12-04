import torch
from ..projective import ProjectiveUHG

def compute_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-ratio of four points using projective geometry.
    
    The cross-ratio is the fundamental invariant in UHG and is computed
    directly using projective operations.
    
    Args:
        p1, p2, p3, p4: Points in projective space
        
    Returns:
        Cross-ratio value
    """
    uhg = ProjectiveUHG()
    return uhg.cross_ratio(p1, p2, p3, p4)

def preserve_cross_ratio(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor,
                        matrix: torch.Tensor) -> torch.Tensor:
    """
    Verify that a transformation preserves the cross-ratio.
    
    In UHG, all valid transformations must preserve the cross-ratio.
    This function checks that property.
    
    Args:
        p1, p2, p3, p4: Points in projective space
        matrix: Transformation matrix
        
    Returns:
        Boolean indicating if cross-ratio is preserved
    """
    uhg = ProjectiveUHG()
    
    # Compute original cross-ratio
    cr1 = uhg.cross_ratio(p1, p2, p3, p4)
    
    # Transform points
    p1_t = uhg.transform(p1, matrix)
    p2_t = uhg.transform(p2, matrix)
    p3_t = uhg.transform(p3, matrix)
    p4_t = uhg.transform(p4, matrix)
    
    # Compute transformed cross-ratio
    cr2 = uhg.cross_ratio(p1_t, p2_t, p3_t, p4_t)
    
    # Check if preserved (up to numerical precision)
    return torch.allclose(cr1, cr2)
