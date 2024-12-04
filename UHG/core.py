from typing import Tuple, Optional, Union, List
import torch
from .projective import ProjectiveUHG

class UHGCore:
    """
    Core operations in Universal Hyperbolic Geometry.
    
    All operations are implemented using pure projective geometry,
    following the principles in UHG.pdf.
    """
    
    def __init__(self):
        """Initialize UHG core operations."""
        self.uhg = ProjectiveUHG()
    
    def join(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute the join of two points in projective space.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Join line in projective coordinates
        """
        return self.uhg.join(p1, p2)
    
    def meet(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """
        Compute the meet of two lines in projective space.
        
        Args:
            l1: First line
            l2: Second line
            
        Returns:
            Meet point in projective coordinates
        """
        return self.uhg.meet(l1, l2)
    
    def cross_ratio(self, p1: torch.Tensor, p2: torch.Tensor, 
                   p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-ratio of four points.
        
        This is the fundamental invariant in UHG.
        
        Args:
            p1, p2, p3, p4: Points in projective space
            
        Returns:
            Cross-ratio value
        """
        return self.uhg.cross_ratio(p1, p2, p3, p4)
    
    def perpendicular(self, p: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """
        Compute the perpendicular from a point to a line.
        
        Uses the absolute polar operation from UHG.pdf.
        
        Args:
            p: Point in projective space
            l: Line in projective space
            
        Returns:
            Perpendicular line in projective coordinates
        """
        # Use polar to compute perpendicular
        polar = self.uhg.absolute_polar(p)
        return self.uhg.meet(polar, l)
    
    def reflect(self, p: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """
        Reflect a point in a line using projective geometry.
        
        Args:
            p: Point to reflect
            l: Line to reflect in
            
        Returns:
            Reflected point in projective coordinates
        """
        return self.uhg.reflect(p, l)
    
    def transform(self, points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply a projective transformation to points.
        
        All transformations preserve the cross-ratio.
        
        Args:
            points: Points to transform
            matrix: Projective transformation matrix
            
        Returns:
            Transformed points
        """
        return self.uhg.transform(points, matrix)
    
    def distance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic distance between points.
        
        Uses the cross-ratio formula from UHG.pdf:
            d(p1,p2) = |log(CR(p1,p2,i1,i2))|
        where i1,i2 are the ideal points on the line through p1,p2.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Hyperbolic distance
        """
        # Get line through points
        line = self.uhg.join(p1, p2)
        
        # Get ideal points using polar
        polar = self.uhg.absolute_polar(line)
        i1 = self.uhg.meet(line, polar)
        i2 = -i1  # Opposite point on absolute
        
        # Compute distance using cross-ratio
        cr = self.uhg.cross_ratio(p1, p2, i1, i2)
        return torch.abs(torch.log(cr))