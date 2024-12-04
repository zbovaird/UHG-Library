import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Union
from ..projective import ProjectiveUHG

class ProjectiveDataset(Dataset):
    """Base class for datasets in projective space.
    
    All data points are represented using projective geometry,
    following UHG principles.
    
    Args:
        points: Data points in projective coordinates
        labels: Optional labels for the points
    """
    def __init__(
        self,
        points: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        self.uhg = ProjectiveUHG()
        self.points = points
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.points)
        
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.labels is None:
            return self.points[idx]
        return self.points[idx], self.labels[idx]
        
    def to(self, device: torch.device) -> 'ProjectiveDataset':
        """Move dataset to device."""
        points = self.points.to(device)
        labels = self.labels.to(device) if self.labels is not None else None
        return ProjectiveDataset(points, labels)
        
    def compute_cross_ratios(
        self,
        idx1: int,
        idx2: int,
        idx3: int,
        idx4: int
    ) -> torch.Tensor:
        """Compute cross-ratio between four points.
        
        Args:
            idx1, idx2, idx3, idx4: Indices of points
            
        Returns:
            Cross-ratio value
        """
        p1 = self.points[idx1]
        p2 = self.points[idx2]
        p3 = self.points[idx3]
        p4 = self.points[idx4]
        
        return self.uhg.cross_ratio(p1, p2, p3, p4)
        
    def get_neighborhood(
        self,
        idx: int,
        radius: float
    ) -> List[int]:
        """Get indices of points within projective distance.
        
        Uses cross-ratio to measure distances.
        
        Args:
            idx: Center point index
            radius: Maximum distance
            
        Returns:
            List of neighbor indices
        """
        center = self.points[idx]
        distances = torch.tensor([
            self.uhg.proj_dist(center, p)
            for p in self.points
        ])
        return torch.where(distances <= radius)[0].tolist()
        
    def sample_triplet(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a triplet of points preserving projective structure.
        
        Uses cross-ratio to determine similarity.
        
        Returns:
            Anchor, positive, and negative points
        """
        # Sample random anchor
        anchor_idx = torch.randint(len(self), (1,)).item()
        anchor = self.points[anchor_idx]
        
        # Sample positive point (close in cross-ratio)
        pos_dists = torch.tensor([
            self.uhg.proj_dist(anchor, p)
            for p in self.points
        ])
        pos_probs = torch.softmax(-pos_dists, dim=0)
        pos_idx = torch.multinomial(pos_probs, 1).item()
        positive = self.points[pos_idx]
        
        # Sample negative point (far in cross-ratio)
        neg_probs = torch.softmax(pos_dists, dim=0)
        neg_idx = torch.multinomial(neg_probs, 1).item()
        negative = self.points[neg_idx]
        
        return anchor, positive, negative 