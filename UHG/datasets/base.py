import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Union
from ..manifolds.base import Manifold

class HyperbolicDataset(Dataset):
    """Base class for datasets in hyperbolic space.
    
    All data points are represented directly in hyperbolic space
    using UHG principles, without tangent space mappings.
    
    Args:
        manifold: The hyperbolic manifold for the data
        points: Data points in hyperbolic space
        labels: Optional labels for the points
    """
    def __init__(
        self,
        manifold: Manifold,
        points: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        self.manifold = manifold
        self.points = self.manifold.normalize(points)
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.points)
        
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.labels is None:
            return self.points[idx]
        return self.points[idx], self.labels[idx]
        
    def to(self, device: torch.device) -> 'HyperbolicDataset':
        """Move dataset to device."""
        points = self.points.to(device)
        labels = self.labels.to(device) if self.labels is not None else None
        return HyperbolicDataset(self.manifold, points, labels)
        
    @classmethod
    def from_euclidean(
        cls,
        manifold: Manifold,
        points: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> 'HyperbolicDataset':
        """Create dataset from Euclidean points.
        
        Points are projected onto the hyperbolic manifold.
        
        Args:
            manifold: Target hyperbolic manifold
            points: Euclidean points
            labels: Optional labels
            
        Returns:
            HyperbolicDataset
        """
        # Project points onto manifold
        hyperbolic_points = manifold.project_from_euclidean(points)
        return cls(manifold, hyperbolic_points, labels)
        
    def compute_distances(self) -> torch.Tensor:
        """Compute pairwise distances between all points.
        
        Returns:
            Distance matrix
        """
        n = len(self)
        distances = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.manifold.dist(self.points[i], self.points[j])
                distances[i,j] = dist
                distances[j,i] = dist
                
        return distances
        
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
        
        return self.manifold.compute_cross_ratio(p1, p2, p3, p4)
        
    def get_neighborhood(
        self,
        idx: int,
        radius: float
    ) -> List[int]:
        """Get indices of points within hyperbolic distance.
        
        Args:
            idx: Center point index
            radius: Maximum distance
            
        Returns:
            List of neighbor indices
        """
        center = self.points[idx]
        distances = torch.tensor([
            self.manifold.dist(center, p)
            for p in self.points
        ])
        return torch.where(distances <= radius)[0].tolist()
        
    def sample_triplet(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a triplet of points preserving hyperbolic structure.
        
        Returns:
            Anchor, positive, and negative points
        """
        # Sample random anchor
        anchor_idx = torch.randint(len(self), (1,)).item()
        anchor = self.points[anchor_idx]
        
        # Sample positive point (close to anchor)
        pos_dists = torch.tensor([
            self.manifold.dist(anchor, p)
            for p in self.points
        ])
        pos_probs = torch.softmax(-pos_dists, dim=0)
        pos_idx = torch.multinomial(pos_probs, 1).item()
        positive = self.points[pos_idx]
        
        # Sample negative point (far from anchor)
        neg_probs = torch.softmax(pos_dists, dim=0)
        neg_idx = torch.multinomial(neg_probs, 1).item()
        negative = self.points[neg_idx]
        
        return anchor, positive, negative 