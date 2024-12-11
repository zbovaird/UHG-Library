"""Batch handling utilities for UHG."""

import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Batch, Data
from ..projective import ProjectiveUHG

class UHGBatch:
    """Batch handling for UHG with cross-ratio preservation."""
    
    def __init__(self):
        self.uhg = ProjectiveUHG()
        
    def to_batch(
        self,
        data_list: List[Data],
        preserve_cr: bool = True
    ) -> Batch:
        """Convert list of Data objects to a batch while preserving cross-ratios.
        
        Args:
            data_list: List of PyG Data objects
            preserve_cr: Whether to preserve cross-ratios
            
        Returns:
            Batched data
        """
        # First apply standard PyG batching
        batch = Batch.from_data_list(data_list)
        
        if preserve_cr and len(data_list) > 0:
            # Store cross-ratios for each graph
            batch.cross_ratios = []
            batch.graph_sizes = []
            
            start_idx = 0
            for data in data_list:
                size = data.x.size(0)
                batch.graph_sizes.append(size)
                
                if size > 3:
                    # Compute cross-ratio for this graph
                    cr = self.uhg.cross_ratio(
                        data.x[0],
                        data.x[1],
                        data.x[2],
                        data.x[3]
                    )
                    batch.cross_ratios.append(cr)
                else:
                    batch.cross_ratios.append(None)
                    
                start_idx += size
                
        return batch
        
    def unbatch(
        self,
        batch: Batch,
        restore_cr: bool = True
    ) -> List[Data]:
        """Convert batch back to list of Data objects.
        
        Args:
            batch: Batched data
            restore_cr: Whether to restore original cross-ratios
            
        Returns:
            List of unbatched Data objects
        """
        # First apply standard PyG unbatching
        data_list = batch.to_data_list()
        
        if restore_cr and hasattr(batch, 'cross_ratios'):
            # Restore cross-ratios for each graph
            for i, (data, cr) in enumerate(zip(data_list, batch.cross_ratios)):
                if cr is not None and data.x.size(0) > 3:
                    # Compute current cross-ratio
                    cr_current = self.uhg.cross_ratio(
                        data.x[0],
                        data.x[1],
                        data.x[2],
                        data.x[3]
                    )
                    
                    if not torch.isnan(cr_current) and not torch.isnan(cr) and cr_current != 0:
                        # Compute scale factor in log space
                        log_scale = 0.5 * (torch.log(cr + 1e-8) - torch.log(cr_current + 1e-8))
                        scale = torch.exp(log_scale)
                        
                        # Apply scale to features
                        features = data.x[..., :-1] * scale
                        
                        # Re-normalize
                        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
                        features = features / (norm + 1e-8)
                        
                        # Update features
                        data.x = torch.cat([features, data.x[..., -1:]], dim=-1)
                        
        return data_list
        
    def get_graph_sizes(self, batch: Batch) -> List[int]:
        """Get sizes of individual graphs in batch."""
        if hasattr(batch, 'graph_sizes'):
            return batch.graph_sizes
        else:
            # Compute from batch assignment
            _, counts = torch.unique(batch.batch, return_counts=True)
            return counts.tolist()
            
    def get_batch_assignment(self, batch: Batch) -> torch.Tensor:
        """Get tensor assigning nodes to their graphs in batch."""
        return batch.batch
        
    def get_graph_mask(self, batch: Batch, graph_idx: int) -> torch.Tensor:
        """Get boolean mask for nodes in specific graph."""
        return batch.batch == graph_idx 