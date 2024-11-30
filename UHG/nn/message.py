import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Callable
from ..manifolds.base import Manifold
from ..utils.cross_ratio import compute_cross_ratio, preserve_cross_ratio

class HyperbolicMessagePassing(nn.Module):
    """Message passing in hyperbolic space using UHG principles.
    
    All operations are performed directly in hyperbolic space without
    tangent space mappings. Cross-ratios are preserved throughout.
    
    Args:
        manifold (Manifold): The hyperbolic manifold to operate on
        aggr (str): The aggregation scheme ('add', 'mean', or 'max')
        flow (str): Message passing flow direction ('source_to_target' or 'target_to_source')
    """
    def __init__(
        self,
        manifold: Manifold,
        aggr: str = 'mean',
        flow: str = 'source_to_target'
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.aggr = aggr
        self.flow = flow
        
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Construct messages in hyperbolic space.
        
        Args:
            x_i: Features of target nodes
            x_j: Features of source nodes
            edge_attr: Optional edge features
            
        Returns:
            Messages in hyperbolic space
        """
        # Compute messages directly in hyperbolic space
        msg = self.manifold.mobius_add(x_j, -x_i)
        
        if edge_attr is not None:
            # Include edge features while preserving hyperbolic structure
            msg = self.manifold.mobius_matvec(edge_attr, msg)
            
        return msg
        
    def aggregate(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        dim_size: Optional[int] = None
    ) -> torch.Tensor:
        """Aggregate messages in hyperbolic space.
        
        Args:
            messages: Messages to aggregate
            edge_index: Graph connectivity
            dim_size: Output dimension size
            
        Returns:
            Aggregated messages in hyperbolic space
        """
        # Aggregate while preserving hyperbolic structure
        if self.aggr == 'add':
            return self._hyperbolic_add(messages, edge_index, dim_size)
        elif self.aggr == 'mean':
            return self._hyperbolic_mean(messages, edge_index, dim_size)
        elif self.aggr == 'max':
            return self._hyperbolic_max(messages, edge_index, dim_size)
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggr}")
            
    def _hyperbolic_add(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        dim_size: Optional[int] = None
    ) -> torch.Tensor:
        """Hyperbolic addition aggregation."""
        # Perform addition directly in hyperbolic space
        if dim_size is None:
            dim_size = edge_index[1].max().item() + 1
            
        out = torch.zeros((dim_size,) + messages.shape[1:],
                         device=messages.device)
                         
        index = edge_index[1] if self.flow == 'source_to_target' else edge_index[0]
        
        # Iterative hyperbolic addition to maintain structure
        for i in range(messages.shape[0]):
            out[index[i]] = self.manifold.mobius_add(
                out[index[i]], messages[i])
                
        return out
        
    def _hyperbolic_mean(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        dim_size: Optional[int] = None
    ) -> torch.Tensor:
        """Hyperbolic mean aggregation."""
        # First compute sum
        total = self._hyperbolic_add(messages, edge_index, dim_size)
        
        # Count number of messages per node
        if dim_size is None:
            dim_size = edge_index[1].max().item() + 1
            
        index = edge_index[1] if self.flow == 'source_to_target' else edge_index[0]
        ones = torch.ones(messages.shape[0], device=messages.device)
        count = torch.zeros(dim_size, device=messages.device)
        count.scatter_add_(0, index, ones)
        
        # Apply scaling in hyperbolic space
        count = count.clamp(min=1)
        scale = 1.0 / count.view(-1, 1)
        return self.manifold.mobius_scalar_mul(scale, total)
        
    def _hyperbolic_max(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        dim_size: Optional[int] = None
    ) -> torch.Tensor:
        """Hyperbolic max aggregation."""
        # Use hyperbolic distance to find maximum
        if dim_size is None:
            dim_size = edge_index[1].max().item() + 1
            
        index = edge_index[1] if self.flow == 'source_to_target' else edge_index[0]
        
        out = torch.zeros((dim_size,) + messages.shape[1:],
                         device=messages.device)
                         
        # For each node, find message with maximum hyperbolic norm
        for i in range(dim_size):
            mask = index == i
            if not mask.any():
                continue
                
            node_messages = messages[mask]
            norms = self.manifold.norm(node_messages)
            max_idx = torch.argmax(norms)
            out[i] = node_messages[max_idx]
            
        return out
        
    def update(
        self,
        aggr_out: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update node embeddings with aggregated messages.
        
        Args:
            aggr_out: Aggregated messages
            x: Optional input node features
            
        Returns:
            Updated node embeddings
        """
        if x is None:
            return aggr_out
            
        # Combine while preserving cross-ratio
        x_adj, aggr_adj = self.preserve_cross_ratio(x, aggr_out)
        return self.manifold.mobius_add(x_adj, aggr_adj)
        
    def preserve_cross_ratio(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure transformations preserve the cross-ratio.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tuple of tensors with preserved cross-ratio
        """
        cr_before = compute_cross_ratio(x, y)
        x_adj, y_adj = preserve_cross_ratio(x, y, cr_before)
        return x_adj, y_adj
        
    def propagate(
        self,
        edge_index: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """The main entry point for message passing.
        
        Args:
            edge_index: Graph connectivity
            x: Node features
            edge_attr: Optional edge features
            size: Optional output size
            
        Returns:
            Updated node embeddings
        """
        dim_size = size[1] if size is not None else None
        
        # Get source and target node features
        x_i = x[edge_index[1]] if x is not None else None
        x_j = x[edge_index[0]] if x is not None else None
        
        # Compute messages
        messages = self.message(x_i, x_j, edge_attr)
        
        # Aggregate messages
        aggr_out = self.aggregate(messages, edge_index, dim_size)
        
        # Update node embeddings
        return self.update(aggr_out, x) 