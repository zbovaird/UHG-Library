"""
High-level CUDA Operations for UHG.

This module provides optimized CUDA operations built on top of custom kernels,
implementing high-level UHG functionality with GPU acceleration.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .kernels import CUDAKernels

class UHGCUDAOps:
    """High-level CUDA operations for UHG."""
    
    def __init__(self):
        self.kernels = CUDAKernels()
        
    def cross_ratio(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute cross-ratio in projective space using CUDA acceleration.
        
        Args:
            p1, p2, p3, p4: Points in projective space
            eps: Small value for numerical stability
            
        Returns:
            Cross-ratio tensor
        """
        # Move tensors to CUDA if needed
        if not p1.is_cuda:
            p1 = p1.cuda()
        if not p2.is_cuda:
            p2 = p2.cuda()
        if not p3.is_cuda:
            p3 = p3.cuda()
        if not p4.is_cuda:
            p4 = p4.cuda()
            
        # Convert to float32 if needed
        if p1.dtype != torch.float32:
            p1 = p1.float()
        if p2.dtype != torch.float32:
            p2 = p2.float()
        if p3.dtype != torch.float32:
            p3 = p3.float()
        if p4.dtype != torch.float32:
            p4 = p4.float()
            
        # Compute cross-ratio using CUDA kernel
        return self.kernels.compute_cross_ratio(p1, p2, p3, p4)
        
    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention using optimized CUDA kernels.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask
            dropout_p: Dropout probability
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Move tensors to CUDA if needed
        if not query.is_cuda:
            query = query.cuda()
        if not key.is_cuda:
            key = key.cuda()
        if not value.is_cuda:
            value = value.cuda()
        if mask is not None and not mask.is_cuda:
            mask = mask.cuda()
            
        # Convert to float32 if needed
        if query.dtype != torch.float32:
            query = query.float()
        if key.dtype != torch.float32:
            key = key.float()
        if value.dtype != torch.float32:
            value = value.float()
            
        # Compute attention scores using CUDA kernel
        attention_weights = self.kernels.compute_attention_scores(query, key, mask)
        
        # Apply dropout if needed
        if dropout_p > 0.0 and self.training:
            attention_weights = F.dropout(attention_weights, p=dropout_p)
            
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
        
    def batch_cross_ratio(
        self,
        points: List[torch.Tensor],
        batch_size: int = 1024
    ) -> List[torch.Tensor]:
        """
        Compute cross-ratios for multiple sets of points in batches.
        
        Args:
            points: List of point tensors
            batch_size: Size of each batch
            
        Returns:
            List of cross-ratio tensors
        """
        results = []
        num_points = len(points[0])
        num_batches = (num_points + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            
            batch_points = [p[start_idx:end_idx] for p in points]
            batch_result = self.cross_ratio(*batch_points)
            results.append(batch_result)
            
        return torch.cat(results)
        
    def batch_attention(
        self,
        queries: List[torch.Tensor],
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
        batch_size: int = 32
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute attention for multiple sets of inputs in batches.
        
        Args:
            queries: List of query tensors
            keys: List of key tensors
            values: List of value tensors
            batch_size: Size of each batch
            
        Returns:
            List of (output, attention_weights) tuples
        """
        results = []
        num_sets = len(queries)
        num_batches = (num_sets + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_sets)
            
            batch_queries = queries[start_idx:end_idx]
            batch_keys = keys[start_idx:end_idx]
            batch_values = values[start_idx:end_idx]
            
            batch_output, batch_weights = self.attention(
                torch.cat(batch_queries),
                torch.cat(batch_keys),
                torch.cat(batch_values)
            )
            
            # Split results back into individual sets
            split_size = batch_queries[0].size(0)
            outputs = batch_output.split(split_size)
            weights = batch_weights.split(split_size)
            
            results.extend(zip(outputs, weights))
            
        return results 