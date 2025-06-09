"""Tests for CUDA-optimized UHG operations."""

import pytest
import torch
import numpy as np
from uhg.cuda.ops import UHGCUDAOps
from uhg.projective import ProjectiveUHG

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUHGCUDAOps:
    """Test suite for CUDA-optimized UHG operations."""
    
    @pytest.fixture
    def cuda_ops(self):
        """Initialize CUDA operations."""
        return UHGCUDAOps()
        
    @pytest.fixture
    def uhg(self):
        """Initialize ProjectiveUHG for comparison."""
        return ProjectiveUHG()
        
    def test_cross_ratio_computation(self, cuda_ops, uhg):
        """Test cross-ratio computation using CUDA kernels."""
        # Create test points
        p1 = torch.randn(100, 3, device='cuda')
        p2 = torch.randn(100, 3, device='cuda')
        p3 = torch.randn(100, 3, device='cuda')
        p4 = torch.randn(100, 3, device='cuda')
        
        # Compute cross-ratio using both methods
        cuda_result = cuda_ops.cross_ratio(p1, p2, p3, p4)
        cpu_result = uhg.cross_ratio(p1.cpu(), p2.cpu(), p3.cpu(), p4.cpu())
        
        # Compare results
        assert torch.allclose(cuda_result.cpu(), cpu_result, rtol=1e-4)
        
    def test_attention_computation(self, cuda_ops):
        """Test attention computation using CUDA kernels."""
        # Create test inputs
        batch_size = 8
        num_heads = 4
        seq_len = 16
        head_dim = 32
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Compute attention
        output, weights = cuda_ops.attention(query, key, value)
        
        # Check shapes
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
        
    def test_batch_cross_ratio(self, cuda_ops):
        """Test batched cross-ratio computation."""
        # Create test points
        num_points = 1000
        points = [
            torch.randn(num_points, 3, device='cuda')
            for _ in range(4)
        ]
        
        # Compute cross-ratios in batches
        batch_result = cuda_ops.batch_cross_ratio(points)
        
        # Compute cross-ratios directly
        direct_result = cuda_ops.cross_ratio(
            points[0], points[1], points[2], points[3]
        )
        
        # Compare results
        assert torch.allclose(batch_result, direct_result, rtol=1e-4)
        
    def test_batch_attention(self, cuda_ops):
        """Test batched attention computation."""
        # Create test inputs
        num_sets = 10
        batch_size = 8
        num_heads = 4
        seq_len = 16
        head_dim = 32
        
        queries = [
            torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            for _ in range(num_sets)
        ]
        keys = [
            torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            for _ in range(num_sets)
        ]
        values = [
            torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            for _ in range(num_sets)
        ]
        
        # Compute attention in batches
        batch_results = cuda_ops.batch_attention(queries, keys, values)
        
        # Check results
        assert len(batch_results) == num_sets
        for output, weights in batch_results:
            assert output.shape == (batch_size, num_heads, seq_len, head_dim)
            assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
            assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
            
    def test_performance_comparison(self, cuda_ops, uhg):
        """Compare performance between CPU and GPU implementations."""
        # Create large test inputs
        num_points = 10000
        p1 = torch.randn(num_points, 3)
        p2 = torch.randn(num_points, 3)
        p3 = torch.randn(num_points, 3)
        p4 = torch.randn(num_points, 3)
        
        # Time CPU computation
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        cpu_result = uhg.cross_ratio(p1, p2, p3, p4)
        end_time.record()
        torch.cuda.synchronize()
        cpu_time = start_time.elapsed_time(end_time)
        
        # Time GPU computation
        p1_gpu = p1.cuda()
        p2_gpu = p2.cuda()
        p3_gpu = p3.cuda()
        p4_gpu = p4.cuda()
        
        start_time.record()
        gpu_result = cuda_ops.cross_ratio(p1_gpu, p2_gpu, p3_gpu, p4_gpu)
        end_time.record()
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time)
        
        # Compare results and timing
        assert torch.allclose(gpu_result.cpu(), cpu_result, rtol=1e-4)
        assert gpu_time < cpu_time, f"GPU time ({gpu_time}ms) should be less than CPU time ({cpu_time}ms)"
        
    def test_numerical_stability(self, cuda_ops):
        """Test numerical stability of CUDA operations."""
        # Test with very small and large values
        scales = [1e-6, 1.0, 1e6]
        results = []
        
        for scale in scales:
            p1 = torch.randn(100, 3, device='cuda') * scale
            p2 = torch.randn(100, 3, device='cuda') * scale
            p3 = torch.randn(100, 3, device='cuda') * scale
            p4 = torch.randn(100, 3, device='cuda') * scale
            
            result = cuda_ops.cross_ratio(p1, p2, p3, p4)
            results.append(result)
            
            # Check for NaN or inf
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
            
        # Results should be scale-invariant
        for r1, r2 in zip(results[:-1], results[1:]):
            assert torch.allclose(r1, r2, rtol=1e-3)
            
    def test_memory_efficiency(self, cuda_ops):
        """Test memory efficiency of CUDA operations."""
        # Record initial memory usage
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.max_memory_allocated()
        
        # Process large batch
        batch_size = 1024
        num_heads = 8
        seq_len = 128
        head_dim = 64
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Process in smaller batches
        output, weights = cuda_ops.attention(query, key, value)
        
        # Check memory usage
        final_memory = torch.cuda.max_memory_allocated()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 2x input size)
        total_input_size = (query.numel() + key.numel() + value.numel()) * 4  # float32 = 4 bytes
        assert memory_increase < 2 * total_input_size 