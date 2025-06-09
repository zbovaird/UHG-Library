import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple
import sys
import os

# Add the parent directory to the path so we can import the UHG library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uhg.utils.metrics import (
    uhg_inner_product,
    uhg_norm,
    uhg_quadrance,
    uhg_spread
)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running benchmarks on: {DEVICE}")

# Batch sizes to test
BATCH_SIZES = [1, 10, 100, 1000, 10000, 100000]
FEATURE_DIM = 16  # Fixed feature dimension for testing

def benchmark_function(
    func: Callable,
    batch_sizes: List[int],
    feature_dim: int,
    num_runs: int = 5
) -> Dict[str, List[float]]:
    """
    Benchmark a function with different batch sizes.
    
    Args:
        func: Function to benchmark
        batch_sizes: List of batch sizes to test
        feature_dim: Feature dimension
        num_runs: Number of runs for each batch size
        
    Returns:
        Dictionary with vectorized and loop times
    """
    vectorized_times = []
    loop_times = []
    
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size: {batch_size}")
        
        # Create random tensors
        a = torch.randn(batch_size, feature_dim + 1, device=DEVICE)
        b = torch.randn(batch_size, feature_dim + 1, device=DEVICE)
        
        # Benchmark vectorized implementation
        vectorized_time = 0
        for _ in range(num_runs):
            torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
            start = time.time()
            _ = func(a, b)
            torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
            vectorized_time += (time.time() - start)
        vectorized_time /= num_runs
        vectorized_times.append(vectorized_time)
        
        # Benchmark loop implementation
        loop_time = 0
        for _ in range(num_runs):
            torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
            start = time.time()
            result = torch.zeros(batch_size, device=DEVICE)
            for i in range(batch_size):
                result[i] = func(a[i:i+1], b[i:i+1]).squeeze()
            torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
            loop_time += (time.time() - start)
        loop_time /= num_runs
        loop_times.append(loop_time)
        
        # Print speedup
        speedup = loop_time / vectorized_time
        print(f"  Vectorized: {vectorized_time:.6f}s, Loop: {loop_time:.6f}s, Speedup: {speedup:.2f}x")
    
    return {
        "vectorized": vectorized_times,
        "loop": loop_times
    }

def plot_results(
    results: Dict[str, Dict[str, List[float]]],
    batch_sizes: List[int],
    save_path: str = None
):
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary with function names and timing results
        batch_sizes: List of batch sizes used
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(12, 10))
    
    # Plot execution times
    plt.subplot(2, 1, 1)
    for func_name, times in results.items():
        plt.plot(batch_sizes, times["vectorized"], 'o-', label=f"{func_name} (Vectorized)")
        plt.plot(batch_sizes, times["loop"], 's--', label=f"{func_name} (Loop)")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Batch Size')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Plot speedup
    plt.subplot(2, 1, 2)
    for func_name, times in results.items():
        speedups = [loop / vec for vec, loop in zip(times["vectorized"], times["loop"])]
        plt.plot(batch_sizes, speedups, 'o-', label=func_name)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor (Loop / Vectorized)')
    plt.title('Speedup Factor vs Batch Size')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    """Run benchmarks for core operations."""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Benchmark inner product
    print("\nBenchmarking uhg_inner_product...")
    results["uhg_inner_product"] = benchmark_function(
        uhg_inner_product, BATCH_SIZES, FEATURE_DIM
    )
    
    # Benchmark quadrance
    print("\nBenchmarking uhg_quadrance...")
    results["uhg_quadrance"] = benchmark_function(
        uhg_quadrance, BATCH_SIZES, FEATURE_DIM
    )
    
    # Benchmark spread
    print("\nBenchmarking uhg_spread...")
    results["uhg_spread"] = benchmark_function(
        uhg_spread, BATCH_SIZES, FEATURE_DIM
    )
    
    # Plot and save results
    plot_results(results, BATCH_SIZES, save_path="results/core_ops_benchmark.png")
    
    # Print summary
    print("\nSummary of Speedup Factors:")
    for func_name, times in results.items():
        speedups = [loop / vec for vec, loop in zip(times["vectorized"], times["loop"])]
        max_speedup = max(speedups)
        max_speedup_batch = BATCH_SIZES[speedups.index(max_speedup)]
        print(f"{func_name}: Max speedup {max_speedup:.2f}x at batch size {max_speedup_batch}")

if __name__ == "__main__":
    main() 