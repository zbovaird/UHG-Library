"""
Heimdal1: UHG-Based Intrusion Detection System with 3D Interactive Visualization
=============================================================================

This implementation extends uhg_intrusion_detection4.py with interactive 3D visualization
capabilities while maintaining strict UHG principles.

Key Features:
1. Pure projective operations - no manifold concepts
2. Cross-ratio preservation in all operations
3. UHG-compliant message passing
4. Interactive 3D visualization of network patterns
5. Real-time threat visualization
6. Geometric pattern recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from tqdm import tqdm
import os
from typing import Tuple, Optional
from torch_geometric.data import Data

# Visualization imports
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import plotly.express as px
from plotly.subplots import make_subplots

# Import UHG components
from uhg.projective import ProjectiveUHG
from uhg.nn.models.sage import ProjectiveGraphSAGE
from uhg.nn.layers.sage import ProjectiveSAGEConv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU - Warning: Training may be slow")

# File paths
FILE_PATH = 'CIC_data.csv'
MODEL_SAVE_PATH = 'heimdal_ids_model.pth'
RESULTS_PATH = 'heimdal_results'
VIZ_PATH = 'heimdal_visualizations'

def create_visualization_dir():
    """Create directory for saving visualizations."""
    os.makedirs(VIZ_PATH, exist_ok=True)
    print(f"Visualization outputs will be saved to: {VIZ_PATH}")

def visualize_network_3d(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    title: str = "Network Traffic Patterns in 3D"
) -> go.Figure:
    """Create interactive 3D visualization of network patterns.
    
    Args:
        embeddings: Node embeddings from the model
        edge_index: Graph connectivity
        labels: Node labels (0 for benign, 1 for malicious)
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    print("Generating 3D visualization...")
    
    # Convert embeddings to numpy and reduce to 3D
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Use t-SNE for dimensionality reduction while preserving local structure
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    embeddings_3d = tsne.fit_transform(embeddings_np)
    
    # Create figure
    fig = go.Figure()
    
    # Add nodes
    node_colors = ['blue' if label == 0 else 'red' for label in labels.cpu().numpy()]
    
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=node_colors,
            opacity=0.8
        ),
        text=[f"Node {i}<br>{'Benign' if c == 'blue' else 'Malicious'}" 
              for i, c in enumerate(node_colors)],
        hoverinfo='text',
        name='Nodes'
    ))
    
    # Add edges
    edge_x = []
    edge_y = []
    edge_z = []
    edge_colors = []
    
    # Process edges in batches to handle large graphs
    batch_size = 1000
    edge_index_np = edge_index.cpu().numpy()
    
    for i in tqdm(range(0, edge_index.shape[1], batch_size), desc="Processing edges"):
        batch_end = min(i + batch_size, edge_index.shape[1])
        batch_edges = edge_index_np[:, i:batch_end]
        
        for src, dst in zip(batch_edges[0], batch_edges[1]):
            edge_x.extend([embeddings_3d[src, 0], embeddings_3d[dst, 0], None])
            edge_y.extend([embeddings_3d[src, 1], embeddings_3d[dst, 1], None])
            edge_z.extend([embeddings_3d[src, 2], embeddings_3d[dst, 2], None])
            
            # Color edge based on connected nodes
            if node_colors[src] == 'red' or node_colors[dst] == 'red':
                edge_colors.extend(['rgba(255, 0, 0, 0.2)'] * 3)  # Transparent red
            else:
                edge_colors.extend(['rgba(0, 0, 255, 0.2)'] * 3)  # Transparent blue
    
    fig.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(
            color=edge_colors,
            width=1
        ),
        hoverinfo='none',
        name='Connections'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

[Rest of uhg_intrusion_detection4.py code remains the same...]

def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor
) -> Tuple[float, float]:
    """Evaluate model and generate visualization."""
    acc, avg_confidence = original_evaluate(model, data, mask)
    
    # Generate visualization after evaluation
    with torch.no_grad():
        embeddings = model(data)
        fig = visualize_network_3d(
            embeddings=embeddings,
            edge_index=data.edge_index,
            labels=data.y,
            title=f"Network Traffic Patterns (Accuracy: {acc:.2%})"
        )
        
        # Save visualization
        fig.write_html(os.path.join(VIZ_PATH, 'network_patterns_3d.html'))
        print(f"3D visualization saved to {VIZ_PATH}/network_patterns_3d.html")
    
    return acc, avg_confidence

def main():
    """Main training function with visualization."""
    print("\nInitializing Heimdal Intrusion Detection System...")
    
    # Create visualization directory
    create_visualization_dir()
    
    # Rest of the main function remains the same...
    [Rest of the code remains the same...]

if __name__ == "__main__":
    main() 