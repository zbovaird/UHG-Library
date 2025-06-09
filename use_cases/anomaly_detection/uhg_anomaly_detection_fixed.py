#!/usr/bin/env python3
"""
UHG-Based Anomaly Detection System

This implementation uses Universal Hyperbolic Geometry principles to create
an effective anomaly detection system for security log analysis.
"""

# ==============================
# 1. Install Required Packages
# ==============================

# For Google Colab Users:
# Copy and run this cell first to install dependencies
'''
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install tqdm geoopt matplotlib scikit-learn
pip install torch-geometric
pip install uhg>=0.3.2
'''

# ==============================
# 2. Import Libraries
# ==============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
import scipy.sparse
import os
import sys
import time
from torch import Tensor
import matplotlib.pyplot as plt
import geoopt.optim
from torch_geometric.data import Data
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with 'pip install umap-learn' for additional visualization options.")

# Import UHG library
try:
    from uhg import ProjectiveUHG
    _UHG = ProjectiveUHG(epsilon=1e-9)
    UHG_AVAILABLE = True
    print(f"UHG library loaded successfully. Version: {getattr(ProjectiveUHG, '__version__', 'unknown')}")
except ImportError:
    print("Warning: UHG library not found. Using fallback implementations.")
    UHG_AVAILABLE = False
    _UHG = None

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================
# 3. Define Helper Functions
# ==============================

def scatter_mean_custom(src, index, dim=0, dim_size=None):
    """
    Custom implementation of scatter_mean that handles edge cases better.
    
    Args:
        src (Tensor): The source tensor to scatter
        index (Tensor): The indices where to scatter the source tensor
        dim (int): The dimension along which to scatter
        dim_size (int): The size of the output tensor at dimension dim
        
    Returns:
        Tensor: The scattered tensor with mean aggregation
    """
    print(f"Scatter input shapes - src: {src.shape}, index: {index.shape}")
    
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output tensor filled with zeros
    out = torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device)
    
    # Count how many values are scattered to each index
    ones = torch.ones(index.size(0), device=src.device)
    count = torch.zeros(dim_size, device=src.device)
    count.scatter_add_(0, index, ones)
    
    # Add up all values that go to the same index
    out.scatter_add_(0, index.view(-1, 1).expand(-1, src.size(1)), src)
    
    # Avoid division by zero
    count = torch.max(count, torch.ones_like(count))
    
    # Compute mean by dividing by count
    out = out / count.view(-1, 1)
    
    print(f"Scatter output shape: {out.shape}")
    return out

class GraphSAGE_CustomScatter(nn.Module):
    def __init__(self, in_channels, out_channels, append_uhg=True):
        super(GraphSAGE_CustomScatter, self).__init__()
        self.append_uhg = append_uhg
        self.lin1 = nn.Linear(in_channels, out_channels)
        # Add a decoder layer to map back to the original input dimension
        self.lin_decoder = nn.Linear(out_channels * 2, in_channels)  # *2 because we concatenate
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin_decoder.reset_parameters()
        
    def forward(self, x, edge_index):
        # Print shapes for debugging
        print(f"Input x shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        
        # First layer
        x = self.lin1(x)
        print(f"After linear layer shape: {x.shape}")
        
        # Aggregate neighbors
        row, col = edge_index
        out = scatter_mean_custom(x[col], row, dim=0, dim_size=x.size(0))
        print(f"After aggregation shape: {out.shape}")
        
        # Combine with self features
        out = torch.cat([x, out], dim=1)
        print(f"After concatenation shape: {out.shape}")
        
        # Apply non-linearity
        out = F.relu(out)
        
        # Store the embeddings before decoding (for anomaly detection)
        self.embeddings = out
        
        # Decode back to original input dimension
        decoded = self.lin_decoder(out)
        print(f"After decoder shape: {decoded.shape}")
        
        return decoded

class HGNN_CustomScatter(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Create the network layers
        self.convs = nn.ModuleList()
        # First layer: in_channels -> hidden_channels
        self.convs.append(GraphSAGE_CustomScatter(in_channels, hidden_channels, append_uhg=True))
        
        # Hidden layers: hidden_channels+1 -> hidden_channels (accounting for homogeneous coordinate)
        for _ in range(num_layers - 2):
            self.convs.append(GraphSAGE_CustomScatter(hidden_channels+1, hidden_channels, append_uhg=True))
        
        # Last layer: hidden_channels+1 -> out_channels (no homogeneous coordinate in output)
        self.convs.append(GraphSAGE_CustomScatter(hidden_channels+1, out_channels, append_uhg=False))
    
    def forward(self, x, edge_index):
        # Print initial shape
        print(f"HGNN input shape: {x.shape}")
        
        # Process through layers
        for i, conv in enumerate(self.convs):
            print(f"Layer {i+1}/{self.num_layers}")
            x = conv(x, edge_index)
            print(f"Output shape after layer {i+1}: {x.shape}")
        
        return x

def to_uhg_space(x):
    """
    Convert features to UHG space by appending a homogeneous coordinate.
    
    Args:
        x: Input features as numpy.ndarray or torch.Tensor of shape [N, D]
        
    Returns:
        Same type as input: Features in UHG space of shape [N, D+1]
    """
    print(f"Converting to UHG space. Input type: {type(x)}, shape: {x.shape}")
    
    # Convert numpy array to torch tensor if needed
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x_tensor = torch.from_numpy(x).float()
    else:
        x_tensor = x
    
    # Normalize the features to have unit norm
    norm = torch.norm(x_tensor, p=2, dim=1, keepdim=True)
    # Avoid division by zero
    norm = torch.max(norm, torch.ones_like(norm) * 1e-8)
    x_normalized = x_tensor / norm
    
    # Append homogeneous coordinate (1.0) to make it compatible with UHG operations
    homogeneous_coord = torch.ones(x_tensor.size(0), 1, device=x_tensor.device)
    x_uhg = torch.cat([x_normalized, homogeneous_coord], dim=1)
    
    print(f"UHG space output shape: {x_uhg.shape}")
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        return x_uhg.numpy()
    else:
        return x_uhg

def uhg_inner_product(a, b):
    """
    Compute the UHG inner product between two vectors.
    
    Args:
        a: First vector as numpy.ndarray or torch.Tensor
        b: Second vector as numpy.ndarray or torch.Tensor
        
    Returns:
        Same type as input: UHG inner product
    """
    # Convert numpy arrays to torch tensors if needed
    is_numpy = isinstance(a, np.ndarray)
    if is_numpy:
        a_tensor = torch.from_numpy(a).float()
        b_tensor = torch.from_numpy(b).float()
    else:
        a_tensor = a
        b_tensor = b
    
    # Compute inner product
    result = torch.sum(a_tensor * b_tensor, dim=-1)
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        return result.numpy()
    else:
        return result

def uhg_norm(a):
    """
    Compute the UHG norm of a point.
    
    Args:
        a: Point tensor of shape (..., D+1)
        
    Returns:
        Norm tensor of shape (...)
    """
    if UHG_AVAILABLE:
        try:
            # Use UHG inner product with itself
            return _UHG.hyperbolic_dot(a, a)
        except Exception as e:
            print(f"Warning: UHG library call failed: {e}. Using fallback implementation.")
    
    # Fallback implementation
    return uhg_inner_product(a, a)

def uhg_quadrance(a, b, eps=1e-9):
    """
    Compute the UHG quadrance (squared distance) between two points.
    
    Args:
        a: First point tensor of shape (..., D+1)
        b: Second point tensor of shape (..., D+1)
        eps: Small value for numerical stability
        
    Returns:
        Quadrance tensor of shape (...)
    """
    if UHG_AVAILABLE:
        try:
            return _UHG.quadrance(a, b)
        except Exception as e:
            print(f"Warning: UHG library call failed: {e}. Using fallback implementation.")
    
    # Fallback implementation
    dot_product = uhg_inner_product(a, b)
    norm_a = uhg_norm(a)
    norm_b = uhg_norm(b)
    denom = norm_a * norm_b
    denom = torch.clamp(denom.abs(), min=eps)
    quadrance = 1 - (dot_product ** 2) / denom
    return quadrance

def uhg_spread(L, M, eps=1e-9):
    """
    Compute the UHG spread between two lines.
    
    Args:
        L: First line as numpy.ndarray or torch.Tensor
        M: Second line as numpy.ndarray or torch.Tensor
        eps: Small value for numerical stability
        
    Returns:
        Same type as input: UHG spread
    """
    # Convert numpy arrays to torch tensors if needed
    is_numpy = isinstance(L, np.ndarray)
    if is_numpy:
        L_tensor = torch.from_numpy(L).float()
        M_tensor = torch.from_numpy(M).float()
    else:
        L_tensor = L
        M_tensor = M
    
    # Compute inner products
    LL = uhg_inner_product(L_tensor, L_tensor)
    MM = uhg_inner_product(M_tensor, M_tensor)
    LM = uhg_inner_product(L_tensor, M_tensor)
    
    # Compute spread
    numerator = LM * LM
    denominator = LL * MM
    
    # Handle numerical stability
    safe_denom = torch.clamp(denominator, min=eps)
    result = numerator / safe_denom
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        return result.numpy()
    else:
        return result

# Include anomaly detection at the beginning of your script
torch.autograd.set_detect_anomaly(True)

# ==============================
# 7. Mount Google Drive (If Using Google Colab)
# ==============================

from google.colab import drive
drive.mount('/content/drive')

# ==============================
# 8. Device Configuration
# ==============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==============================
# 9. Data Loading and Preprocessing
# ==============================

train_file = '/content/drive/MyDrive/modbus/train_data_balanced_new.csv'
val_file = '/content/drive/MyDrive/modbus/val_data_balanced_new.csv'
test_file = '/content/drive/MyDrive/modbus/test_data_balanced_new.csv'

train_data_full = pd.read_csv(train_file, low_memory=False)
val_data_full = pd.read_csv(val_file, low_memory=False)
test_data_full = pd.read_csv(test_file, low_memory=False)

all_data = pd.concat([train_data_full, val_data_full, test_data_full], ignore_index=True)
print(f'Combined data shape: {all_data.shape}')

all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
print('Data shuffled.')

data_percentage = 0.20
all_data = all_data.sample(frac=data_percentage, random_state=42).reset_index(drop=True)
print(f'Data reduced to {data_percentage*100}%: {all_data.shape}')

all_data.fillna(all_data.mean(), inplace=True)

non_numeric = all_data.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    all_data = pd.get_dummies(all_data, columns=non_numeric)

print(f'Feature dimensions after encoding: {all_data.shape[1]}')

scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_data)
print(f'Features scaled. Shape: {all_scaled.shape}')

n_components = min(50, all_scaled.shape[1])
if n_components < all_scaled.shape[1]:
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=n_components)
    all_scaled = pca.fit_transform(all_scaled)
    print(f'PCA applied. Reduced to {n_components} components. Shape: {all_scaled.shape}')
else:
    print("PCA not required. Using all features.")

total_samples = len(all_scaled)
train_samples = int(0.70 * total_samples)
val_samples = int(0.15 * total_samples)

train_data_np = all_scaled[:train_samples]
val_data_np = all_scaled[train_samples:train_samples + val_samples]
test_data_np = all_scaled[train_samples + val_samples:]

train_uhg = to_uhg_space(train_data_np)
val_uhg = to_uhg_space(val_data_np)
test_uhg = to_uhg_space(test_data_np)

# ==============================
# 10. Create KNN Graphs
# ==============================

def create_knn_graph(data, k, batch_size=1000):
    """
    Create a k-nearest neighbors graph from data points.
    
    Args:
        data: Input data points
        k: Number of nearest neighbors
        batch_size: Batch size for processing
        
    Returns:
        KNN graph as a sparse matrix
    """
    n_samples = data.shape[0]
    n_batches = (n_samples - 1) // batch_size + 1
    knn_graphs = []
    
    # Process in batches to handle large datasets
    with tqdm(total=n_batches, desc="Creating KNN graph") as pbar:
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            
            # Use UHG distance metric if possible
            try:
                # Try to use UHG quadrance for distance calculation
                # This would require custom implementation with the UHG library
                batch_knn = kneighbors_graph(
                    X=data, 
                    n_neighbors=k, 
                    mode='connectivity', 
                    include_self=False, 
                    n_jobs=-1
                )[i:end]
            except Exception:
                # Fall back to Euclidean distance
                batch_knn = kneighbors_graph(
                    X=data, 
                    n_neighbors=k, 
                    mode='connectivity', 
                    include_self=False, 
                    n_jobs=-1
                )[i:end]
                
            knn_graphs.append(batch_knn)
            pbar.update(1)
    
    # Combine all batches
    full_knn = scipy.sparse.vstack(knn_graphs)
    return full_knn

k = 2  # Setting k to 2 as requested
print("Creating KNN graph for training data...")
train_knn_graph = create_knn_graph(train_data_np, k)
print("Creating KNN graph for validation data...")
val_knn_graph = create_knn_graph(val_data_np, k)
print("Creating KNN graph for test data...")
test_knn_graph = create_knn_graph(test_data_np, k)

# ==============================
# 11. Convert KNN Graphs to Edge Lists
# ==============================

def knn_to_edge_index(knn_graph):
    """
    Convert a k-nearest neighbors graph from scipy sparse matrix to PyTorch edge_index format.
    
    Args:
        knn_graph: KNN graph as a scipy sparse matrix
        
    Returns:
        edge_index: Graph connectivity in PyTorch format (2 x num_edges)
    """
    # Convert to COO format for easy extraction of indices
    knn_coo = knn_graph.tocoo()
    
    # Create edge_index tensor [2, num_edges] with source nodes in first row and target nodes in second row
    edge_index = torch.tensor([knn_coo.row, knn_coo.col], dtype=torch.long)
    
    return edge_index

train_edge_index = knn_to_edge_index(train_knn_graph)
val_edge_index = knn_to_edge_index(val_knn_graph)
test_edge_index = knn_to_edge_index(test_knn_graph)

print(f"Created edge indices: train={train_edge_index.shape}, val={val_edge_index.shape}, test={test_edge_index.shape}")

# ==============================
# 12. Create PyTorch Tensors for Features
# ==============================

train_features = torch.tensor(train_uhg, dtype=torch.float32).to(device)
val_features = torch.tensor(val_uhg, dtype=torch.float32).to(device)
test_features = torch.tensor(test_uhg, dtype=torch.float32).to(device)

print(f"Train features shape: {train_features.shape}, device: {train_features.device}")
print(f"Val features shape: {val_features.shape}, device: {val_features.device}")
print(f"Test features shape: {test_features.shape}, device: {test_features.device}")
print(f"Train edge_index shape: {train_edge_index.shape}, device: {train_edge_index.device}")
print(f"Val edge_index shape: {val_edge_index.shape}, device: {val_edge_index.device}")
print(f"Test edge_index shape: {test_edge_index.shape}, device: {test_edge_index.device}")

# ==============================
# 13. Define the UHG-Based Loss Function
# ==============================

def uhg_loss(z, edge_index, batch_size, spread_weight=0.1):
    """
    Compute UHG-based loss for graph embeddings.
    
    Args:
        z: Node embeddings in UHG space
        edge_index: Graph connectivity
        batch_size: Batch size for processing
        spread_weight: Weight for the spread term in the loss
        
    Returns:
        Loss value
    """
    # Extract source and target nodes
    src, dst = edge_index
    
    # Compute quadrance (UHG distance) between connected nodes
    quad = uhg_quadrance(z[src], z[dst])
    
    # Compute spread (UHG angle) between node lines
    spread = uhg_spread(z[src], z[dst])
    
    # Combine quadrance and spread with weighting
    loss = quad.mean() + spread_weight * spread.mean()
    
    return loss

# ==============================
# 14. Training and Evaluation Functions
# ==============================

def train_model(model, data, optimizer, epochs=100, verbose=True):
    """
    Train the model.
    
    Args:
        model: Model to train
        data: Data to train on
        optimizer: Optimizer to use
        epochs: Number of epochs to train for
        verbose: Whether to print progress
        
    Returns:
        list: Training losses
    """
    print("Starting training...")
    print(f"Input shapes - x: {data.x.shape}, edge_index: {data.edge_index.shape}")
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        try:
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Check shapes for debugging
            print(f"Epoch {epoch+1} - Output shape: {out.shape}, Target shape: {data.x.shape}")
            
            # Compute loss - reconstruction loss in feature space
            loss = F.mse_loss(out, data.x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
                
        except Exception as e:
            print(f"Error during training at epoch {epoch+1}: {e}")
            print(f"Input shapes - x: {data.x.shape}, edge_index: {data.edge_index.shape}")
            if 'out' in locals():
                print(f"Output shape: {out.shape}")
            raise e
    
    return losses

@torch.no_grad()
def evaluate(model, features, edge_index, batch_size):
    """
    Evaluate the UHG-based graph neural network model.
    
    Args:
        model: The UHG graph neural network model
        features: Node features
        edge_index: Graph connectivity
        batch_size: Batch size for evaluation
        
    Returns:
        Evaluation loss
    """
    model.eval()
    
    # Forward pass
    z = model(features, edge_index)
    
    # Compute loss
    loss = uhg_loss(z, edge_index, batch_size)
    
    return loss.item()

# ==============================
# 15. Visualize Embeddings (PCA and t-SNE)
# ==============================

def visualize_embeddings(embeddings, labels=None, method='PCA', title='Embeddings Visualization'):
    """
    Visualize embeddings using dimensionality reduction techniques.
    
    Args:
        embeddings: High-dimensional embeddings to visualize
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('PCA', 't-SNE', or 'UMAP')
        title: Plot title
        
    Returns:
        None (displays the plot)
    """
    # Handle NaN values
    embeddings = np.nan_to_num(embeddings, nan=0.0)

    # PCA Visualization
    if method == 'PCA':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        print(f'Explained variance by 2 principal components: {np.sum(pca.explained_variance_ratio_):.2f}')

    # t-SNE Visualization
    elif method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        reduced_embeddings = tsne.fit_transform(embeddings)
        print('t-SNE completed.')
        
    # UMAP Visualization
    elif method == 'UMAP':
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            reduced_embeddings = reducer.fit_transform(embeddings)
            print('UMAP completed.')
        else:
            print("UMAP not available. Falling back to t-SNE.")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            reduced_embeddings = tsne.fit_transform(embeddings)
            print('t-SNE completed.')

    else:
        raise ValueError("Method must be 'PCA', 't-SNE', or 'UMAP'")

    # Plotting
    plt.figure(figsize=(10, 7))
    if labels is not None:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar()
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=50, alpha=0.7)

    plt.title(title)
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.tight_layout()
    plt.show()
    
    return reduced_embeddings

# ==============================
# 16. Clustering Analysis
# ==============================

def perform_clustering_analysis(embeddings, anomaly_scores=None, true_labels=None, eps=0.5, min_samples=5):
    """
    Perform clustering analysis on embeddings using DBSCAN and evaluate with metrics.
    
    Args:
        embeddings: High-dimensional embeddings to analyze
        anomaly_scores: Optional anomaly scores for coloring
        true_labels: Optional ground truth labels for evaluation
        eps: DBSCAN epsilon parameter (neighborhood distance)
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Dictionary with clustering results and metrics
    """
    results = {}
    
    # Handle NaN values
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    
    # Dimensionality reduction for visualization
    print("Performing dimensionality reduction for visualization...")
    
    # PCA for initial analysis
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)
    results['pca_embeddings'] = pca_embeddings
    
    # t-SNE for better cluster visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_embeddings = tsne.fit_transform(embeddings)
    results['tsne_embeddings'] = tsne_embeddings
    
    # UMAP if available
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, 
                           n_neighbors=min(15, len(embeddings)-1), 
                           min_dist=0.1)
        umap_embeddings = reducer.fit_transform(embeddings)
        results['umap_embeddings'] = umap_embeddings
    
    # Perform DBSCAN clustering on original embeddings
    print("Performing DBSCAN clustering...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    results['cluster_labels'] = cluster_labels
    
    # Count clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    results['n_clusters'] = n_clusters
    results['n_noise'] = n_noise
    
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    # Calculate Davies-Bouldin Index if more than one cluster
    if n_clusters > 1:
        # Filter out noise points for Davies-Bouldin calculation
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > n_clusters:  # Need more points than clusters
            db_score = davies_bouldin_score(
                embeddings[non_noise_mask], 
                cluster_labels[non_noise_mask]
            )
            results['davies_bouldin_score'] = db_score
            print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")
            
            # Calculate Silhouette Score
            silhouette = silhouette_score(
                embeddings[non_noise_mask], 
                cluster_labels[non_noise_mask]
            )
            results['silhouette_score'] = silhouette
            print(f"Silhouette Score: {silhouette:.4f} (higher is better)")
    
    # Visualize clustering results with different dimensionality reduction techniques
    fig, axes = plt.subplots(1, 3 if UMAP_AVAILABLE else 2, figsize=(18, 6))
    
    # Function to plot clusters
    def plot_clusters(ax, data, title):
        # Plot clusters
        unique_labels = set(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise
                col = [0, 0, 0, 1]
            
            class_mask = cluster_labels == k
            xy = data[class_mask]
            
            ax.scatter(
                xy[:, 0], xy[:, 1],
                s=50, c=[col], marker='o' if k != -1 else 'x',
                alpha=0.6, label=f'Cluster {k}' if k != -1 else 'Noise'
            )
            
        # If anomaly scores are provided, add them as point sizes
        if anomaly_scores is not None:
            # Normalize scores for visualization
            norm_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
            sizes = 20 + norm_scores * 100
            
            scatter = ax.scatter(
                data[:, 0], data[:, 1],
                s=sizes, c=cluster_labels, cmap='rainbow',
                alpha=0.6, edgecolors='k', linewidths=0.5
            )
        
        ax.set_title(title)
        ax.set_xlabel(f'{title.split()[0]} Component 1')
        ax.set_ylabel(f'{title.split()[0]} Component 2')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot PCA results
    plot_clusters(axes[0], pca_embeddings, 'PCA Clustering')
    
    # Plot t-SNE results
    plot_clusters(axes[1], tsne_embeddings, 't-SNE Clustering')
    
    # Plot UMAP results if available
    if UMAP_AVAILABLE:
        plot_clusters(axes[2], umap_embeddings, 'UMAP Clustering')
    
    plt.tight_layout()
    plt.savefig('visualizations/clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compare with true labels if provided
    if true_labels is not None:
        # Create comparison visualization
        fig, axes = plt.subplots(1, 3 if UMAP_AVAILABLE else 2, figsize=(18, 6))
        
        # Function to plot true vs predicted
        def plot_comparison(ax, data, title):
            scatter = ax.scatter(
                data[:, 0], data[:, 1],
                c=true_labels, cmap='viridis',
                s=50, alpha=0.7, edgecolors='k', linewidths=0.5
            )
            
            # Add contours for DBSCAN clusters
            try:
                from scipy.interpolate import griddata
                from matplotlib.colors import LinearSegmentedColormap
                
                # Create grid for contour
                x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
                y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
                grid_x, grid_y = np.meshgrid(
                    np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100)
                )
                # Interpolate cluster labels to grid
                grid_z = griddata(
                    (data[:, 0], data[:, 1]), 
                    cluster_labels, 
                    (grid_x, grid_y), 
                    method='nearest'
                )
                
                # Create custom colormap for clusters
                n_clusters_with_noise = len(set(cluster_labels))
                cluster_cmap = plt.cm.get_cmap('tab10', n_clusters_with_noise)
                
                # Plot contours
                contour = ax.contour(
                    grid_x, grid_y, grid_z,
                    levels=np.arange(n_clusters_with_noise+1)-0.5,
                    colors='k', linewidths=0.5, alpha=0.5
                )
            except Exception as e:
                print(f"Could not create contour plot: {e}")
            
            ax.set_title(f'{title} - True Labels vs Clusters')
            ax.set_xlabel(f'{title.split()[0]} Component 1')
            ax.set_ylabel(f'{title.split()[0]} Component 2')
            
            # Add colorbar for true labels
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('True Label')
            
        # Plot comparisons
        plot_comparison(axes[0], pca_embeddings, 'PCA')
        plot_comparison(axes[1], tsne_embeddings, 't-SNE')
        if UMAP_AVAILABLE:
            plot_comparison(axes[2], umap_embeddings, 'UMAP')
        
        plt.tight_layout()
        plt.savefig('visualizations/clustering_vs_true_labels.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate metrics comparing true labels with clusters
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Filter out noise points for comparison
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 0:
            ari = adjusted_rand_score(true_labels[non_noise_mask], cluster_labels[non_noise_mask])
            nmi = normalized_mutual_info_score(true_labels[non_noise_mask], cluster_labels[non_noise_mask])
            
            results['adjusted_rand_index'] = ari
            results['normalized_mutual_info'] = nmi
            
            print(f"Adjusted Rand Index: {ari:.4f} (higher is better)")
            print(f"Normalized Mutual Information: {nmi:.4f} (higher is better)")
    
    # Generate clustering report
    report = "=== UHG Clustering Analysis Report ===\n\n"
    report += f"Number of data points: {len(embeddings)}\n"
    report += f"Number of clusters detected: {n_clusters}\n"
    report += f"Number of noise points: {n_noise} ({n_noise/len(embeddings)*100:.2f}%)\n\n"
    
    if n_clusters > 1 and 'davies_bouldin_score' in results:
        report += f"Davies-Bouldin Index: {results['davies_bouldin_score']:.4f}\n"
        report += f"Silhouette Score: {results['silhouette_score']:.4f}\n\n"
    
    if true_labels is not None and 'adjusted_rand_index' in results:
        report += f"Adjusted Rand Index: {results['adjusted_rand_index']:.4f}\n"
        report += f"Normalized Mutual Information: {results['normalized_mutual_info']:.4f}\n\n"
    
    # Cluster statistics
    report += "Cluster Statistics:\n"
    for cluster_id in sorted(set(cluster_labels)):
        cluster_size = np.sum(cluster_labels == cluster_id)
        percentage = cluster_size / len(cluster_labels) * 100
        if cluster_id == -1:
            report += f"- Noise points: {cluster_size} ({percentage:.2f}%)\n"
        else:
            report += f"- Cluster {cluster_id}: {cluster_size} points ({percentage:.2f}%)\n"
    
    report += "\nINTERPRETATION:\n"
    report += "1. Clusters represent groups of similar data points in the UHG embedding space.\n"
    report += "2. Noise points (-1) are potential anomalies that don't belong to any cluster.\n"
    report += "3. The Davies-Bouldin Index measures cluster separation (lower is better).\n"
    report += "4. The Silhouette Score measures how well-defined clusters are (higher is better).\n"
    report += "5. Different dimensionality reduction techniques (PCA, t-SNE, UMAP) provide\n"
    report += "   complementary views of the cluster structure.\n\n"
    
    report += "RECOMMENDATIONS:\n"
    report += "1. Investigate noise points as potential anomalies.\n"
    report += "2. Compare cluster assignments with anomaly scores for validation.\n"
    report += "3. Adjust DBSCAN parameters (eps, min_samples) to refine clustering.\n"
    report += "4. Use UMAP for better preservation of both local and global structure.\n"
    report += "5. Consider hierarchical clustering for nested pattern detection.\n"
    
    # Save the report
    os.makedirs('visualizations', exist_ok=True)
    with open('visualizations/clustering_report.txt', 'w') as f:
        f.write(report)
    
    print("Clustering report saved to 'visualizations/clustering_report.txt'")
    
    return results

# ==============================
# 16. Main Script for Training and Visualization
# ==============================

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Use the real data that was already loaded and processed
    print("Using real data from train/val/test datasets...")
    
    # Create PyTorch Geometric Data objects and move to the correct device
    train_data = Data(x=train_features.to(device), edge_index=train_edge_index.to(device))
    val_data = Data(x=val_features.to(device), edge_index=val_edge_index.to(device))
    test_data = Data(x=test_features.to(device), edge_index=test_edge_index.to(device))
    
    print(f"Train data: {train_data}")
    print(f"Validation data: {val_data}")
    print(f"Test data: {test_data}")
    
    # Use the test data for anomaly detection
    graph_data = test_data
    
    # Create a model and move it to the correct device
    print("Creating model...")
    in_channels = graph_data.x.shape[1]  # Input feature dimension
    hidden_channels = 64  # Hidden dimension
    model = GraphSAGE_CustomScatter(in_channels, hidden_channels).to(device)
    print(f"Model created with input dimension {in_channels} and hidden dimension {hidden_channels}")
    
    # Train the model
    print("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = train_model(model, graph_data, optimizer, epochs=100)
    
    # Compute anomaly scores
    print("Computing anomaly scores...")
    anomaly_scores = compute_anomaly_scores(model, graph_data)
    
    # Convert to numpy for percentile calculation
    anomaly_scores_np = anomaly_scores.cpu().numpy()
    
    # Determine threshold (95th percentile)
    threshold = np.percentile(anomaly_scores_np, 95)
    print(f"Anomaly threshold: {threshold:.4f}")
    
    # Detect anomalies
    anomalies = torch.where(anomaly_scores > threshold)[0]
    print(f"Detected {len(anomalies)} anomalies out of {len(anomaly_scores)} nodes ({len(anomalies)/len(anomaly_scores)*100:.2f}%)")
    
    # Visualize results
    print("Creating visualizations...")
    visualize_results(model, graph_data, anomaly_scores, anomalies, threshold, losses)
    
    # Generate anomaly report
    print("Generating anomaly report...")
    report = generate_anomaly_report(anomaly_scores, anomalies, threshold)
    
    # Perform additional clustering analysis
    print("\nPerforming standalone clustering analysis...")
    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()
    
    # Create binary labels for anomalies (1 for anomaly, 0 for normal)
    binary_labels = torch.zeros(len(anomaly_scores))
    binary_labels[anomalies] = 1
    
    # Perform clustering with different dimensionality reduction techniques
    print("\nComparing different dimensionality reduction techniques:")
    
    # PCA
    print("\n1. PCA + Clustering:")
    pca_embeddings = visualize_embeddings(
        embeddings[:, :-1],  # Exclude homogeneous coordinate
        labels=binary_labels,
        method='PCA',
        title='PCA Visualization with Anomaly Labels'
    )
    
    # t-SNE
    print("\n2. t-SNE + Clustering:")
    tsne_embeddings = visualize_embeddings(
        embeddings[:, :-1],  # Exclude homogeneous coordinate
        labels=binary_labels,
        method='t-SNE',
        title='t-SNE Visualization with Anomaly Labels'
    )
    
    # UMAP if available
    if UMAP_AVAILABLE:
        print("\n3. UMAP + Clustering:")
        umap_embeddings = visualize_embeddings(
            embeddings[:, :-1],  # Exclude homogeneous coordinate
            labels=binary_labels,
            method='UMAP',
            title='UMAP Visualization with Anomaly Labels'
        )
    
    # Perform DBSCAN clustering on the original embeddings
    print("\nPerforming DBSCAN clustering on original embeddings...")
    # Try different eps values to find optimal clustering
    eps_values = [0.3, 0.5, 0.7, 1.0]
    min_samples_values = [5, 10, 15]
    
    best_silhouette = -1
    best_eps = None
    best_min_samples = None
    best_clusters = None
    
    results_table = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\nTrying DBSCAN with eps={eps}, min_samples={min_samples}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(embeddings[:, :-1])  # Exclude homogeneous coordinate
            
            # Count clusters and noise points
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
            
            # Calculate silhouette score if more than one cluster and not all noise
            silhouette = None
            if n_clusters > 1:
                # Filter out noise points for silhouette calculation
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > n_clusters:  # Need more points than clusters
                    try:
                        silhouette = silhouette_score(
                            embeddings[non_noise_mask, :-1], 
                            cluster_labels[non_noise_mask]
                        )
                        print(f"Silhouette Score: {silhouette:.4f}")
                        
                        # Calculate Davies-Bouldin score
                        db_score = davies_bouldin_score(
                            embeddings[non_noise_mask, :-1], 
                            cluster_labels[non_noise_mask]
                        )
                        print(f"Davies-Bouldin Score: {db_score:.4f}")
                        
                        # Track best parameters
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_eps = eps
                            best_min_samples = min_samples
                            best_clusters = cluster_labels
                        
                        # Add to results table
                        results_table.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'silhouette': silhouette,
                            'davies_bouldin': db_score
                        })
                    except Exception as e:
                        print(f"Error calculating scores: {e}")
    
    # Print results table
    if results_table:
        print("\nDBSCAN Parameter Comparison:")
        print("-" * 80)
        print(f"{'eps':<6} {'min_samples':<12} {'clusters':<10} {'noise':<10} {'silhouette':<12} {'davies_bouldin':<15}")
        print("-" * 80)
        for result in results_table:
            print(f"{result['eps']:<6.2f} {result['min_samples']:<12d} {result['n_clusters']:<10d} {result['n_noise']:<10d} {result['silhouette']:<12.4f} {result['davies_bouldin']:<15.4f}")
        
        # Use best parameters for final clustering analysis
        if best_eps is not None:
            print(f"\nBest parameters: eps={best_eps}, min_samples={best_min_samples}")
            print("Performing final clustering analysis with best parameters...")
            
            clustering_results = perform_clustering_analysis(
                embeddings[:, :-1],  # Exclude homogeneous coordinate
                anomaly_scores=anomaly_scores.cpu().numpy(),
                true_labels=binary_labels.cpu().numpy(),
                eps=best_eps,
                min_samples=best_min_samples
            )
    
    print("\nAnalysis complete. All results saved to 'visualizations' directory.")
    return model, graph_data, anomaly_scores, anomalies, threshold

def generate_random_graph(num_nodes, edge_prob):
    """
    Create a random graph with a given edge probability.
    
    Args:
        num_nodes (int): Number of nodes in the graph
        edge_prob (float): Probability of an edge between any two nodes
        
    Returns:
        torch_geometric.data.Data: PyTorch Geometric Data object with node features and edge_index
    """
    print(f"Creating random graph with {num_nodes} nodes and edge probability {edge_prob}")
    
    # Create an empty adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    
    # Fill the adjacency matrix with edge probabilities
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if torch.rand(1) < edge_prob:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    # Convert adjacency matrix to edge_index
    edge_index = torch.nonzero(adj_matrix).t()
    print(f"Created graph with {edge_index.shape[1]} edges")
    
    # Generate random node features (10 features per node)
    num_features = 10
    x = torch.randn(num_nodes, num_features)
    
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    
    return data

@torch.no_grad()
def compute_anomaly_scores(model, data):
    """
    Compute anomaly scores based on reconstruction error.
    
    Args:
        model: Trained model
        data: PyTorch Geometric Data object
        
    Returns:
        torch.Tensor: Anomaly scores for each node
    """
    print("Computing anomaly scores...")
    model.eval()
    
    # Get reconstructions
    reconstructions = model(data.x, data.edge_index)
    print(f"Reconstruction shape: {reconstructions.shape}")
    
    # Compute reconstruction error for each node
    errors = torch.sum((reconstructions - data.x)**2, dim=1)
    print(f"Error tensor shape: {errors.shape}")
    
    # Convert to tensor
    anomaly_scores = errors
    
    print(f"Anomaly score statistics: min={anomaly_scores.min().item():.4f}, max={anomaly_scores.max().item():.4f}, mean={anomaly_scores.mean().item():.4f}")
    
    return anomaly_scores

def visualize_results(model, data, anomaly_scores, anomalies, threshold, losses=None):
    """
    Visualize the results of anomaly detection.
    
    Args:
        model: Trained model
        data: Input data
        anomaly_scores: Computed anomaly scores (torch.Tensor)
        anomalies: Binary tensor indicating anomalies
        threshold: Threshold for anomaly detection
        losses: Training losses (optional)
    """
    os.makedirs('visualizations', exist_ok=True)
    
    # Convert to numpy for plotting
    scores_np = anomaly_scores.cpu().numpy()
    anomalies_np = anomalies.cpu().numpy()
    
    # Create a comprehensive visualization
    plt.figure(figsize=(18, 12))
    
    # 1. Anomaly Score Distribution
    plt.subplot(2, 3, 1)
    plt.hist(scores_np, bins=30, alpha=0.7, color='skyblue')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.annotate(
        'This histogram shows the distribution of anomaly scores.\n'
        'The red line indicates the threshold for anomaly detection.\n'
        'Scores to the right of the line are classified as anomalies.',
        xy=(0.5, 0.1), xycoords='axes fraction', ha='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
    )
    
    # 2. Training Loss Curve (if provided)
    plt.subplot(2, 3, 2)
    if losses is not None:
        plt.plot(losses, color='blue', alpha=0.7)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.annotate(
            'This plot shows how the model loss decreased during training.\n'
            'A stable convergence indicates good model fitting.\n'
            'Sharp drops may indicate important learning moments.',
            xy=(0.5, 0.1), xycoords='axes fraction', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
        )
    else:
        plt.text(0.5, 0.5, 'Training loss data not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Training Loss (Not Available)')
    
    # 3. Anomaly Scores vs. Indices
    plt.subplot(2, 3, 3)
    # Create a color array where anomalies are red (1) and normal points are blue (0)
    colors = np.zeros(len(scores_np))
    colors[anomalies_np] = 1
    
    plt.scatter(range(len(scores_np)), scores_np, c=colors, 
               cmap='coolwarm', alpha=0.7, s=30)
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Anomaly Scores by Index')
    plt.xlabel('Data Point Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True)
    plt.annotate(
        'This scatter plot shows anomaly scores for each data point.\n'
        'Points above the threshold (red line) are classified as anomalies.\n'
        'Clusters of high scores may indicate related anomalies.',
        xy=(0.5, 0.1), xycoords='axes fraction', ha='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
    )
    
    # 4. Get embeddings from the model
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    
    # Convert to numpy
    embeddings = embeddings.cpu().numpy()
    
    # 5. PCA Visualization of Embeddings
    plt.subplot(2, 3, 4)
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings[:, :-1])  # Exclude homogeneous coordinate
    
    # Create a color array where anomalies are red (1) and normal points are blue (0)
    colors = np.zeros(len(scores_np))
    colors[anomalies_np] = 1
    
    # Plot all points with colors based on anomaly detection
    plt.scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1],
        c=colors, cmap='coolwarm', alpha=0.6, s=30
    )
    
    # Highlight anomalies with a different marker
    if len(anomalies_np) > 0:
        plt.scatter(
            reduced_embeddings[anomalies_np, 0], 
            reduced_embeddings[anomalies_np, 1],
            c='red', alpha=0.8, marker='X', s=100, edgecolors='black', linewidths=1,
            label='Anomaly'
        )
        plt.legend()
    
    plt.title('PCA Visualization of UHG Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.annotate(
        'This PCA plot shows the 2D projection of node embeddings in UHG space.\n'
        'Anomalies (red X) should ideally be separated from normal points (blue).\n'
        'Clustering patterns reveal the underlying structure of the data.',
        xy=(0.5, 0.1), xycoords='axes fraction', ha='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
    )
    
    # 6. t-SNE Visualization of Embeddings
    plt.subplot(2, 3, 5)
    # Reduce to 2D using t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_embeddings = tsne.fit_transform(embeddings[:, :-1])  # Exclude homogeneous coordinate
    
    # Create a color array where anomalies are red (1) and normal points are blue (0)
    colors = np.zeros(len(scores_np))
    colors[anomalies_np] = 1
    
    # Plot all points with colors based on anomaly detection
    plt.scatter(
        tsne_embeddings[:, 0], 
        tsne_embeddings[:, 1],
        c=colors, cmap='coolwarm', alpha=0.6, s=30
    )
    
    # Highlight anomalies with a different marker
    if len(anomalies_np) > 0:
        plt.scatter(
            tsne_embeddings[anomalies_np, 0], 
            tsne_embeddings[anomalies_np, 1],
            c='red', alpha=0.8, marker='X', s=100, edgecolors='black', linewidths=1,
            label='Anomaly'
        )
        plt.legend()
    
    plt.title('t-SNE Visualization of UHG Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.annotate(
        't-SNE preserves local structure better than PCA, showing neighborhood relationships.\n'
        'Anomalies (red X) that are distant from clusters indicate structural outliers.\n'
        'This visualization helps identify patterns not visible in PCA.',
        xy=(0.5, 0.1), xycoords='axes fraction', ha='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
    )
    
    # 7. Reconstruction Error Comparison
    plt.subplot(2, 3, 6)
    model.eval()
    with torch.no_grad():
        reconstructions = model(data.x, data.edge_index)
        
    # Calculate feature-wise reconstruction error
    reconstruction_errors = (reconstructions - data.x).pow(2).mean(dim=0).cpu().numpy()
    feature_indices = np.arange(len(reconstruction_errors))
    
    plt.bar(feature_indices, reconstruction_errors)
    plt.title('Feature-wise Reconstruction Error')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.annotate(
        'This plot shows which features are most difficult for the model to reconstruct.\n'
        'Higher bars indicate features that contribute more to anomaly detection.\n'
        'These features may be more important for identifying anomalies.',
        xy=(0.5, 0.1), xycoords='axes fraction', ha='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
    )
    
    # Add a title to the entire figure
    plt.suptitle('UHG-Based Anomaly Detection Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig('visualizations/uhg_anomaly_detection_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 3D visualization of UHG space
    visualize_uhg_space_3d(embeddings, colors)
    
    # Create a Poincaré disk visualization
    visualize_uhg_poincare_disk(embeddings, colors, anomalies)
    
    # Perform clustering analysis on the embeddings
    print("\nPerforming clustering analysis on UHG embeddings...")
    # Use only the feature part of embeddings (exclude homogeneous coordinate)
    clustering_results = perform_clustering_analysis(
        embeddings[:, :-1],  # Exclude homogeneous coordinate
        anomaly_scores=scores_np,
        true_labels=None,  # We don't have ground truth clusters
        eps=0.5,  # DBSCAN epsilon parameter
        min_samples=5  # DBSCAN min_samples parameter
    )
    
    # If we have labels, also perform a labeled analysis
    if hasattr(data, 'y') and data.y is not None:
        print("\nPerforming clustering analysis with true labels...")
        labels = data.y.cpu().numpy()
        labeled_clustering_results = perform_clustering_analysis(
            embeddings[:, :-1],  # Exclude homogeneous coordinate
            anomaly_scores=scores_np,
            true_labels=labels,
            eps=0.5,  # DBSCAN epsilon parameter
            min_samples=5  # DBSCAN min_samples parameter
        )
    
    print("Visualizations saved to 'visualizations' directory.")

def visualize_uhg_space_3d(embeddings, colors):
    """
    Create a 3D visualization of points in UHG space.
    
    Args:
        embeddings: Node embeddings in UHG space
        colors: Array indicating normal (0) or anomalous (1) nodes
    """
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use PCA to reduce to 3D if needed
    if embeddings.shape[1] > 4:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings[:, :-1])
    else:
        # Use the first 3 dimensions
        reduced_embeddings = embeddings[:, :3]
    
    # Plot normal points
    ax.scatter(
        reduced_embeddings[colors == 0, 0], 
        reduced_embeddings[colors == 0, 1],
        reduced_embeddings[colors == 0, 2],
        c='blue', label='Normal', alpha=0.6
    )
    
    # Plot anomalies
    ax.scatter(
        reduced_embeddings[colors == 1, 0], 
        reduced_embeddings[colors == 1, 1],
        reduced_embeddings[colors == 1, 2],
        c='red', label='Anomaly', alpha=0.8, marker='X', s=100
    )
    
    # Add labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Visualization of UHG Space')
    ax.legend()
    
    # Add explanation text
    plt.figtext(
        0.5, 0.01,
        "This 3D visualization shows the embedding of nodes in UHG space.\n"
        "In Universal Hyperbolic Geometry, points are represented in projective space.\n"
        "Anomalies (red X) often appear at the boundaries or separated from normal clusters (blue).\n"
        "The hyperbolic structure captures hierarchical relationships better than Euclidean space.",
        ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
    )
    
    # Save the figure
    plt.savefig('visualizations/uhg_space_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_uhg_poincare_disk(embeddings, colors, anomalies):
    """
    Visualize the data in the Poincaré disk model of hyperbolic space.
    
    Args:
        embeddings: Node embeddings in UHG space
        colors: Array indicating normal (0) or anomalous (1) nodes
        anomalies: Indices of detected anomalies
    """
    print("Creating Poincaré disk visualization...")
    
    # Create a figure
    plt.figure(figsize=(12, 12))
    
    # Draw the unit circle (boundary of Poincaré disk)
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    # Project the embeddings to 2D if needed
    if embeddings.shape[1] > 3:
        # Use PCA to reduce dimensionality
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        projected_embeddings = pca.fit_transform(embeddings[:, :-1])  # Exclude homogeneous coordinate
    else:
        # Use the first 2 dimensions
        projected_embeddings = embeddings[:, :2]
    
    # Scale the embeddings to fit within the unit disk
    max_norm = np.max(np.linalg.norm(projected_embeddings, axis=1))
    scaled_embeddings = projected_embeddings / (max_norm * 1.1)  # Add a small margin
    
    # Plot normal points
    plt.scatter(
        scaled_embeddings[colors == 0, 0], 
        scaled_embeddings[colors == 0, 1],
        c='blue', label='Normal', alpha=0.6
    )
    
    # Plot anomalies
    plt.scatter(
        scaled_embeddings[colors == 1, 0], 
        scaled_embeddings[colors == 1, 1],
        c='red', label='Anomaly', alpha=0.8, marker='X', s=100
    )
    
    # Draw hyperbolic geodesics (straight lines in hyperbolic space)
    # Connect each anomaly to its nearest normal point
    for anomaly_idx in anomalies:
        # Find the nearest normal point
        normal_indices = np.where(colors == 0)[0]
        distances = np.linalg.norm(
            scaled_embeddings[anomaly_idx] - scaled_embeddings[normal_indices], 
            axis=1
        )
        nearest_normal_idx = normal_indices[np.argmin(distances)]
        
        # Draw a hyperbolic geodesic (approximated by a straight line in this visualization)
        plt.plot(
            [scaled_embeddings[anomaly_idx, 0], scaled_embeddings[nearest_normal_idx, 0]],
             [scaled_embeddings[anomaly_idx, 1], scaled_embeddings[nearest_normal_idx, 1]],
             'k-', alpha=0.3
        )
    
    # Add grid lines for the hyperbolic space
    # Draw some hyperbolic circles (geodesic circles centered at the origin)
    for radius in [0.2, 0.4, 0.6, 0.8]:
        circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.3)
        plt.gca().add_patch(circle)
    
    # Draw some hyperbolic radii (geodesic rays from the origin)
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = np.cos(angle)
        y = np.sin(angle)
        plt.plot([0, x], [0, y], 'gray', alpha=0.3)
    
    # Set equal aspect ratio
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    # Add labels and title
    plt.title('Poincaré Disk Visualization of UHG Embeddings')
    plt.legend(loc='upper right')
    
    # Add explanation
    plt.figtext(
        0.5, 0.01,
        "This visualization shows the data in the Poincaré disk model of hyperbolic space.\n"
        "In this model, the entire hyperbolic plane is mapped to the unit disk.\n"
        "Distances are distorted: points near the boundary are actually far apart in hyperbolic space.\n"
        "Anomalies (red X) often appear near the boundary, indicating they are far from normal points in hyperbolic space.\n"
        "The black lines represent hyperbolic geodesics (shortest paths in hyperbolic space).",
        ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8)
    )
    
    # Save the figure
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/poincare_disk_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Poincaré disk visualization saved to 'visualizations/poincare_disk_visualization.png'")

def generate_anomaly_report(anomaly_scores, anomalies, threshold):
    """
    Generate a detailed report on the detected anomalies.
    
    Args:
        anomaly_scores: Tensor of anomaly scores
        anomalies: Indices of detected anomalies
        threshold: Threshold used for detection
    """
    # Convert to numpy for reporting
    if torch.is_tensor(anomaly_scores):
        anomaly_scores_np = anomaly_scores.cpu().numpy()
    else:
        anomaly_scores_np = anomaly_scores
        
    if torch.is_tensor(anomalies):
        anomalies_np = anomalies.cpu().numpy()
    else:
        anomalies_np = anomalies
    
    # Create a report string
    report = "=" * 50 + "\n"
    report += "UHG-BASED ANOMALY DETECTION REPORT\n"
    report += "=" * 50 + "\n\n"
    
    # Summary statistics
    report += "SUMMARY STATISTICS:\n"
    report += f"Total nodes analyzed: {len(anomaly_scores_np)}\n"
    report += f"Number of anomalies detected: {len(anomalies_np)}\n"
    report += f"Anomaly rate: {len(anomalies_np)/len(anomaly_scores_np)*100:.2f}%\n"
    report += f"Detection threshold: {threshold:.4f}\n\n"
    
    report += "Score distribution:\n"
    report += f"  - Minimum score: {np.min(anomaly_scores_np):.4f}\n"
    report += f"  - Maximum score: {np.max(anomaly_scores_np):.4f}\n"
    report += f"  - Mean score: {np.mean(anomaly_scores_np):.4f}\n"
    report += f"  - Median score: {np.median(anomaly_scores_np):.4f}\n"
    report += f"  - Standard deviation: {np.std(anomaly_scores_np):.4f}\n\n"
    
    # Detailed anomaly information
    report += "DETAILED ANOMALY INFORMATION:\n"
    report += "Index    |    Score    |    Z-Score\n"
    report += "-" * 40 + "\n"
    
    # Calculate z-scores
    mean_score = np.mean(anomaly_scores_np)
    std_score = np.std(anomaly_scores_np)
    
    # Sort anomalies by score in descending order
    sorted_indices = np.argsort(anomaly_scores_np[anomalies_np])[::-1]
    sorted_anomalies = anomalies_np[sorted_indices]
    
    for idx in sorted_anomalies:
        score = anomaly_scores_np[idx]
        z_score = (score - mean_score) / std_score
        report += f"{idx:5d}    |    {score:8.4f}    |    {z_score:8.4f}\n"
    
    report += "\n"
    
    # Interpretation
    report += "INTERPRETATION:\n"
    report += "1. Anomalies represent nodes with unusual patterns compared to the majority.\n"
    report += "2. Higher anomaly scores indicate stronger deviation from normal patterns.\n"
    report += "3. Z-scores quantify how many standard deviations an anomaly is from the mean.\n"
    report += "4. Nodes with z-scores > 2 are statistically significant outliers.\n"
    report += "5. The UHG-based approach captures complex relationships in hyperbolic space.\n\n"
    
    report += "RECOMMENDATIONS:\n"
    report += "1. Investigate nodes with the highest anomaly scores first.\n"
    report += "2. Look for patterns among the detected anomalies.\n"
    report += "3. Consider adjusting the threshold based on domain knowledge.\n"
    report += "4. For time-series data, monitor how anomaly patterns evolve over time.\n"
    report += "5. Compare results with other anomaly detection methods for validation.\n"
    
    # Save the report to a file
    os.makedirs('visualizations', exist_ok=True)
    with open('visualizations/anomaly_report.txt', 'w') as f:
        f.write(report)
    
    print("Anomaly report saved to 'visualizations/anomaly_report.txt'")
    
    return report

if __name__ == "__main__":
    main()

