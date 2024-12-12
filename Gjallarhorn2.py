"""
Gjallarhorn2: Enhanced UHG-Compliant Hyperbolic Graph Neural Network for Anomaly Detection
Version: 0.3.0
"""

# Core imports that don't require installation
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def check_install_packages():
    """Check and install required packages with specific versions to avoid conflicts."""
    try:
        import pkg_resources
    except ImportError:
        return

    # Define required packages with versions
    required = {
        'uhg': '>=0.3.0',  # Add UHG first to ensure its dependencies are handled
        'torch': '>=1.9.0',
        'pandas': '>=1.3.0',
        'scikit-learn': '>=0.24.2',
        'rich': '>=10.0.0',
        'python-dateutil': '>=2.8.2',
        'torch-geometric': '>=2.0.0',
        'scipy': '>=1.7.0'
    }

    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    missing = []

    for package, version in required.items():
        if package not in installed:
            missing.append(f"{package}{version}")

    if missing:
        print("Installing required packages...")
        import subprocess
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Package installation complete. Please restart the runtime if needed.")
        return True
    return False

# Check and install packages if needed
if check_install_packages():
    print("Please restart the runtime to continue.")
    sys.exit(0)

# Import core packages first
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Import UHG components directly (only what we need)
from uhg.projective import ProjectiveUHG

# Import remaining packages
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.console import Console
from rich.theme import Theme
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import scipy.sparse
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from typing import Optional, Tuple
from google.colab import drive

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')

def create_knn_graph(data, k, batch_size=1000, console=None):
    """Create KNN graph with batching for memory efficiency."""
    n_samples = data.shape[0]
    n_batches = (n_samples - 1) // batch_size + 1
    knn_graphs = []
    
    # Progress bar for graph construction
    construction_progress = Progress(
        SpinnerColumn(spinner_name="dots12", style="yellow"),
        TextColumn("[bold yellow]Gjallarhorn under construction[/]"),
        BarColumn(complete_style="yellow", finished_style="green"),
        TimeElapsedColumn(),
        "|",
        TimeRemainingColumn(),
        "|",
        MofNCompleteColumn(),
        console=console,
        expand=True
    )
    
    with construction_progress:
        task = construction_progress.add_task("Building", total=n_batches)
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_knn = kneighbors_graph(
                X=data,
                n_neighbors=k,
                mode='distance',
                include_self=False,
                n_jobs=-1
            )[i:end]
            knn_graphs.append(batch_knn)
            construction_progress.update(task, advance=1)
            
    return scipy.sparse.vstack(knn_graphs)

def to_uhg_space(x):
    """Convert data to UHG space by adding homogeneous coordinate."""
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# Load and preprocess data
print("Loading data...")
train_file = '/content/drive/MyDrive/modbus/train_data_balanced_new.csv'
val_file = '/content/drive/MyDrive/modbus/val_data_balanced_new.csv'
test_file = '/content/drive/MyDrive/modbus/test_data_balanced_new.csv'

train_data = pd.read_csv(train_file)
val_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)

# Handle missing values and non-numeric columns
print("Preprocessing data...")
for df in [train_data, val_data, test_data]:
    df.fillna(df.mean(), inplace=True)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df = pd.get_dummies(df, columns=non_numeric)

# Scale the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Apply PCA if needed
n_components = min(50, train_scaled.shape[1])
if n_components < train_scaled.shape[1]:
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=n_components)
    train_scaled = pca.fit_transform(train_scaled)
    val_scaled = pca.transform(val_scaled)
    test_scaled = pca.transform(test_scaled)
    print(f'PCA applied. Reduced to {n_components} components.')

# Convert to UHG space first (in numpy)
print("Converting to UHG space...")
train_uhg = to_uhg_space(train_scaled)
val_uhg = to_uhg_space(val_scaled)
test_uhg = to_uhg_space(test_scaled)

# Then convert to tensors and move to device
print("Converting to tensors...")
try:
    train_features = torch.tensor(train_uhg, dtype=torch.float32).to(device)
    val_features = torch.tensor(val_uhg, dtype=torch.float32).to(device)
    test_features = torch.tensor(test_uhg, dtype=torch.float32).to(device)
except RuntimeError as e:
    print(f"Error during tensor conversion: {e}")
    raise

print(f"Feature shapes: Train {train_features.shape}, Val {val_features.shape}, Test {test_features.shape}")

# Create graph structures with batching
console = Console()
k = 2  # number of neighbors

print("Creating graph structures...")
train_A = create_knn_graph(train_scaled, k, console=console)
val_A = create_knn_graph(val_scaled, k, console=console)
test_A = create_knn_graph(test_scaled, k, console=console)

# Convert to edge indices
train_edge_index, train_edge_weight = from_scipy_sparse_matrix(train_A)
val_edge_index, val_edge_weight = from_scipy_sparse_matrix(val_A)
test_edge_index, test_edge_weight = from_scipy_sparse_matrix(test_A)

# Move to device
train_edge_index = train_edge_index.to(device)
val_edge_index = val_edge_index.to(device)
test_edge_index = test_edge_index.to(device)

console.print("[bold green]All graph structures constructed![/]")

def uhg_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute UHG inner product between points."""
    return torch.sum(a[..., :-1] * b[..., :-1], dim=-1) - a[..., -1] * b[..., -1]

def uhg_norm(a: torch.Tensor) -> torch.Tensor:
    """Compute UHG norm of points."""
    return torch.sum(a[..., :-1] ** 2, dim=-1) - a[..., -1] ** 2

def uhg_quadrance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG quadrance between points."""
    dot_product = uhg_inner_product(a, b)
    norm_a = uhg_norm(a)
    norm_b = uhg_norm(b)
    denom = norm_a * norm_b
    denom = torch.clamp(denom.abs(), min=eps)
    quadrance = 1 - (dot_product ** 2) / denom
    return quadrance

def uhg_spread(L: torch.Tensor, M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute UHG spread between lines."""
    dot_product = uhg_inner_product(L, M)
    norm_L = uhg_norm(L)
    norm_M = uhg_norm(M)
    denom = norm_L * norm_M
    denom = torch.clamp(denom.abs(), min=eps)
    spread = 1 - (dot_product ** 2) / denom
    return spread

class ProjectiveBaseLayer(nn.Module):
    """Base layer for UHG-compliant neural network operations."""
    
    def __init__(self):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
    def projective_transform(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation."""
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        if features.size(1) != weight.size(1):
            raise ValueError(f"Feature dimension {features.size(1)} does not match weight dimension {weight.size(1)}")
        
        transformed = torch.matmul(features, weight.t())
        out = torch.cat([transformed, homogeneous], dim=-1)
        return self.normalize_points(out)
        
    def normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points in projective space."""
        features = points[..., :-1]
        homogeneous = points[..., -1:]
        
        zero_mask = torch.all(features == 0, dim=-1, keepdim=True)
        features = torch.where(zero_mask, torch.ones_like(features), features)
        
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        normalized_features = features / torch.clamp(norm, min=1e-8)
        
        normalized = torch.cat([normalized_features, homogeneous], dim=-1)
        sign = torch.sign(normalized[..., -1:])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        
        return normalized * sign

def aggregate_neighbors(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Aggregate neighbor features using native PyTorch operations."""
    row, col = edge_index
    
    # Sum neighbor features
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col])
    
    # Compute mean
    ones = torch.ones(col.size(0), device=x.device)
    count = torch.zeros(x.size(0), device=x.device)
    count.index_add_(0, row, ones)
    count = count.clamp(min=1).view(-1, 1)
    
    return out / count

class UHGSAGEConv(ProjectiveBaseLayer):
    """UHG-compliant GraphSAGE convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, append_uhg: bool = True):
        super().__init__()
        self.append_uhg = append_uhg
        self.linear = nn.Linear(in_channels * 2, out_channels)
        
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        with torch.no_grad():
            self.linear.weight.div_(
                torch.norm(self.linear.weight, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            )
            
    def uhg_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """UHG-compliant normalization."""
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        norm = torch.sqrt(torch.clamp(uhg_norm(x), min=1e-8))
        features = features / norm.unsqueeze(-1)
        homogeneous = torch.sign(homogeneous) * torch.ones_like(homogeneous)
        
        return torch.cat([features, homogeneous], dim=-1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with UHG-compliant operations."""
        agg = aggregate_neighbors(x, edge_index)
        out = torch.cat([x, agg], dim=1)
        out = self.linear(out)
        out = F.relu(out)
        
        if self.append_uhg:
            ones = torch.ones((out.size(0), 1), device=out.device)
            out = torch.cat([out, ones], dim=1)
            out = self.uhg_normalize(out)
            out = self.normalize_points(out)
        
        return out

class UHGModel(nn.Module):
    """UHG-compliant graph neural network model."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(UHGSAGEConv(in_channels, hidden_channels, append_uhg=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(UHGSAGEConv(hidden_channels + 1, hidden_channels, append_uhg=True))
            
        self.convs.append(UHGSAGEConv(hidden_channels + 1, out_channels, append_uhg=True))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

class UHGLoss(nn.Module):
    """UHG-compliant loss function optimized for anomaly detection."""
    
    def __init__(self, spread_weight: float = 0.01, quad_weight: float = 0.5):
        super().__init__()
        self.spread_weight = spread_weight
        self.quad_weight = quad_weight
        
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute UHG-compliant loss."""
        mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
        pos_edge_index = edge_index[:, mask]
        
        if pos_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=z.device)
            
        pos_quad = torch.clamp(uhg_quadrance(z[pos_edge_index[0]], z[pos_edge_index[1]]), max=10.0)
        
        neg_edge_index = torch.randint(0, batch_size, (2, batch_size), device=z.device)
        neg_quad = torch.clamp(uhg_quadrance(z[neg_edge_index[0]], z[neg_edge_index[1]]), max=10.0)
        
        spread = torch.clamp(uhg_spread(z[pos_edge_index[0]], z[pos_edge_index[1]]), max=10.0)
        
        pos_loss = torch.mean(pos_quad)
        neg_loss = torch.mean(F.relu(1 - neg_quad))
        spread_loss = self.spread_weight * spread.mean()
        
        total_loss = self.quad_weight * (pos_loss + neg_loss) + spread_loss
        return torch.clamp(total_loss, min=0, max=1000.0)

if __name__ == "__main__":
    # Initialize model with reduced complexity
    print("Initializing model...")
    model = UHGModel(
        in_channels=train_features.shape[1],
        hidden_channels=32,  # Reduced from 64
        out_channels=16,     # Reduced from 32
        num_layers=2         # Reduced from 3
    ).to(device)

    # Optimizer with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Reduced from 0.001
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    criterion = UHGLoss(spread_weight=0.01, quad_weight=0.5)  # Updated weights
    scaler = torch.cuda.amp.GradScaler()

    # Training with smaller batch size
    num_epochs = 100
    batch_size = 256  # Reduced from 1024

    print("Starting training...")
    model.train()

    # Progress bar for training
    hunting_progress = Progress(
        SpinnerColumn(spinner_name="arrow3", style="red"),
        TextColumn("[bold red]Gjallarhorn is hunting[/]"),
        BarColumn(complete_style="red", finished_style="green"),
        TimeElapsedColumn(),
        "|",
        TimeRemainingColumn(),
        "|",
        MofNCompleteColumn(),
        console=console,
        expand=True
    )

    best_loss = float('inf')
    patience_counter = 0

    with hunting_progress:
        hunt_task = hunting_progress.add_task("Training", total=num_epochs)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                out = model(train_features, train_edge_index)
                loss = criterion(out, train_edge_index, batch_size)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Learning rate scheduling
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            hunting_progress.update(hunt_task, advance=1)
            
            if (epoch + 1) % 10 == 0:
                console.print(
                    f'[bold yellow]Epoch [{epoch+1}/{num_epochs}][/], '
                    f'[red]Loss: {loss.item():.4f}[/], '
                    f'[cyan]LR: {current_lr:.6f}[/]'
                )
                torch.cuda.empty_cache()

            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:  # Stop if no improvement for 20 epochs
                    console.print("[yellow]Early stopping triggered![/]")
                    break

    console.print("[bold green]Gjallarhorn's hunt is complete![/]")

    # Generate embeddings
    print("Generating embeddings...")
    model.eval()
    with torch.no_grad():
        train_embeddings = model(train_features, train_edge_index).cpu().numpy()
        val_embeddings = model(val_features, val_edge_index).cpu().numpy()  # Use val_edge_index
        test_embeddings = model(test_features, test_edge_index).cpu().numpy()  # Use test_edge_index

    # Save embeddings
    np.save('/content/drive/MyDrive/modbus/train_embeddings.npy', train_embeddings)
    np.save('/content/drive/MyDrive/modbus/val_embeddings.npy', val_embeddings)
    np.save('/content/drive/MyDrive/modbus/test_embeddings.npy', test_embeddings)

    print("Embeddings saved to Google Drive!")

    # Visualization
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    print("Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    train_tsne = tsne.fit_transform(train_embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], alpha=0.5)
    plt.title('t-SNE visualization of UHG embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()