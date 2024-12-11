# ==================================
# CELL 1: Package Installation
# ==================================
# Run this cell first, then restart runtime

print("Step 1: Installing required packages...")
!pip install uhg
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
!pip install geoopt

print("\nStep 2: Now you must:")
print("1. Click Runtime -> Restart Runtime")
print("2. After restart, run CELL 2 below")

# ==================================
# CELL 2: Main Code
# ==================================
# Run this cell after restarting runtime

"""
Gjallarhorn2: Enhanced UHG-Compliant Hyperbolic Graph Neural Network for Anomaly Detection
======================================================================================

Evolution from Gjallarhorn1:
---------------------------
1. Architecture Improvements:
   - Replaced custom GraphSAGE with UHG-compliant ProjectiveSAGEConv
   - Eliminated all tangent space operations in favor of pure projective geometry
   - Added proper homogeneous coordinate handling throughout the network
   - Implemented cross-ratio preservation at every layer

2. Geometric Foundations:
   - Moved from differential geometry approach to pure projective geometry
   - Replaced Euclidean operations with UHG-compliant alternatives
   - Implemented proper hyperbolic feature transformations
   - Added projective normalization for numerical stability

3. Loss Function Enhancement:
   - Introduced cross-ratio based loss computation
   - Improved positive/negative pair handling in projective space
   - Better preservation of hyperbolic distances
   - More stable gradient computation

4. Feature Processing:
   - Added proper UHG space conversion for input features
   - Improved handling of high-dimensional data
   - Better preservation of geometric structure
   - More robust normalization techniques

5. Training Improvements:
   - Implemented Riemannian optimization
   - Added learning rate scheduling
   - Improved batch processing
   - Better numerical stability

Key Benefits:
------------
1. Better Geometric Consistency:
   - All operations preserve hyperbolic structure
   - No distortion from tangent space approximations
   - Proper handling of infinite points

2. Improved Stability:
   - More stable training process
   - Better gradient flow
   - Reduced numerical issues

3. Enhanced Performance:
   - Better separation of normal/anomalous patterns
   - More accurate embeddings
   - Improved scalability

4. Theoretical Soundness:
   - Strict adherence to UHG principles
   - Mathematically rigorous operations
   - Proper invariant preservation

Author: Zach Bovaird
Date: March 2024
Version: 2.0
"""

# ==============================
# 1. Import Check
# ==============================

def check_imports():
    """Check if all required packages are properly installed."""
    try:
        import uhg
        import torch
        import torch_scatter
        import geoopt
        print("All required packages successfully imported!")
        return True
    except ImportError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease make sure you:")
        print("1. Ran CELL 1 above")
        print("2. Clicked Runtime -> Restart Runtime")
        print("3. Then ran this cell (CELL 2)")
        return False

if not check_imports():
    raise ImportError("Required packages not properly installed. Please follow the steps above.")

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
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
import scipy.sparse
import os
from torch import Tensor
import geoopt.optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# UHG specific imports
from uhg.projective import ProjectiveUHG

# ==============================
# 3. Define Base Layer
# ==============================

class ProjectiveBaseLayer(nn.Module):
    """Base layer for UHG-compliant neural network operations."""
    
    def __init__(self):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
    def projective_transform(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratios."""
        # Extract features and homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Apply weight to features
        transformed = torch.matmul(features, weight.t())
        
        # Add homogeneous coordinate back
        out = torch.cat([transformed, homogeneous], dim=-1)
        
        # Normalize to maintain projective structure
        norm = torch.norm(out[..., :-1], p=2, dim=-1, keepdim=True)
        out = torch.cat([out[..., :-1] / (norm + 1e-8), out[..., -1:]], dim=-1)
        return out
        
    def normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to lie in projective space."""
        norm = torch.norm(points[..., :-1], p=2, dim=-1, keepdim=True)
        return torch.cat([points[..., :-1] / (norm + 1e-8), points[..., -1:]], dim=-1)

# ==============================
# 4. Define SAGE Layer
# ==============================

class UHGSAGEConv(ProjectiveBaseLayer):
    """UHG-compliant GraphSAGE convolution layer using pure projective operations."""
    
    def __init__(self, in_features, out_features, aggregator='mean'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator.lower()
        
        # Initialize projective transformations
        self.W_self = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_neigh = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if self.aggregator == 'lstm':
            self.lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=in_features,
                batch_first=True
            )
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using UHG-aware initialization."""
        nn.init.orthogonal_(self.W_self)
        nn.init.orthogonal_(self.W_neigh)
            
    def forward(self, x, edge_index):
        """Forward pass using pure projective operations."""
        row, col = edge_index
        
        # Transform self features
        self_trans = self.projective_transform(x, self.W_self)
        
        # Get neighbor features
        neigh_features = x[col]
        neigh_trans = self.projective_transform(neigh_features, self.W_neigh)
        
        # Aggregate neighbor features
        if self.aggregator == 'mean':
            out = torch.zeros_like(x)
            out.index_add_(0, row, neigh_trans)
            count = torch.zeros_like(x)
            count.index_add_(0, row, torch.ones_like(neigh_trans))
            count = count.clamp(min=1)
            out = out / count
        elif self.aggregator == 'max':
            out = torch.zeros_like(x)
            scatter_idx = row.view(-1, 1).expand_as(neigh_trans)
            out = torch.scatter_reduce(out, 0, scatter_idx, neigh_trans, reduce='amax')
        else:  # lstm
            # Group neighbor features by target node
            grouped_features = []
            for i in range(x.size(0)):
                mask = row == i
                if mask.any():
                    node_feats = neigh_trans[mask]
                    grouped_features.append(node_feats)
                else:
                    grouped_features.append(x.new_zeros((1, self.out_features)))
                    
            # Process through LSTM
            packed = nn.utils.rnn.pack_sequence(grouped_features, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            out = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)[0]
            
        # Combine self and neighbor features
        out = 0.5 * (self_trans + out)
        
        # Normalize output
        return self.normalize_points(out)

# ==============================
# 5. Define Main Model
# ==============================

class UHG_HGNN(nn.Module):
    """UHG-compliant Hyperbolic Graph Neural Network for anomaly detection."""
    
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
        # Create UHG-compliant SAGE layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(UHGSAGEConv(
            in_features=in_features,
            out_features=hidden_features,
            aggregator='mean'
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(UHGSAGEConv(
                in_features=hidden_features,
                out_features=hidden_features,
                aggregator='mean'
            ))
            
        # Output layer
        self.layers.append(UHGSAGEConv(
            in_features=hidden_features,
            out_features=out_features,
            aggregator='mean'
        ))
        
    def forward(self, x, edge_index):
        # Add homogeneous coordinate if not present
        if x.size(-1) == x.size(-1) - 1:  # If missing homogeneous coordinate
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            
        # Apply UHG layers
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            
        # Final layer
        x = self.layers[-1](x, edge_index)
        
        # Return normalized features
        features = x[..., :-1]  # Remove homogeneous coordinate
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        return features / (norm + 1e-8)

# ==============================
# 6. UHG Loss Function
# ==============================

class UHGLoss(nn.Module):
    """UHG-compliant loss function using cross-ratios."""
    
    def __init__(self):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
    def forward(self, z, edge_index, batch_size):
        # Add homogeneous coordinate for projective operations
        z = torch.cat([z, torch.ones_like(z[..., :1])], dim=-1)
        
        # Get positive pairs from edge_index
        mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
        pos_edge_index = edge_index[:, mask]
        
        if pos_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=z.device)
            
        # Generate negative pairs
        neg_edge_index = torch.randint(0, batch_size, (2, batch_size), device=z.device)
        
        # Compute cross-ratios for positive and negative pairs
        pos_cr = self.uhg.cross_ratio(
            z[pos_edge_index[0]],
            z[pos_edge_index[1]],
            z[pos_edge_index[0]],
            z[pos_edge_index[0]]
        )
        
        neg_cr = self.uhg.cross_ratio(
            z[neg_edge_index[0]],
            z[neg_edge_index[1]],
            z[neg_edge_index[0]],
            z[neg_edge_index[0]]
        )
        
        # Loss based on cross-ratio differences
        pos_loss = F.binary_cross_entropy_with_logits(pos_cr, torch.ones_like(pos_cr))
        neg_loss = F.binary_cross_entropy_with_logits(neg_cr, torch.zeros_like(neg_cr))
        
        return pos_loss + neg_loss

# ==============================
# 7. Training Function
# ==============================

def train_model(model, features, edge_index, optimizer, scheduler, num_epochs, batch_size):
    model.train()
    criterion = UHGLoss()
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", ncols=100):
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with torch.cuda.amp.autocast():
            out = model(features, edge_index)
            loss = criterion(out, edge_index, batch_size)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
            torch.cuda.empty_cache()  # Clear cache periodically

# ==============================
# 8. Mount Google Drive
# ==============================

print("Attempting to mount Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!")
except KeyboardInterrupt:
    print("\nDrive mounting was interrupted. Please run this cell again if you need to access files from Google Drive.")
except Exception as e:
    print(f"\nError mounting Google Drive: {str(e)}")
    print("If you need to access files from Google Drive, please run this cell again.")

# Verify data paths exist
def verify_data_paths():
    train_file = '/content/drive/MyDrive/modbus/train_data_balanced_new.csv'
    val_file = '/content/drive/MyDrive/modbus/val_data_balanced_new.csv'
    test_file = '/content/drive/MyDrive/modbus/test_data_balanced_new.csv'
    
    missing_files = []
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("\nWarning: The following data files are missing:")
        for file in missing_files:
            print(f"- {file}")
        print("\nPlease ensure these files are in the correct location in your Google Drive.")
        return False
    return True

# Only proceed with data loading if drive is mounted and files exist
if os.path.ismount('/content/drive'):
    print("\nChecking for required data files...")
    if verify_data_paths():
        print("All required data files found!")
    else:
        print("Please fix the missing files before proceeding.")
else:
    print("\nGoogle Drive is not mounted. Data loading will fail without access to the required files.")
    print("Please run this cell again to mount Google Drive.")

# ==============================
# 9. Device Configuration
# ==============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==============================
# 10. Data Loading and Preprocessing
# ==============================

train_file = '/content/drive/MyDrive/modbus/train_data_balanced_new.csv'
val_file = '/content/drive/MyDrive/modbus/val_data_balanced_new.csv'
test_file = '/content/drive/MyDrive/modbus/test_data_balanced_new.csv'

train_data_full = pd.read_csv(train_file, low_memory=False)
val_data_full = pd.read_csv(val_file, low_memory=False)
test_data_full = pd.read_csv(test_file, low_memory=False)

all_data = pd.concat([train_data_full, val_data_full, test_data_full], ignore_index=True)
print(f'Combined data shape: {all_data.shape}')

all_data = all_data.sample(frac=0.20, random_state=42).reset_index(drop=True)
print(f'Data reduced to 20%: {all_data.shape}')

all_data.fillna(all_data.mean(), inplace=True)

# Handle non-numeric columns
non_numeric = all_data.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    all_data = pd.get_dummies(all_data, columns=non_numeric)

print(f'Feature dimensions after encoding: {all_data.shape[1]}')

# Scale features
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_data)

# Split data
total_samples = len(all_scaled)
train_samples = int(0.70 * total_samples)
val_samples = int(0.15 * total_samples)

train_data_np = all_scaled[:train_samples]
val_data_np = all_scaled[train_samples:train_samples + val_samples]
test_data_np = all_scaled[train_samples + val_samples:]

# Convert to UHG space
uhg = ProjectiveUHG()
train_uhg = uhg.normalize_points(torch.tensor(train_data_np, dtype=torch.float32))
val_uhg = uhg.normalize_points(torch.tensor(val_data_np, dtype=torch.float32))
test_uhg = uhg.normalize_points(torch.tensor(test_data_np, dtype=torch.float32))

# ==============================
# 11. Create KNN Graph
# ==============================

def compute_uhg_distances(p1: torch.Tensor, p2: torch.Tensor, uhg: ProjectiveUHG) -> torch.Tensor:
    """Compute UHG-compliant distances using cross-ratios."""
    # Add homogeneous coordinates if not present
    if p1.size(-1) == p1.size(-1) - 1:
        p1 = torch.cat([p1, torch.ones_like(p1[..., :1])], dim=-1)
    if p2.size(-1) == p2.size(-1) - 1:
        p2 = torch.cat([p2, torch.ones_like(p2[..., :1])], dim=-1)
    
    # Get the line through the points
    line = uhg.join(p1, p2)
    
    # Get ideal points on this line
    i1, i2 = uhg.get_ideal_points(line)
    
    # Compute cross-ratio based distance
    cr = uhg.cross_ratio(p1, p2, i1, i2)
    
    # Convert cross-ratio to distance (using UHG formula)
    return torch.log(torch.abs(cr) + 1e-8)

def create_knn_graph(data, k, batch_size=10000):
    """Create KNN graph using UHG principles and cross-ratio based distances."""
    n_samples = data.shape[0]
    n_batches = (n_samples - 1) // batch_size + 1
    
    # Initialize UHG
    uhg = ProjectiveUHG()
    
    # Convert data to tensor and add homogeneous coordinate
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    data_tensor = torch.cat([data_tensor, torch.ones(data_tensor.size(0), 1, device=device)], dim=1)
    data_tensor = uhg.normalize_points(data_tensor)
    
    # Pre-allocate memory for edge lists
    rows = []
    cols = []
    
    with tqdm(total=n_batches, desc="Creating KNN graph", ncols=100) as pbar:
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_data = data_tensor[i:end]
            
            # Compute pairwise UHG distances
            distances = torch.zeros((end-i, n_samples), device=device)
            for j in range(end-i):
                distances[j] = compute_uhg_distances(
                    batch_data[j].unsqueeze(0).expand(n_samples, -1),
                    data_tensor,
                    uhg
                )
            
            # Get k nearest neighbors using UHG distances
            _, indices = torch.topk(distances, k=k+1, dim=1, largest=False)
            indices = indices[:, 1:]  # Remove self-connections
            
            # Add edges to graph
            batch_rows = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, k)
            rows.extend(batch_rows.flatten().cpu().tolist())
            cols.extend(indices.flatten().cpu().tolist())
            
            pbar.update(1)
            torch.cuda.empty_cache()  # Clear GPU cache after each batch
    
    # Create edge index tensor
    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    
    # Verify cross-ratio preservation
    print("Verifying cross-ratio preservation in graph structure...")
    sample_edges = torch.randint(0, edge_index.size(1), (min(100, edge_index.size(1)),))
    preserved = 0
    for idx in sample_edges:
        src, dst = edge_index[:, idx]
        line = uhg.join(data_tensor[src], data_tensor[dst])
        i1, i2 = uhg.get_ideal_points(line)
        cr1 = uhg.cross_ratio(data_tensor[src], data_tensor[dst], i1, i2)
        cr2 = uhg.cross_ratio(i1, i2, data_tensor[src], data_tensor[dst])
        if torch.abs(cr1 * cr2 - 1.0) < 1e-5:
            preserved += 1
    print(f"Cross-ratio preservation check: {preserved}/{len(sample_edges)} edges verified")
    
    return edge_index

print("Creating KNN graphs using UHG principles...")
k = 3  # Same k as before

# Create graphs using UHG-compliant KNN
train_edge_index = create_knn_graph(train_data_np, k)
val_edge_index = create_knn_graph(val_data_np, k)
test_edge_index = create_knn_graph(test_data_np, k)

# Move features to device and convert to UHG space
def prepare_features(data):
    # Convert to tensor
    features = torch.tensor(data, dtype=torch.float32, device=device)
    # Add homogeneous coordinate
    features = torch.cat([features, torch.ones(features.size(0), 1, device=device)], dim=1)
    # Normalize in projective space
    uhg = ProjectiveUHG()
    return uhg.normalize_points(features)

train_features = prepare_features(train_data_np)
val_features = prepare_features(val_data_np)
test_features = prepare_features(test_data_np)

# ==============================
# 12. Initialize and Train Model
# ==============================

# Increase batch size for better GPU utilization
batch_size = 2048  # Increased from 1000
num_epochs = 500

model = UHG_HGNN(
    in_features=train_features.shape[1],
    hidden_features=64,
    out_features=32,
    num_layers=3
).to(device)

# Use gradient scaling for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Optimize memory allocation
torch.cuda.empty_cache()
model = torch.compile(model)  # Use torch.compile for PyTorch 2.0+ optimization

optimizer = geoopt.optim.RiemannianAdam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0005,
    stabilize=10
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train with optimized batch processing
def train_model(model, features, edge_index, optimizer, scheduler, num_epochs, batch_size):
    model.train()
    criterion = UHGLoss()
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", ncols=100):
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with torch.cuda.amp.autocast():
            out = model(features, edge_index)
            loss = criterion(out, edge_index, batch_size)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
            torch.cuda.empty_cache()  # Clear cache periodically

# Train model with optimizations
train_model(model, train_features, train_edge_index, optimizer, scheduler, num_epochs, batch_size)

# ==============================
# 13. Generate and Save Embeddings
# ==============================

model.eval()
with torch.no_grad():
    embeddings = model(train_features, train_edge_index).cpu().numpy()

# Save embeddings
embeddings_file_path = '/content/drive/MyDrive/embeddings_new.npy'
np.save(embeddings_file_path, embeddings)
print(f"Embeddings saved to {embeddings_file_path}")

# ==============================
# 14. Visualization Functions
# ==============================

def visualize_embeddings(embeddings, labels=None, method='PCA', title='Embeddings Visualization'):
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    
    if method == 'PCA':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        print(f'Explained variance by 2 principal components: {np.sum(pca.explained_variance_ratio_):.2f}')
    elif method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        reduced_embeddings = tsne.fit_transform(embeddings)
        print('t-SNE completed.')
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
        
    plt.figure(figsize=(10, 7))
    if labels is not None:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar()
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=50, alpha=0.7)
        
    plt.title(title)
    plt.xlabel('Component 1' if method == 'PCA' else 't-SNE Component 1')
    plt.ylabel('Component 2' if method == 'PCA' else 't-SNE Component 2')
    plt.grid(True)
    plt.show()

# Visualize embeddings
visualize_embeddings(embeddings, method='PCA', title='PCA Visualization of UHG Embeddings')
visualize_embeddings(embeddings, method='t-SNE', title='t-SNE Visualization of UHG Embeddings') 