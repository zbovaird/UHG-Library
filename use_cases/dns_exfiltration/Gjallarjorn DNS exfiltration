"""
Gjallarhorn DNS Exfiltration Detection Module
Uses Universal Hyperbolic Graph Neural Networks for DNS exfiltration detection.
Integrates with Gjallarhorn's core UHG architecture for enhanced detection capabilities.
"""

# Import shared theme configuration
from theme_config import console

# Standard library imports
import os
import math
import string
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from rich.progress import Progress, SpinnerColumn, TextColumn
from uhg.projective import ProjectiveUHG

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_DIRECTORY = "/content/models"
PRINTABLE_CHARS = list(string.printable.strip())
BATCH_SIZE = 256
THRESHOLD = 0.5

class DNSFeatureExtractor:
    """Enhanced DNS feature extraction using UHG principles."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.char_scaler = StandardScaler()
        self.printable_chars = set(PRINTABLE_CHARS)
        
    def compute_entropy(self, domain: str) -> float:
        """Compute Shannon entropy of domain string."""
        char_counts = Counter(domain)
        domain_len = float(len(domain))
        return -sum(count / domain_len * math.log(count / domain_len, 2) 
                   for count in char_counts.values())
    
    def extract_domain_features(self, query: str) -> Dict[str, float]:
        """Extract comprehensive domain features."""
        # Basic features
        subdomain = str(query).rsplit('.', 2)[0]
        
        # Character-level features
        char_freq = {char: query.count(char) / len(query) 
                    for char in set(query) & self.printable_chars}
        
        # Statistical features
        features = {
            'length': len(query),
            'entropy': self.compute_entropy(query),
            'unique_chars': len(set(query)),
            'digit_ratio': sum(c.isdigit() for c in query) / len(query),
            'alpha_ratio': sum(c.isalpha() for c in query) / len(query),
            'subdomain_depth': query.count('.'),
            'max_label_length': max(len(label) for label in query.split('.')),
            'consonant_ratio': sum(c.lower() in 'bcdfghjklmnpqrstvwxyz' 
                                 for c in query) / len(query)
        }
        
        # Add character frequencies
        features.update({f'char_freq_{char}': freq 
                        for char, freq in char_freq.items()})
        
        return features

    def build_graph(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph representation of DNS queries."""
        console.print("[info]Building graph representation...[/]")
        
        # Create node features
        features_list = []
        for _, row in df.iterrows():
            features = self.extract_domain_features(row['query'])
            features_list.append(list(features.values()))
        
        # Scale features
        node_features = self.scaler.fit_transform(np.array(features_list))
        
        # Build adjacency matrix using k-nearest neighbors
        adj_matrix = kneighbors_graph(
            node_features, 
            n_neighbors=min(10, len(node_features)),
            mode='distance',
            metric='cosine'
        )
        
        # Convert to PyTorch Geometric format
        edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
        
        console.print("[success]Graph construction complete[/]")
        return torch.FloatTensor(node_features), edge_index

class UHGDNSModel(nn.Module):
    """UHG-based model for DNS exfiltration detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.uhg = ProjectiveUHG(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
            dropout=0.2
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through UHG and MLP layers."""
        # UHG embedding
        h = self.uhg(x, edge_index)
        
        # Final classification
        return self.mlp(h)

class GjallarhornDNSDetector:
    """Main class for DNS exfiltration detection using UHG."""
    
    def __init__(self):
        self.feature_extractor = DNSFeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_dim = None
        console.print(f"[info]Using device: {self.device}[/]")
    
    def initialize_model(self, input_dim: int):
        """Initialize or update model with correct input dimension."""
        self.input_dim = input_dim
        self.model = UHGDNSModel(input_dim).to(self.device)
        console.print(f"[success]Model initialized with input dimension: {input_dim}[/]")
    
    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
             epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Train the model on DNS query data."""
        console.print("[info]Starting model training...[/]")
        
        # Extract features and build graph
        node_features, edge_index = self.feature_extractor.build_graph(train_df)
        
        # Initialize model if needed
        if self.model is None:
            self.initialize_model(node_features.shape[1])
        
        # Prepare training data
        labels = torch.FloatTensor(train_df['is_exfiltration'].values).to(self.device)
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training progress
        history = {'loss': [], 'val_loss': []}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Training...", total=epochs)
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                optimizer.zero_grad()
                
                outputs = self.model(node_features, edge_index)
                loss = criterion(outputs.squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                
                history['loss'].append(loss.item())
                
                # Validation
                if val_df is not None:
                    val_features, val_edge_index = self.feature_extractor.build_graph(val_df)
                    val_labels = torch.FloatTensor(val_df['is_exfiltration'].values).to(self.device)
                    
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(val_features.to(self.device), 
                                              val_edge_index.to(self.device))
                        val_loss = criterion(val_outputs.squeeze(), val_labels)
                        history['val_loss'].append(val_loss.item())
                
                progress.update(task, advance=1)
        
        console.print("[success]Training complete![/]")
        return history
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict DNS exfiltration probability for queries."""
        console.print("[info]Making predictions...[/]")
        
        # Extract features and build graph
        node_features, edge_index = self.feature_extractor.build_graph(df)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            node_features = node_features.to(self.device)
            edge_index = edge_index.to(self.device)
            
            outputs = self.model(node_features, edge_index)
            predictions = outputs.cpu().numpy()
        
        # Add predictions to dataframe
        df['exfiltration_probability'] = predictions
        df['is_exfiltration'] = predictions >= THRESHOLD
        
        console.print("[success]Predictions complete![/]")
        return df
    
    def save(self, path: str):
        """Save model and feature extractors."""
        if not os.path.exists(MODEL_DIRECTORY):
            os.makedirs(MODEL_DIRECTORY)
            
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'feature_scaler': self.feature_extractor.scaler,
            'char_scaler': self.feature_extractor.char_scaler
        }
        
        torch.save(model_state, os.path.join(MODEL_DIRECTORY, path))
        console.print(f"[success]Model saved to {path}[/]")
    
    def load(self, path: str):
        """Load model and feature extractors."""
        model_state = torch.load(os.path.join(MODEL_DIRECTORY, path), 
                               map_location=self.device)
        
        self.input_dim = model_state['input_dim']
        self.initialize_model(self.input_dim)
        
        self.model.load_state_dict(model_state['model_state_dict'])
        self.feature_extractor.scaler = model_state['feature_scaler']
        self.feature_extractor.char_scaler = model_state['char_scaler']
        console.print(f"[success]Model loaded from {path}[/]")

def init(df: pd.DataFrame, param: Dict[str, Any]) -> GjallarhornDNSDetector:
    """Initialize the DNS detector."""
    detector = GjallarhornDNSDetector()
    if os.path.exists(os.path.join(MODEL_DIRECTORY, 'dns_model.pt')):
        detector.load('dns_model.pt')
    return detector

def fit(model: GjallarhornDNSDetector, df: pd.DataFrame, 
        param: Dict[str, Any]) -> Dict[str, str]:
    """Train the model on new data."""
    history = model.train(df)
    model.save('dns_model.pt')
    return {"message": "Model trained successfully", "history": str(history)}

def apply(model: GjallarhornDNSDetector, df: pd.DataFrame, 
         param: Dict[str, Any]) -> pd.DataFrame:
    """Apply the model to new data."""
    return model.predict(df)

if __name__ == "__main__":
    # Example usage
    console.print("[cyan]Initializing Gjallarhorn DNS Exfiltration Detector...[/]")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'query': [
            'normal-domain.com',
            'a'.join(['x' * 20]) + '.evil.com',
            'api.legitimate-service.com',
            'b64.' + 'a' * 50 + '.attacker.net'
        ],
        'is_exfiltration': [0, 1, 0, 1]
    })
    
    # Initialize detector
    detector = GjallarhornDNSDetector()
    
    # Train on sample data
    console.print("[cyan]Training on sample data...[/]")
    detector.train(sample_data)
    
    # Make predictions
    console.print("[cyan]Making predictions...[/]")
    results = detector.predict(sample_data)
    
    # Print results
    console.print("\n[green]Results:[/]")
    for _, row in results.iterrows():
        console.print(f"Query: {row['query']}")
        console.print(f"Probability: {row['exfiltration_probability']:.3f}")
        console.print(f"Detected: {'Yes' if row['is_exfiltration'] else 'No'}\n") 