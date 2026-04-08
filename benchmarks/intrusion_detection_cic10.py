"""
Benchmark: Intrusion Detection on 10% CIC data with Projective UHG GraphSAGE
- Full-batch training per epoch
- Mixed precision
- JSON metrics written to RESULTS_PATH
"""

import os, sys, time, json, platform
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from torch_geometric.data import Data

from uhg.projective import ProjectiveUHG
from uhg.nn.layers.sage import ProjectiveSAGEConv

# Paths
FILE_PATH = '/content/drive/MyDrive/CIC_data.csv'
MODEL_SAVE_PATH = '/content/drive/MyDrive/uhg_ids_model.pth'
RESULTS_PATH = '/content/drive/MyDrive/uhg_ids_results'
os.makedirs(RESULTS_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# Helpers

def env_info():
    info = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda})
    return info

# UHG helpers
uhg = ProjectiveUHG(epsilon=1e-9)

def projective_normalize(x: torch.Tensor) -> torch.Tensor:
    spatial = x[..., :-1]
    time_like = torch.sqrt(torch.clamp(1.0 + (spatial * spatial).sum(dim=-1, keepdim=True), min=1e-9))
    return torch.cat([spatial, time_like], dim=-1)

def uhg_quadrance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx = uhg.inner_product(x, x)
    yy = uhg.inner_product(y, y)
    xy = uhg.inner_product(x, y)
    denom = torch.clamp(xx * yy, min=1e-9)
    q = 1.0 - (xy * xy) / denom
    return torch.clamp(q, 0.0, 1.0)

# Data

def load_and_preprocess(file_path: str):
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].str.strip()
    labels = df['Label']
    unique = labels.unique()
    df = df.sample(frac=0.10, random_state=42)
    feats = df.drop(columns=['Label']).apply(pd.to_numeric, errors='coerce')
    feats = feats.fillna(feats.mean()).replace([np.inf,-np.inf], np.nan).fillna(feats.max())
    if feats.isnull().values.any():
        feats = feats.fillna(0)
    scaler = StandardScaler()
    X = torch.tensor(scaler.fit_transform(feats), dtype=torch.float32)
    y = torch.tensor(labels.map({l:i for i,l in enumerate(unique)}).values, dtype=torch.long)
    return X, y, {l:i for i,l in enumerate(unique)}

# Graph

def make_graph(node_features: torch.Tensor, labels: torch.Tensor, k: int = 2) -> Data:
    feats_np = node_features.cpu().numpy()
    knn = kneighbors_graph(feats_np, k, mode='connectivity', include_self=False, n_jobs=-1)
    coo = coo_matrix(knn)
    edge_index = torch.from_numpy(np.vstack((coo.row, coo.col))).long().to(device)
    x = torch.cat([node_features.to(device), torch.ones(node_features.size(0),1, device=device)], dim=1)
    x = projective_normalize(x)
    N = x.size(0)
    idx = torch.randperm(N)
    train = torch.zeros(N, dtype=torch.bool, device=device)
    val = torch.zeros(N, dtype=torch.bool, device=device)
    test = torch.zeros(N, dtype=torch.bool, device=device)
    train[idx[:int(0.7*N)]] = True
    val[idx[int(0.7*N):int(0.85*N)]] = True
    test[idx[int(0.85*N):]] = True
    return Data(x=x, edge_index=edge_index, y=labels.to(device), train_mask=train, val_mask=val, test_mask=test).to(device)

# Model

class ProjectiveGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        actual_in = in_channels - 1
        self.layers.append(ProjectiveSAGEConv(actual_in, hidden))
        for _ in range(num_layers-2):
            self.layers.append(ProjectiveSAGEConv(hidden, hidden))
        self.layers.append(ProjectiveSAGEConv(hidden, out_channels))
    def forward(self, x, edge_index):
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, edge_index)
            spatial = F.relu(h[:, :-1])
            h = torch.cat([spatial, h[:, -1:]], dim=1)
            h = self.dropout(h)
        h = self.layers[-1](h, edge_index)
        return h[:, :-1]

@torch.no_grad()
def evaluate(model, data: Data, mask: torch.Tensor) -> float:
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[mask].argmax(dim=1)
    return (pred == data.y[mask]).float().mean().item()

def main():
    run_id = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    metrics = {'env': env_info(), 'run_id': run_id, 'timing': {}, 'train': {'epochs': []}}
    t0 = time.perf_counter()
    X, y, label_map = load_and_preprocess(FILE_PATH)
    t1 = time.perf_counter(); metrics['timing']['data_s'] = t1 - t0
    G0 = time.perf_counter(); data = make_graph(X, y); G1 = time.perf_counter(); metrics['timing']['graph_s'] = G1 - G0

    model = ProjectiveGraphSAGE(in_channels=data.x.size(1), hidden=128, out_channels=len(label_map), num_layers=2).to(device)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        model = torch.compile(model, mode='reduce-overhead')
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()

    epochs = 100
    print('\nStarting training...')
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    train_start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(data.x, data.edge_index)
            loss = crit(logits[data.train_mask], data.y[data.train_mask])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        val = evaluate(model, data, data.val_mask)
        test = evaluate(model, data, data.test_mask)
        sched.step(val)
        lr = opt.param_groups[0]['lr']
        ep = {'epoch': epoch, 'loss': float(loss.item()), 'val': float(val), 'test': float(test), 'lr': float(lr)}
        metrics['train']['epochs'].append(ep)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val {val:.4f} | Test {test:.4f} | LR {lr:.5f}")
    metrics['timing']['train_s'] = time.perf_counter() - train_start
    metrics['timing']['total_s'] = time.perf_counter() - t0

    out = os.path.join(RESULTS_PATH, f"benchmark_metrics_{run_id}.json")
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {out}")

if __name__ == '__main__':
    main() 