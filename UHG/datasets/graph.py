import torch
from typing import Optional, Dict, List, Union, Tuple
from torch_geometric.data import Data
from .base import HyperbolicDataset
from ..manifolds.base import Manifold

class HyperbolicGraphDataset(HyperbolicDataset):
    """Dataset class for graphs in hyperbolic space.
    
    Extends HyperbolicDataset to handle graph-structured data
    using UHG principles, without tangent space mappings.
    
    Args:
        manifold: The hyperbolic manifold for the data
        node_features: Node feature tensors in hyperbolic space
        edge_index: Graph connectivity (COO format)
        edge_attr: Optional edge features
        node_labels: Optional node labels
        graph_labels: Optional graph labels
    """
    def __init__(
        self,
        manifold: Manifold,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None
    ):
        super().__init__(manifold, node_features, node_labels)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.graph_labels = graph_labels
        
    def __getitem__(self, idx: int) -> Data:
        """Get a graph from the dataset.
        
        Returns a PyG Data object with hyperbolic features.
        
        Args:
            idx: Graph index
            
        Returns:
            PyG Data object
        """
        data = Data(
            x=self.points[idx],
            edge_index=self.edge_index[idx] if self.edge_index.dim() > 2 else self.edge_index,
            edge_attr=self.edge_attr[idx] if self.edge_attr is not None else None,
            y=self.graph_labels[idx] if self.graph_labels is not None else None
        )
        
        if self.labels is not None:
            data.node_labels = self.labels[idx]
            
        return data
        
    def to(self, device: torch.device) -> 'HyperbolicGraphDataset':
        """Move dataset to device."""
        node_features = self.points.to(device)
        edge_index = self.edge_index.to(device)
        edge_attr = self.edge_attr.to(device) if self.edge_attr is not None else None
        node_labels = self.labels.to(device) if self.labels is not None else None
        graph_labels = self.graph_labels.to(device) if self.graph_labels is not None else None
        
        return HyperbolicGraphDataset(
            self.manifold,
            node_features,
            edge_index,
            edge_attr,
            node_labels,
            graph_labels
        )
        
    def get_neighbors(
        self,
        node_idx: int,
        edge_index: Optional[torch.Tensor] = None
    ) -> List[int]:
        """Get indices of neighboring nodes.
        
        Args:
            node_idx: Target node index
            edge_index: Optional custom edge index
            
        Returns:
            List of neighbor indices
        """
        if edge_index is None:
            edge_index = self.edge_index
            
        mask = edge_index[0] == node_idx
        return edge_index[1, mask].tolist()
        
    def compute_edge_distances(self) -> torch.Tensor:
        """Compute distances along graph edges.
        
        Returns:
            Tensor of edge distances
        """
        src, dst = self.edge_index
        return torch.tensor([
            self.manifold.dist(self.points[s], self.points[d])
            for s, d in zip(src, dst)
        ])
        
    def get_subgraph(
        self,
        node_idx: int,
        n_hops: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract local subgraph around node.
        
        Args:
            node_idx: Center node index
            n_hops: Number of hops to include
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
        """
        # Get n-hop neighborhood
        nodes = {node_idx}
        current_nodes = {node_idx}
        
        for _ in range(n_hops):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(self.get_neighbors(node))
            current_nodes = next_nodes - nodes
            nodes.update(current_nodes)
            
        # Get node indices and features
        node_list = sorted(nodes)
        node_features = self.points[node_list]
        
        # Get edges within subgraph
        src, dst = self.edge_index
        mask = torch.tensor([
            s in nodes and d in nodes
            for s, d in zip(src, dst)
        ])
        
        edge_index = self.edge_index[:, mask]
        edge_attr = self.edge_attr[mask] if self.edge_attr is not None else None
        
        # Remap node indices
        node_map = {old: new for new, old in enumerate(node_list)}
        edge_index = torch.tensor([
            [node_map[idx.item()] for idx in edge_index[0]],
            [node_map[idx.item()] for idx in edge_index[1]]
        ])
        
        return node_features, edge_index, edge_attr
        
    @classmethod
    def from_networkx(
        cls,
        manifold: Manifold,
        G: 'networkx.Graph',
        node_features: Optional[torch.Tensor] = None
    ) -> 'HyperbolicGraphDataset':
        """Create dataset from NetworkX graph.
        
        Args:
            manifold: Target hyperbolic manifold
            G: NetworkX graph
            node_features: Optional node features
            
        Returns:
            HyperbolicGraphDataset
        """
        import networkx as nx
        
        # Get edge index
        edge_index = torch.tensor([
            [e[0] for e in G.edges()],
            [e[1] for e in G.edges()]
        ])
        
        # Create random features if none provided
        if node_features is None:
            node_features = torch.randn(G.number_of_nodes(), manifold.dim)
            node_features = manifold.project_from_euclidean(node_features)
            
        return cls(manifold, node_features, edge_index)
        
    def to_networkx(self) -> 'networkx.Graph':
        """Convert dataset to NetworkX graph.
        
        Returns:
            NetworkX graph
        """
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes with features
        for i, features in enumerate(self.points):
            G.add_node(i, features=features)
            
        # Add edges
        src, dst = self.edge_index
        edges = list(zip(src.tolist(), dst.tolist()))
        
        if self.edge_attr is not None:
            edges = [
                (s, d, {'features': attr})
                for (s, d), attr in zip(edges, self.edge_attr)
            ]
            
        G.add_edges_from(edges)
        return G 