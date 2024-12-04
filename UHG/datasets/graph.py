import torch
from typing import Optional, Dict, List, Union, Tuple
from torch_geometric.data import Data
from .base import ProjectiveDataset
from ..projective import ProjectiveUHG

class ProjectiveGraphDataset(ProjectiveDataset):
    """Dataset class for graphs in projective space.
    
    Extends ProjectiveDataset to handle graph-structured data
    using UHG principles.
    
    Args:
        node_features: Node feature tensors in projective coordinates
        edge_index: Graph connectivity (COO format)
        edge_attr: Optional edge features
        node_labels: Optional node labels
        graph_labels: Optional graph labels
    """
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None
    ):
        super().__init__(node_features, node_labels)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.graph_labels = graph_labels
        
    def __getitem__(self, idx: int) -> Data:
        """Get a graph from the dataset.
        
        Returns a PyG Data object with projective features.
        
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
        
    def to(self, device: torch.device) -> 'ProjectiveGraphDataset':
        """Move dataset to device."""
        node_features = self.points.to(device)
        edge_index = self.edge_index.to(device)
        edge_attr = self.edge_attr.to(device) if self.edge_attr is not None else None
        node_labels = self.labels.to(device) if self.labels is not None else None
        graph_labels = self.graph_labels.to(device) if self.graph_labels is not None else None
        
        return ProjectiveGraphDataset(
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
        
    def compute_edge_cross_ratios(self) -> torch.Tensor:
        """Compute cross-ratios along graph edges.
        
        For each edge (u,v), computes CR(u,v,i1,i2) where i1,i2
        are the ideal points on the line through u,v.
        
        Returns:
            Tensor of edge cross-ratios
        """
        src, dst = self.edge_index
        cross_ratios = []
        
        for s, d in zip(src, dst):
            # Get line through edge endpoints
            line = self.uhg.join(self.points[s], self.points[d])
            
            # Get ideal points using polar
            polar = self.uhg.absolute_polar(line)
            i1 = self.uhg.meet(line, polar)
            i2 = -i1  # Opposite point on absolute
            
            # Compute cross-ratio
            cr = self.uhg.cross_ratio(
                self.points[s],
                self.points[d],
                i1,
                i2
            )
            cross_ratios.append(cr)
            
        return torch.tensor(cross_ratios)
        
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
        G: 'networkx.Graph',
        node_features: Optional[torch.Tensor] = None
    ) -> 'ProjectiveGraphDataset':
        """Create dataset from NetworkX graph.
        
        Args:
            G: NetworkX graph
            node_features: Optional node features
            
        Returns:
            ProjectiveGraphDataset
        """
        import networkx as nx
        
        # Get edge index
        edge_index = torch.tensor([
            [e[0] for e in G.edges()],
            [e[1] for e in G.edges()]
        ])
        
        # Create random features if none provided
        if node_features is None:
            uhg = ProjectiveUHG()
            matrix = uhg.get_projective_matrix(3)
            node_features = uhg.transform(torch.eye(3), matrix)[:G.number_of_nodes()]
            
        return cls(node_features, edge_index)
        
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