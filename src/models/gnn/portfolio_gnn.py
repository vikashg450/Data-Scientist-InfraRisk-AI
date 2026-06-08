import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

# Try to import PyTorch Geometric, fallback to pure PyTorch GCN if not available
try:
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# Fallback GCN Layer implemented in pure PyTorch
class PurePyTorchGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(PurePyTorchGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (N, in_features)
        edge_index: (2, E)
        edge_weight: (E,) or None
        """
        num_nodes = x.size(0)
        
        # Build adjacency matrix A
        A = torch.zeros((num_nodes, num_nodes), device=x.device)
        if edge_weight is not None:
            A[edge_index[0], edge_index[1]] = edge_weight
        else:
            A[edge_index[0], edge_index[1]] = 1.0
            
        # Add self-loops (A_tilde = A + I)
        A_tilde = A + torch.eye(num_nodes, device=x.device)
        
        # Compute degree matrix D_tilde
        D_diag = torch.sum(A_tilde, dim=1)
        # Compute D_tilde^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(D_diag, min=1e-6)))
        
        # Normalize adjacency matrix: D^{-1/2} A D^{-1/2}
        A_norm = torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)
        
        # GCN forward pass: X_new = A_norm * X * W
        x_proj = self.linear(x)
        out = torch.mm(A_norm, x_proj)
        return out

class PortfolioGNN(nn.Module):
    """
    Graph Neural Network for modeling project portfolio dependency and default contagion.
    Uses GCN Conv from PyTorch Geometric, or falls back to a pure PyTorch GCN layer.
    """
    def __init__(self, in_features: int, hidden_dim: int = 32, out_dim: int = 16):
        super(PortfolioGNN, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        if PYG_AVAILABLE:
            self.conv1 = GCNConv(in_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        else:
            self.conv1 = PurePyTorchGCNLayer(in_features, hidden_dim)
            self.conv2 = PurePyTorchGCNLayer(hidden_dim, out_dim)
            
        # Head to predict probability of default
        self.pd_head = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Node features (N, in_features)
        edge_index: Graph edge indices (2, E)
        edge_weight: Weights of edges (E,)
        Returns:
            embeddings: Node embeddings (N, out_dim)
            pd_scores: Estimated default probability per node (N, 1)
        """
        # First Layer
        h = self.conv1(x, edge_index, edge_weight) if PYG_AVAILABLE else self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        
        # Second Layer
        embeddings = self.conv2(h, edge_index, edge_weight) if PYG_AVAILABLE else self.conv2(h, edge_index, edge_weight)
        
        # PD Prediction
        pd_scores = self.pd_head(embeddings)
        
        return embeddings, pd_scores

    @staticmethod
    def compute_contagion_index(
        initial_pd: np.ndarray, 
        edge_index: np.ndarray, 
        edge_weight: np.ndarray, 
        num_nodes: int, 
        alpha: float = 0.5, 
        steps: int = 5
    ) -> np.ndarray:
        """
        Runs PageRank-style default risk propagation over the network.
        Formula:
            PD_t+1 = alpha * initial_pd + (1 - alpha) * W^T * PD_t
        """
        # Create weighted adjacency matrix W (column-normalized)
        W = np.zeros((num_nodes, num_nodes))
        for i in range(len(edge_weight)):
            u, v = edge_index[0, i], edge_index[1, i]
            W[u, v] = edge_weight[i]
            
        # Column normalization
        col_sums = W.sum(axis=0)
        for j in range(num_nodes):
            if col_sums[j] > 0:
                W[:, j] = W[:, j] / col_sums[j]
                
        pd = initial_pd.copy()
        
        for _ in range(steps):
            pd = alpha * initial_pd + (1.0 - alpha) * np.dot(W.T, pd)
            
        return pd
