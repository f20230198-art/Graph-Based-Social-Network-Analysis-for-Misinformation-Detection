"""
PropNet: Propagation-aware Network for Misinformation Detection.
Hybrid architecture combining RoBERTa text encoding with heterogeneous
Graph Attention Networks for propagation pattern analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

try:
    from torch_geometric.nn import GATConv, SAGEConv, HeteroConv
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not installed. GNN components unavailable.")

try:
    from transformers import AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: Transformers not installed. Text encoder unavailable.")


class TextBranch(nn.Module):
    """Text feature processing branch.

    Takes pre-extracted 797-d text features (or raw text via RoBERTa)
    and projects to 128-d.
    """

    def __init__(self, input_dim: int = 797, hidden_dim: int = 128, freeze_roberta: bool = False):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.projection(text_features)


class StructuralMLP(nn.Module):
    """Processes 65-d structural features into 128-d embeddings."""

    def __init__(self, input_dim: int = 65, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, structural_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(structural_features)


class GraphBranch(nn.Module):
    """Graph neural network branch using GAT + GraphSAGE.

    Processes heterogeneous graph data with multi-head attention.
    Falls back to StructuralMLP if PyG is not available.
    """

    def __init__(self, node_dim: int = 128, hidden_dim: int = 128, heads: int = 8):
        super().__init__()
        self.structural_mlp = StructuralMLP(input_dim=65, hidden_dim=hidden_dim)

        if HAS_PYG:
            # Layer 1: GAT with multi-head attention
            self.conv1 = GATConv(
                in_channels=node_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                dropout=0.2,
                concat=True,
            )
            # Layer 2: GraphSAGE aggregation
            self.conv2 = SAGEConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            )
            self.has_gnn = True
        else:
            self.has_gnn = False

        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        structural_features: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Always process structural features
        struct_out = self.structural_mlp(structural_features)

        # If graph data is available, run GNN layers
        if self.has_gnn and node_features is not None and edge_index is not None:
            x = F.leaky_relu(self.conv1(node_features, edge_index), 0.2)
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            # Global mean pool over nodes
            graph_out = x.mean(dim=0, keepdim=True)
            # Combine with structural features
            return struct_out + graph_out.expand_as(struct_out)

        return struct_out


class FusionLayer(nn.Module):
    """Attention-weighted fusion of text and graph embeddings.

    h_combined = alpha * h_text + (1 - alpha) * h_struct
    where alpha = sigmoid(W * [h_text || h_struct || h_text * h_struct])
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_text: torch.Tensor, h_struct: torch.Tensor) -> torch.Tensor:
        interaction = h_text * h_struct
        combined_input = torch.cat([h_text, h_struct, interaction], dim=-1)
        alpha = self.attention(combined_input)
        return alpha * h_text + (1 - alpha) * h_struct


class ClassifierHead(nn.Module):
    """Final classification: 128 -> 64 -> 2."""

    def __init__(self, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PropNet(nn.Module):
    """
    PropNet: Full hybrid model.

    Architecture:
        Text Features (797-d) -> TextBranch -> 128-d
        Graph Features (65-d) -> GraphBranch -> 128-d
        [128-d text, 128-d graph] -> FusionLayer -> 128-d
        128-d -> ClassifierHead -> [P(real), P(fake)]
    """

    def __init__(
        self,
        text_dim: int = 797,
        structural_dim: int = 65,
        hidden_dim: int = 128,
        num_classes: int = 2,
        gnn_heads: int = 8,
    ):
        super().__init__()
        self.text_branch = TextBranch(input_dim=text_dim, hidden_dim=hidden_dim)
        self.graph_branch = GraphBranch(node_dim=hidden_dim, hidden_dim=hidden_dim, heads=gnn_heads)
        self.fusion = FusionLayer(hidden_dim=hidden_dim)
        self.classifier = ClassifierHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(
        self,
        text_features: torch.Tensor,
        structural_features: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_text = self.text_branch(text_features)
        h_struct = self.graph_branch(structural_features, node_features, edge_index)
        h_fused = self.fusion(h_text, h_struct)
        logits = self.classifier(h_fused)
        return logits

    def predict_proba(
        self,
        text_features: torch.Tensor,
        structural_features: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        logits = self.forward(text_features, structural_features, **kwargs)
        return F.softmax(logits, dim=-1)
