"""
Baseline models for ablation study comparison.

Models:
1. TextOnlyClassifier - RoBERTa text features only
2. NetworkOnlyClassifier - Graph structural features + XGBoost
3. SimpleFusionMLP - Naive concatenation of text + graph features
"""

import numpy as np
from typing import Optional

import torch
import torch.nn as nn


class TextOnlyClassifier(nn.Module):
    """Baseline 1: Text features only (RoBERTa + sentiment + linguistic)."""

    def __init__(self, input_dim: int = 797, num_classes: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.model(text_features)


class SimpleFusionMLP(nn.Module):
    """Baseline 3: Simple concatenation of text + graph features in MLP."""

    def __init__(
        self, text_dim: int = 797, graph_dim: int = 65, num_classes: int = 2
    ):
        super().__init__()
        combined_dim = text_dim + graph_dim
        self.model = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, text_features: torch.Tensor, graph_features: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([text_features, graph_features], dim=-1)
        return self.model(combined)


def get_network_only_classifier():
    """Baseline 2: Graph features + XGBoost (scikit-learn compatible).

    Returns an XGBoost classifier configured for the structural features.
    """
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        print("XGBoost not installed, falling back to Random Forest")
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        )
