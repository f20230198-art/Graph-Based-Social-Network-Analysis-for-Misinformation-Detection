"""
Training loop for PropNet and baseline models.
Handles training, validation, early stopping, and checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)


class Trainer:
    """Training manager for PropNet."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        text_lr: float = 2e-5,
        gnn_lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = "results/checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Differential learning rates
        text_params = list(model.text_branch.parameters())
        other_params = [
            p for name, p in model.named_parameters()
            if "text_branch" not in name
        ]

        self.optimizer = optim.AdamW([
            {"params": text_params, "lr": text_lr, "weight_decay": 0.01},
            {"params": other_params, "lr": gnn_lr, "weight_decay": 0.01},
        ])

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.best_f1 = 0.0
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    def train_epoch(
        self,
        train_text: torch.Tensor,
        train_struct: torch.Tensor,
        train_labels: torch.Tensor,
        batch_size: int = 32,
    ) -> float:
        self.model.train()
        total_loss = 0
        n_batches = 0

        indices = torch.randperm(len(train_labels))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            text_batch = train_text[batch_idx].to(self.device)
            struct_batch = train_struct[batch_idx].to(self.device)
            label_batch = train_labels[batch_idx].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(text_batch, struct_batch)
            loss = self.criterion(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(
        self,
        val_text: torch.Tensor,
        val_struct: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.eval()
        text = val_text.to(self.device)
        struct = val_struct.to(self.device)
        labels = val_labels.numpy()

        logits = self.model(text, struct)
        loss = self.criterion(logits, val_labels.to(self.device)).item()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        metrics = {
            "loss": loss,
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
        }

        if len(np.unique(labels)) == 2:
            metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])

        return metrics

    def fit(
        self,
        train_text: torch.Tensor,
        train_struct: torch.Tensor,
        train_labels: torch.Tensor,
        val_text: torch.Tensor,
        val_struct: torch.Tensor,
        val_labels: torch.Tensor,
        batch_size: int = 32,
    ) -> Dict:
        print(f"Training PropNet for up to {self.max_epochs} epochs...")
        print(f"Train: {len(train_labels)} | Val: {len(val_labels)}")

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(train_text, train_struct, train_labels, batch_size)
            val_metrics = self.evaluate(val_text, val_struct, val_labels)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1_macro"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            self.scheduler.step(val_metrics["f1_macro"])

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Early stopping
            if val_metrics["f1_macro"] > self.best_f1:
                self.best_f1 = val_metrics["f1_macro"]
                self.epochs_no_improve = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        print(f"\nBest validation F1: {self.best_f1:.4f}")
        return self.history

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
        }, path)

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_f1 = checkpoint.get("best_f1", 0)
