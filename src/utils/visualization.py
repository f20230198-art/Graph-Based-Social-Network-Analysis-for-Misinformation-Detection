"""
Visualization utilities for results and analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot training loss and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    # F1
    axes[1].plot(history["val_f1"], label="Val F1 (macro)", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Validation F1 Score")
    axes[1].legend()

    # Accuracy
    axes[2].plot(history["val_acc"], label="Val Accuracy", color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Validation Accuracy")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["accuracy", "f1_macro", "roc_auc"],
    save_path: Optional[str] = None,
):
    """Plot comparison bar chart across ablation models."""
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(models):
        values = [results[model_name].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Ablation Study)")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cascade_comparison(
    fake_cascades: List[Dict],
    real_cascades: List[Dict],
    save_path: Optional[str] = None,
):
    """Compare structural features between fake and real cascades."""
    features = ["max_depth", "cascade_size", "velocity_early", "structural_virality"]

    fig, axes = plt.subplots(1, len(features), figsize=(16, 4))
    for i, feat in enumerate(features):
        fake_vals = [c.get(feat, 0) for c in fake_cascades]
        real_vals = [c.get(feat, 0) for c in real_cascades]
        axes[i].boxplot([fake_vals, real_vals], labels=["Fake", "Real"])
        axes[i].set_title(feat)

    plt.suptitle("Cascade Feature Comparison: Fake vs Real News")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
