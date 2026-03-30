"""
Main entry point for the PropNet Misinformation Detection System.
Orchestrates data loading, feature extraction, training, and evaluation.
"""

import os
import sys
import torch
import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.load_data import load_fakenewsnet, get_cascade_sizes, create_train_test_split
from src.data.preprocess import preprocess_dataset, apply_quality_filters
from src.features.text_features import extract_linguistic_features, batch_extract_linguistic
from src.features.graph_features import extract_cascade_features, build_cascade_graph
from src.models.propnet import PropNet
from src.models.train import Trainer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(os.path.join(PROJECT_ROOT, config_path)) as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("PropNet: Misinformation Detection System")
    print("=" * 60)

    # Load config
    config = load_config()

    # Step 1: Load data
    print("\n[Step 1] Loading FakeNewsNet dataset...")
    data_dir = os.path.join(PROJECT_ROOT, config["data"]["raw_dir"])
    df = load_fakenewsnet(data_dir)
    df["cascade_size"] = get_cascade_sizes(df)

    # Step 2: Preprocess
    print("\n[Step 2] Preprocessing...")
    df = preprocess_dataset(df)
    df = apply_quality_filters(
        df,
        min_cascade_size=config["preprocessing"]["cascade"]["min_size"],
    )

    # Step 3: Extract features
    print("\n[Step 3] Extracting features...")

    # Linguistic features (12-d) as a starting point
    titles = df["title_clean"].fillna("").tolist()
    linguistic_features = batch_extract_linguistic(titles)
    print(f"  Linguistic features shape: {linguistic_features.shape}")

    # Placeholder for full 797-d text features (needs RoBERTa)
    # For now, pad with zeros for the demo
    text_dim = config["model"]["text_dim"]
    text_features = np.zeros((len(df), text_dim), dtype=np.float32)
    text_features[:, -12:] = linguistic_features  # Last 12 dims = linguistic
    print(f"  Text features shape: {text_features.shape}")

    # Structural features (65-d) from cascade sizes
    structural_features = np.zeros((len(df), 65), dtype=np.float32)
    structural_features[:, 0] = np.log1p(df["cascade_size"].values)  # cascade size as proxy
    print(f"  Structural features shape: {structural_features.shape}")

    labels = df["label_binary"].values

    # Step 4: Train/test split
    print("\n[Step 4] Splitting data...")
    train_df, test_df = create_train_test_split(df)
    train_idx = train_df.index.values
    test_idx = test_df.index.values

    train_text = torch.tensor(text_features[train_idx], dtype=torch.float32)
    test_text = torch.tensor(text_features[test_idx], dtype=torch.float32)
    train_struct = torch.tensor(structural_features[train_idx], dtype=torch.float32)
    test_struct = torch.tensor(structural_features[test_idx], dtype=torch.float32)
    train_labels = torch.tensor(labels[train_idx], dtype=torch.long)
    test_labels = torch.tensor(labels[test_idx], dtype=torch.long)

    print(f"  Train: {len(train_labels)} | Test: {len(test_labels)}")

    # Step 5: Build and train model
    print("\n[Step 5] Training PropNet...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Class weights for imbalanced data
    class_counts = np.bincount(labels)
    class_weights = torch.tensor(
        len(labels) / (len(class_counts) * class_counts),
        dtype=torch.float32,
    )
    print(f"  Class weights: {class_weights.tolist()}")

    model = PropNet(
        text_dim=text_dim,
        structural_dim=65,
        hidden_dim=config["model"]["gnn_hidden_dim"],
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        device=device,
        text_lr=config["training"]["text_lr"],
        gnn_lr=config["training"]["gnn_lr"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["early_stopping_patience"],
        class_weights=class_weights,
    )

    history = trainer.fit(
        train_text, train_struct, train_labels,
        test_text, test_struct, test_labels,
        batch_size=config["training"]["batch_size"],
    )

    # Step 6: Final evaluation
    print("\n[Step 6] Final Evaluation...")
    trainer.load_checkpoint("best_model.pt")
    final_metrics = trainer.evaluate(test_text, test_struct, test_labels)
    print("\nTest Results:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    print("\n[Step 7] Saving results...")
    from src.utils.visualization import plot_training_history
    plot_training_history(history, save_path="results/figures/training_history.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
