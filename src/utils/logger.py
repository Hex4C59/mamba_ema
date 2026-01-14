"""Experiment logging system."""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from .config import save_yaml


class ExpLogger:
    """Experiment logger for tracking metrics and saving results.

    Args:
        exp_dir: Experiment directory
        config: Configuration dict to save
    """

    def __init__(self, exp_dir: str, config: dict):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_yaml(config, self.exp_dir / "config.yaml")

        # Initialize log files
        self.log_file = self.exp_dir / "train.log"

        # Storage for plotting
        self.train_metrics = {"epoch": [], "loss": []}
        self.val_metrics = {"epoch": [], "loss": [], "ccc_v": [], "ccc_a": [], "ccc_avg": []}

    def log(self, metrics: Dict[str, float], step: int, split: str) -> None:
        """Log metrics for a step.

        Args:
            metrics: Dict of metric values
            step: Step number (epoch)
            split: "train" or "val"
        """
        # Store for plotting
        if split == "train":
            self.train_metrics["epoch"].append(step)
            self.train_metrics["loss"].append(metrics.get("loss", 0.0))
        elif split == "val":
            self.val_metrics["epoch"].append(step)
            self.val_metrics["loss"].append(metrics.get("loss", 0.0))
            self.val_metrics["ccc_v"].append(metrics.get("ccc_v", 0.0))
            self.val_metrics["ccc_a"].append(metrics.get("ccc_a", 0.0))
            self.val_metrics["ccc_avg"].append(metrics.get("ccc_avg", 0.0))

    def log_text(self, message: str) -> None:
        """Log text message.

        Args:
            message: Text message
        """
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

        print(message)

    def save_results(self, results: dict, filename: str = "results.txt") -> None:
        """Save results dict as text.

        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = self.exp_dir / filename
        with open(output_path, "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

    def plot_curves(self) -> None:
        """Plot and save training & validation loss curves + CCC metrics."""
        # 1. Loss curve (CCC Loss = 1 - CCC, lower is better)
        fig, ax = plt.subplots(figsize=(10, 6))

        if self.train_metrics["epoch"]:
            ax.plot(
                self.train_metrics["epoch"],
                self.train_metrics["loss"],
                label="Train Loss (1 - CCC)",
                marker='o',
                linewidth=2
            )
        if self.val_metrics["epoch"]:
            ax.plot(
                self.val_metrics["epoch"],
                self.val_metrics["loss"],
                label="Val Loss (1 - CCC)",
                marker='s',
                linewidth=2
            )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("CCC Loss (Lower is Better)", fontsize=12)
        ax.set_title("Training & Validation CCC Loss", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 2])  # CCC Loss range: [0, 2]

        plt.tight_layout()
        plt.savefig(self.exp_dir / "loss_curve.png", dpi=100)
        plt.close()

        # 2. CCC metrics curve (validation only, higher is better)
        if self.val_metrics["epoch"]:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                self.val_metrics["epoch"],
                self.val_metrics["ccc_v"],
                label="CCC-Valence",
                marker='o',
                linewidth=2
            )
            ax.plot(
                self.val_metrics["epoch"],
                self.val_metrics["ccc_a"],
                label="CCC-Arousal",
                marker='s',
                linewidth=2
            )
            ax.plot(
                self.val_metrics["epoch"],
                self.val_metrics["ccc_avg"],
                label="CCC-Average",
                marker='^',
                linewidth=2,
                linestyle='--'
            )

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("CCC Score (Higher is Better)", fontsize=12)
            ax.set_title("Validation CCC Metrics", fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])  # CCC range: [-1, 1], but typically [0, 1] for good models

            plt.tight_layout()
            plt.savefig(self.exp_dir / "ccc_curve.png", dpi=100)
            plt.close()
