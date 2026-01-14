"""Experiment logging system."""

from pathlib import Path
from typing import Dict
import csv
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
        self.metrics_file = self.exp_dir / "metrics.csv"

        # Initialize metrics CSV
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "split", "loss", "ccc_v", "ccc_a", "ccc_avg"])

        # Storage for plotting
        self.train_metrics = {"epoch": [], "loss": []}
        self.val_metrics = {"epoch": [], "ccc_v": [], "ccc_a": [], "ccc_avg": []}

    def log(self, metrics: Dict[str, float], step: int, split: str) -> None:
        """Log metrics for a step.

        Args:
            metrics: Dict of metric values
            step: Step number (epoch)
            split: "train" or "val"
        """
        # Write to CSV
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = [
                step,
                split,
                metrics.get("loss", 0.0),
                metrics.get("ccc_v", 0.0),
                metrics.get("ccc_a", 0.0),
                metrics.get("ccc_avg", 0.0),
            ]
            writer.writerow(row)

        # Store for plotting
        if split == "train":
            self.train_metrics["epoch"].append(step)
            self.train_metrics["loss"].append(metrics.get("loss", 0.0))
        elif split == "val":
            self.val_metrics["epoch"].append(step)
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
        """Plot training curves."""
        # Loss curve
        if self.train_metrics["epoch"]:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.train_metrics["epoch"],
                self.train_metrics["loss"],
                label="Train Loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.exp_dir / "loss_curve.png")
            plt.close()

        # Metrics curve
        if self.val_metrics["epoch"]:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.val_metrics["epoch"], self.val_metrics["ccc_v"], label="CCC-V"
            )
            plt.plot(
                self.val_metrics["epoch"], self.val_metrics["ccc_a"], label="CCC-A"
            )
            plt.plot(
                self.val_metrics["epoch"],
                self.val_metrics["ccc_avg"],
                label="CCC-Avg",
                linewidth=2,
            )
            plt.xlabel("Epoch")
            plt.ylabel("CCC")
            plt.title("Validation CCC")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.exp_dir / "metrics_curve.png")
            plt.close()
