"""Experiment directory management."""

from pathlib import Path
from datetime import datetime
from typing import Tuple
from .logger import ExpLogger


def create_exp_dir(base_dir: str = "runs", exp_name: str = None) -> str:
    """Create experiment directory with timestamp.

    Args:
        base_dir: Base directory for experiments
        exp_name: Optional experiment name

    Returns:
        Path to created experiment directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Generate experiment name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if exp_name:
        dir_name = f"{exp_name}_{timestamp}"
    else:
        dir_name = f"exp_{timestamp}"

    exp_dir = base_path / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return str(exp_dir)


def init_experiment(config: dict) -> Tuple[str, ExpLogger]:
    """Initialize experiment: create directory and logger.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (exp_dir, logger)
    """
    # Create experiment directory
    base_dir = config.get("experiment", {}).get("base_dir", "runs")
    exp_name = config.get("experiment", {}).get("name", None)

    exp_dir = create_exp_dir(base_dir, exp_name)

    # Create logger
    logger = ExpLogger(exp_dir, config)

    logger.log_text(f"Experiment directory: {exp_dir}")
    logger.log_text(f"Configuration saved to: {exp_dir}/config.yaml")

    return exp_dir, logger
