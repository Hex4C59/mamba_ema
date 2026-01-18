from pathlib import Path
from datetime import datetime
from .logger import ExpLogger


def create_exp_dir(base_dir, exp_name):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dir_name = f"{exp_name}_{timestamp}"
    exp_dir = base_path / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return str(exp_dir)


# 初始化实验结果目录
def init_experiment(config):
    base_dir = config.get("experiment", {}).get("base_dir", "runs")
    exp_name = config.get("experiment", {}).get("name", "exp")
    fold = config.get("data", {}).get("params", {}).get("fold", None)

    if fold is not None:
        exp_name = f"{exp_name}_fold{fold}"
    exp_dir = create_exp_dir(base_dir, exp_name)

    logger = ExpLogger(exp_dir, config)
    logger.log_text(f"Experiment directory: {exp_dir}")
    logger.log_text(f"Configuration saved to: {exp_dir}/config.yaml")

    return exp_dir, logger
