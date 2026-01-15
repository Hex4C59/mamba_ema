from pathlib import Path
from typing import Any, Dict, List, Union
import yaml
import copy

def load_yaml(path: Union[str, Path]) -> dict:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def save_yaml(data: dict, path: Union[str, Path]) -> None:
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _parse_value(value_str: str) -> Any:
    # Try bool
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def apply_overrides(config: dict, overrides: List[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Use 'key=value'")

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        value = _parse_value(value_str)

        current = config
        for i, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise ValueError(f"Cannot override non-dict value at '{'.'.join(keys[:i+1])}'")
            current = current[key]

        current[keys[-1]] = value

    return config


def merge_configs(base: dict, override: dict) -> dict:
    import copy

    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result

    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result
