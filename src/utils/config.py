"""Configuration system for loading, saving, and overriding YAML configs.

This module provides utilities for:
- Loading/saving YAML configuration files
- Command-line override parsing (e.g., "train.epochs=50")
- Config merging with nested dictionaries
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
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
    """Parse string value to appropriate type.

    Args:
        value_str: String value from command line

    Returns:
        Parsed value (int/float/bool/str)

    Examples:
        >>> _parse_value("42")
        42
        >>> _parse_value("3.14")
        3.14
        >>> _parse_value("true")
        True
        >>> _parse_value("hello")
        'hello'
    """
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
    """Apply command-line overrides to config.

    Args:
        config: Base configuration dictionary
        overrides: List of override strings (e.g., ["train.epochs=50"])

    Returns:
        Updated configuration

    Examples:
        >>> config = {"train": {"epochs": 100}}
        >>> apply_overrides(config, ["train.epochs=50"])
        {"train": {"epochs": 50}}
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Use 'key=value'")

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        value = _parse_value(value_str)

        # Navigate to nested dict and set value
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
    """Merge two config dicts recursively.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration (deep copy)

    Examples:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> override = {"b": {"d": 3}}
        >>> merge_configs(base, override)
        {"a": 1, "b": {"c": 2, "d": 3}}
    """
    import copy

    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result
