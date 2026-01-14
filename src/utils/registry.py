"""Registry mechanism for component management.

This module provides a Registry class that enables dynamic registration and
instantiation of components (models, datasets, losses, metrics) through a
decorator pattern.
"""

from typing import Any, Callable, Dict, Type


class Registry:
    """Registry for managing and building components dynamically.

    Example:
        >>> MODEL_REGISTRY = Registry("model")
        >>> @MODEL_REGISTRY.register("ResNet50")
        ... class ResNet50(nn.Module):
        ...     def __init__(self, num_classes=1000):
        ...         super().__init__()
        >>> model = MODEL_REGISTRY.build({"name": "ResNet50", "params": {"num_classes": 10}})
    """

    def __init__(self, name: str):
        """Initialize a registry.

        Args:
            name: Registry name (e.g., "model", "dataset")
        """
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str) -> Callable:
        """Register a component with a given name.

        Args:
            name: Unique name for the component

        Returns:
            Decorator function

        Raises:
            ValueError: If name already registered
        """

        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self._name} registry"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """Get a registered component by name.

        Args:
            name: Component name

        Returns:
            Registered component class

        Raises:
            KeyError: If name not found
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self._name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def build(self, config: dict) -> Any:
        """Build component from config dict.

        Args:
            config: Config with 'name' and optional 'params' keys
                Example: {"name": "ResNet50", "params": {"num_classes": 10}}

        Returns:
            Instantiated component

        Raises:
            ValueError: If config format invalid
        """
        if "name" not in config:
            raise ValueError("Config must contain 'name' key")

        name = config["name"]
        params = config.get("params", {})
        cls = self.get(name)
        return cls(**params)

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', registered={list(self._registry.keys())})"


# Global registry instances
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
