# !/usr/bin/python
# coding=utf-8
"""Map-conversion registry primitives for the texture MapFactory.

``MapConversion`` describes one source->target conversion; ``ConversionRegistry``
collects them and resolves the best converter for a target map type. Split out
of the monolithic ``map_factory`` module so the registry plumbing stays free of
the factory orchestrator and its workflow handlers.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Union
from collections import defaultdict
import inspect


@dataclass
class MapConversion:
    """Defines a single map conversion operation."""

    target_type: str
    source_types: List[str]
    converter: Callable
    priority: int = 0


class ConversionRegistry:
    """Central registry for all map type conversions.

    This eliminates duplicate conversion logic across methods.
    """

    def __init__(self):
        self._conversions: Dict[str, List[MapConversion]] = defaultdict(list)
        self._registered_classes = set()
        self._pending_plugins = set()

    def add_plugin(self, cls):
        """Register a class to be scanned for conversions later."""
        self._pending_plugins.add(cls)

    def _scan_pending(self):
        """Scan any pending plugins."""
        while self._pending_plugins:
            cls = self._pending_plugins.pop()
            if hasattr(cls, "register_conversions"):
                cls.register_conversions(self)
            else:
                # Fallback for backward compatibility or mixed usage
                self.register_from_class(cls)

    def register(
        self,
        target_type: Union[str, MapConversion],
        source_types: Union[str, List[str]] = None,
        converter: Callable = None,
        priority: int = 0,
    ):
        """Register a new conversion strategy.

        Can be called with a MapConversion object or with individual arguments.
        """
        if isinstance(target_type, MapConversion):
            conversion = target_type
        else:
            if source_types is None or converter is None:
                raise ValueError(
                    "source_types and converter are required when registering by arguments"
                )

            if isinstance(source_types, str):
                source_types = [source_types]

            conversion = MapConversion(
                target_type=target_type,
                source_types=source_types,
                converter=converter,
                priority=priority,
            )

        # Prevent duplicate registrations
        current_list = self._conversions[conversion.target_type]
        for existing in current_list:
            if (
                existing.converter == conversion.converter
                and existing.source_types == conversion.source_types
            ):
                return

        self._conversions[conversion.target_type].append(conversion)
        # Sort by priority (higher first)
        self._conversions[conversion.target_type].sort(
            key=lambda c: c.priority, reverse=True
        )

    def register_from_class(self, cls):
        """Register all decorated conversion methods from a class."""
        if cls in self._registered_classes:
            return

        for name, method in inspect.getmembers(cls):
            if hasattr(method, "_conversion_info"):
                infos = method._conversion_info
                if isinstance(infos, dict):
                    infos = [infos]
                for info in infos:
                    self.register(
                        target_type=info["target_type"],
                        source_types=info["source_types"],
                        converter=method,
                        priority=info["priority"],
                    )

        self._registered_classes.add(cls)

    def get_conversions_for(self, target_type: str) -> List[MapConversion]:
        """Get all conversions that can produce target type."""
        self._scan_pending()
        return self._conversions.get(target_type, [])

    def __getattr__(self, name):
        """Allow property-style access to conversions (e.g. registry.Metallic).

        Underscore-prefixed names raise AttributeError so protocol probes
        (copy/pickle dunders, private-attr lookups) don't silently receive an
        empty conversion list.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return self.get_conversions_for(name)
