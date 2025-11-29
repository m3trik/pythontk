# !/usr/bin/python
# coding=utf-8
from typing import Dict, Optional, Any


class SingletonMixin:
    """Reusable singleton mixin that supports optional key-based instances.

    Automatically handles initialization suppression for existing instances.
    """

    _instances: Dict[Any, Any] = {}

    def __init__(self, *args, **kwargs):
        """Prevent object.__init__() from being called with arguments."""
        pass

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        key: Any = kwargs.pop("singleton_key", cls)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            return instance
        return cls._instances[key]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            if not getattr(self, "_initialized", False):
                original_init(self, *args, **kwargs)
                self._initialized = True

        cls.__init__ = new_init

    @classmethod
    def instance(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(*args, **kwargs)

    @classmethod
    def has_instance(cls, singleton_key: Optional[Any] = None) -> bool:
        return (
            singleton_key in cls._instances if singleton_key else cls in cls._instances
        )

    @classmethod
    def reset_instance(cls, singleton_key: Optional[Any] = None) -> None:
        cls._instances.pop(singleton_key or cls, None)


