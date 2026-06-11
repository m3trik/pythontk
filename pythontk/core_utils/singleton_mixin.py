# !/usr/bin/python
# coding=utf-8
from typing import Any, Dict, Optional


class SingletonMixin:
    """Reusable singleton mixin that supports optional key-based instances.

    Instances are stored per ``(class, singleton_key)`` pair, so two different
    subclasses using the same ``singleton_key`` never collide. The
    ``singleton_key`` kwarg is consumed before reaching the subclass
    ``__init__``, and re-initialization of an existing instance is suppressed.
    """

    _instances: Dict[Any, Any] = {}

    def __init__(self, *args, **kwargs):
        """Prevent object.__init__() from being called with arguments."""
        pass

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        key = (cls, kwargs.pop("singleton_key", None))
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # ``singleton_key`` is routing data for __new__, not an init arg.
            kwargs.pop("singleton_key", None)
            if not getattr(self, "_initialized", False):
                original_init(self, *args, **kwargs)
                self._initialized = True

        cls.__init__ = new_init

    @classmethod
    def instance(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(*args, **kwargs)

    @classmethod
    def has_instance(cls, singleton_key: Optional[Any] = None) -> bool:
        return (cls, singleton_key) in cls._instances

    @classmethod
    def reset_instance(cls, singleton_key: Optional[Any] = None) -> None:
        cls._instances.pop((cls, singleton_key), None)
