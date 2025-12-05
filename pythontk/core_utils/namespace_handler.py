import weakref
from functools import partial
from typing import Callable, Optional, Any, Type
from pythontk.core_utils.logging_mixin import LoggingMixin


class Placeholder:
    def __init__(
        self,
        class_type: Type,
        factory: Optional[Callable] = None,
        *,
        args: Optional[tuple] = (),
        kwargs: Optional[dict] = None,
        meta: Optional[dict] = None,
    ):
        self.class_type = class_type
        self.factory = factory
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.meta = meta or {}

    def info(self) -> dict:
        return {
            "type": self.class_type.__name__,
            "factory": self.factory.__name__ if self.factory else None,
            "args": self.args,
            "kwargs": self.kwargs,
            "meta": self.meta,
        }

    def create(self, *args, **kwargs):
        ctor = self.factory or self.class_type
        return ctor(*self.args, *args, **{**self.kwargs, **kwargs})

    def __repr__(self):
        return f"<Placeholder for {self.class_type.__name__}>"


class NamespaceHandler(LoggingMixin):
    """A NamespaceHandler that manages its own internal dictionary without attaching
    attributes directly to the owner object.

    Parameters:
        owner (Any): The owner object that the namespace is attached to.
        identifier (str, optional): An identifier for logging or tracking purposes.
        resolver (Callable[[str], Any], optional): A function that resolves attribute names.
        log_level (str, optional): The logging level for the logger.

    Example:
        def resolver(name: str) -> Any:
            # Custom resolver logic here
            return f"Resolved {name}" if name != "unknown_attr" else None

        handler = NamespaceHandler(owner=my_object, identifier="example", resolver=resolver)
        print(handler.some_attr)  # Calls the resolver if not cached
        handler.some_attr = "New Value"  # Sets the attribute directly
    """

    def __init__(
        self,
        owner: Any,
        identifier: str = None,
        resolver: Optional[Callable[[str], Any]] = None,
        use_weakref: bool = False,
        log_level: str = "WARNING",
    ):
        self.logger.setLevel(log_level)

        self.__dict__["Placeholder"] = Placeholder
        self.__dict__["_identifier"] = identifier
        self.__dict__["_resolver"] = resolver
        self.__dict__["_owner"] = owner
        self.__dict__["set"] = self.__setitem__
        self.__dict__["get"] = partial(
            lambda self, k, d=None, resolve_placeholders=True: self.resolve(
                k, d, resolve_placeholders
            ),
            self,
        )
        self.__dict__["raw"] = self.raw
        self.__dict__["_use_weakref"] = use_weakref
        if use_weakref:
            self.__dict__["_attributes"] = weakref.WeakValueDictionary()
        else:
            self.__dict__["_attributes"] = {}
        self.__dict__["_placeholders"] = {}

    @property
    def placeholders(self) -> dict[str, Any]:
        return self._placeholders

    def is_placeholder(self, value: Any) -> bool:
        return isinstance(value, self.Placeholder)

    def get_placeholder(self, key: str) -> Optional[Placeholder]:
        return self._placeholders.get(key)

    def set_placeholder(self, key: str, placeholder: Placeholder):
        self.logger.debug(f"[{self._identifier}] Set placeholder for: {key}")
        self._placeholders[key] = placeholder

    def resolve_all_placeholders(self):
        for key in list(self._placeholders):
            _ = self[key]  # Triggers resolution

    def has_placeholder(self, key: str) -> bool:
        return key in self._placeholders

    def __repr__(self):
        return (
            f"<NamespaceHandler id='{self._identifier}' keys={list(self.keys(True))}>"
        )

    def __contains__(self, key: str) -> bool:
        return key in self._attributes or key in self._placeholders

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

        # Check actual attributes
        if name in self._attributes:
            return self._attributes[name]

        # Check placeholders
        if name in self._placeholders:
            return self._placeholders[name]

        try:
            return self._resolve_and_cache(name)
        except KeyError:
            self.logger.debug(f"[{self._identifier}] Attribute '{name}' not found.")
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            self.__dict__[name] = value
            return

        if isinstance(value, self.Placeholder):
            self._placeholders[name] = value
            self._attributes.pop(name, None)
            return

        self._placeholders.pop(name, None)

        if self._use_weakref:
            try:
                self._attributes[name] = value
            except TypeError:
                self._attributes.data[name] = value
        else:
            self._attributes[name] = value

    def __getitem__(self, key: str, resolve_placeholders: bool = True) -> Any:
        if key in self._attributes:
            return self._attributes[key]

        if key in self._placeholders:
            if not resolve_placeholders:
                return self._placeholders[key]
            return self._resolve_placeholder(key)

        return self._resolve_and_cache(key)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, self.Placeholder):
            self._placeholders[key] = value
            self._attributes.pop(key, None)  # Ensure not duplicated
            return

        self._placeholders.pop(key, None)  # Clean up any previous placeholder

        if self._use_weakref:
            try:
                self._attributes[key] = value
            except TypeError:
                self._attributes.data[key] = value
        else:
            self._attributes[key] = value

    def __delitem__(self, key: str):
        self._attributes.pop(key, None)
        self._placeholders.pop(key, None)

    def keys(self, inc_placeholders=False):
        keys = set(self._attributes.keys())
        if inc_placeholders:
            keys.update(self._placeholders.keys())
        return keys

    def items(self, inc_placeholders=False):
        combined = dict(self._attributes)
        if inc_placeholders:
            combined.update(self._placeholders)
        return combined.items()

    def values(self, inc_placeholders=False):
        values = list(self._attributes.values())
        if inc_placeholders:
            values += list(self._placeholders.values())
        return values

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        self[key] = default
        return default

    def has(self, key: str) -> bool:
        return key in self._attributes

    def raw(self, key: str) -> Optional[Any]:
        if key in self._attributes:
            return self._attributes[key]
        if key in self._placeholders:
            return self._placeholders[key]
        return None

    def resolve(
        self, key: str, default: Any = None, resolve_placeholders: bool = True
    ) -> Any:
        try:
            return self.__getitem__(key, resolve_placeholders=resolve_placeholders)
        except KeyError:
            return default

    def is_resolving(self, key: str) -> bool:
        """Returns True if this key is currently being resolved (to prevent recursion)."""
        return getattr(self, f"__resolving__{key}", False)

    def _resolve_placeholder(self, key: str) -> Any:
        placeholder = self._placeholders[key]
        self.logger.debug(f"[{self._identifier}] Resolving placeholder: {key}")
        value = placeholder.create()
        self[key] = value
        del self._placeholders[key]
        return value

    def _resolve_and_cache(self, key: str) -> Any:
        attributes = self.__dict__["_attributes"]
        if key in attributes:
            return attributes[key]

        # Prevent recursion
        guard_key = f"__resolving__{key}"
        if getattr(self, guard_key, False):
            raise RuntimeError(f"Recursive resolution detected for key: '{key}'")
        setattr(self, guard_key, True)

        try:
            resolver = self.__dict__.get("_resolver")
            if resolver:
                resolved = resolver(key)
                if resolved is not None:
                    if self._use_weakref:
                        try:
                            attributes[key] = resolved
                        except TypeError:
                            attributes.data[key] = resolved
                            self.logger.debug(
                                f"[{self._identifier}] Not weakref-able, stored strong ref for: {key}"
                            )
                    else:
                        attributes[key] = resolved
                    return resolved
        finally:
            if hasattr(self, guard_key):
                delattr(self, guard_key)

        raise KeyError(key)


# --------------------------------------------------------------------------------------------


if __name__ == "__main__":
    ...


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
