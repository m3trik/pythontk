# !/usr/bin/python
# coding=utf-8
import copy
from collections import namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from pythontk.core_utils.logging_mixin import LoggingMixin

# Sentinel distinguishing "field missing" from any real value in queries.
_MISSING = object()


class NamedTupleContainer(LoggingMixin):
    """A generic container class for managing collections of named tuples.

    The class provides methods to query, modify, extend, and remove elements
    within the container. It supports dynamic field access, duplicate handling,
    and flexible extension mechanisms.

    Rows may be supplied as named tuples or plain tuples: plain tuples are
    converted to the container's tuple class when the fields are known. A
    container created without fields adopts its schema from the first batch
    of named tuples it receives.

    Attributes:
        named_tuples (List[tuple]): The named tuples stored in this container.
        metadata (Dict[str, Any]): Additional information such as "fields",
            an "allow_duplicates" flag, and an optional "signature_func" used
            for duplicate comparison.
        fields (List[str]): Field names of the contained tuples.
        extender_func (Optional[Callable]): Hook that converts arbitrary
            objects into rows on `extend`.
        _tuple_class: The named tuple class used to build rows from raw data.

    Examples:
        Basic usage:
        ```python
        container = NamedTupleContainer(
            named_tuples=[],
            fields=["name", "age", "email"],
            metadata={"allow_duplicates": False},
        )

        # Add some data (plain tuples are converted automatically)
        container.extend([
            ("Alice", 30, "alice@example.com"),
            ("Bob", 25, "bob@example.com"),
        ])

        # Query data
        adults = container.get(age=30)
        names = container.name  # All names via dynamic attribute access
        email = container.get(name="Bob", return_field="email")  # First match
        ```

        With custom extender function:
        ```python
        def file_extender(container, objects, **metadata):
            # Custom logic to process objects and return list of row tuples
            return [(obj.name, obj.path) for obj in objects]

        container = NamedTupleContainer(
            extender_func=file_extender,
            fields=["filename", "filepath"],
        )
        container.extend(file_objects)
        ```

    Notes:
        `extend` treats a list as a batch and anything else as a single row,
        so pass batches as lists. Logging level can be set at instantiation.
    """

    def __init__(
        self,
        named_tuples: Optional[Sequence[tuple]] = None,
        fields: Union[str, Sequence[str], None] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extender_func: Optional[Callable] = None,
        tuple_class_name: str = "TupleClass",
        log_level: str = "WARNING",
    ) -> None:
        """
        Creates a container for named tuples, providing dynamic attribute access
        and query capabilities.

        Args:
            named_tuples: Rows to initialize the container with. Named tuples
                are stored as-is; plain tuples are converted to the container's
                tuple class when fields are known.
            fields: Field names — a sequence, or a single comma/whitespace-
                delimited string (the same forms `collections.namedtuple`
                accepts). Falls back to `metadata["fields"]`, then to the
                fields of the first named tuple.
            metadata: Metadata related to the container (see class docstring).
            extender_func: Optional function to handle extending the container
                with arbitrary objects.
                Signature: func(container, objects, **metadata) -> List[tuple]
            tuple_class_name: Name for the dynamically created tuple class.
            log_level: Logging level. Defaults to "WARNING".
        """
        self.set_log_level(log_level)

        self.named_tuples = list(named_tuples) if named_tuples else []
        self.metadata = metadata or {}
        self.extender_func = extender_func

        resolved = self._normalize_fields(fields) or self._normalize_fields(
            self.metadata.get("fields")
        )
        if not resolved and self.named_tuples:
            first = self.named_tuples[0]
            if hasattr(first, "_fields"):
                resolved = list(first._fields)
        self.fields = resolved

        self._tuple_class = (
            namedtuple(tuple_class_name, self.fields) if self.fields else None
        )
        if self._tuple_class:
            self.named_tuples = [
                nt if hasattr(nt, "_fields") else self._tuple_class(*nt)
                for nt in self.named_tuples
            ]

    @staticmethod
    def _normalize_fields(fields: Union[str, Sequence[str], None]) -> List[str]:
        """Normalize a fields spec to a list of names.

        Args:
            fields: A sequence of names, or a single comma/whitespace-delimited
                string (mirroring `collections.namedtuple`), or None.

        Returns:
            List of field names (empty if `fields` is falsy).
        """
        if not fields:
            return []
        if isinstance(fields, str):
            return fields.replace(",", " ").split()
        return list(fields)

    def __iter__(self) -> Iterator[tuple]:
        """
        Allows iteration over the named tuples in the container.

        Returns:
            An iterator over the named tuples.
        """
        return iter(self.named_tuples)

    def __repr__(self) -> str:
        """
        Returns the string representation of the container.

        Returns:
            The string representation of the container.
        """
        return (
            f"<{type(self).__name__}"
            f"({len(self.named_tuples)} items, fields={self.fields})>"
        )

    def __len__(self) -> int:
        """
        Returns the number of named tuples in the container.

        Returns:
            The count of named tuples.
        """
        return len(self.named_tuples)

    def __getattr__(self, name: str) -> List[Any]:
        """
        Dynamic attribute access for field names.

        Args:
            name: The attribute name.

        Returns:
            A list of values for the specified field name if found in fields.

        Raises:
            AttributeError: If the attribute is not found in fields.
        """
        # Read via __dict__ so a partially-initialized instance (copy/pickle
        # reconstruction) raises AttributeError instead of recursing on the
        # `self.fields` lookup re-entering __getattr__.
        if name in self.__dict__.get("fields", ()):
            return [getattr(nt, name) for nt in self.named_tuples]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, index: Union[int, slice]) -> Union[tuple, List[tuple]]:
        """
        Get named tuple(s) by index or slice.

        Args:
            index: Integer index or slice object.

        Returns:
            Single named tuple or list of named tuples.
        """
        return self.named_tuples[index]

    def __setitem__(self, index: int, value: tuple) -> None:
        """
        Set a named tuple at a specific index.

        Args:
            index: The index to set.
            value: The named tuple to set.
        """
        self.named_tuples[index] = value

    def __delitem__(self, index: int) -> None:
        """
        Delete a named tuple at a specific index.

        Args:
            index: The index to delete.
        """
        del self.named_tuples[index]

    def _normalize_index(self, index: int) -> int:
        """Resolve a possibly-negative index and bounds-check it.

        Args:
            index: The index to resolve (list semantics; negatives allowed).

        Returns:
            The equivalent non-negative index.

        Raises:
            IndexError: If the index is out of range.
        """
        length = len(self.named_tuples)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError(
                f"Index {index} is out of range for container with {length} items"
            )
        return index

    @staticmethod
    def _default_signature(nt: tuple) -> Any:
        """Build a comparison signature for duplicate detection.

        Values that look like classes/functions (have both `__name__` and
        `__module__`) compare by qualified name rather than identity, so
        re-imported classes still register as duplicates.

        Args:
            nt: The tuple to build a signature for.

        Returns:
            A signature value suitable for equality comparison.
        """
        if not hasattr(nt, "_fields"):
            return nt
        sig = []
        for field_name in nt._fields:
            value = getattr(nt, field_name)
            if hasattr(value, "__name__") and hasattr(value, "__module__"):
                sig.append((field_name, value.__name__, value.__module__))
            else:
                sig.append((field_name, value))
        return tuple(sig)

    @staticmethod
    def _handle_duplicates(
        existing: List[tuple],
        new: List[tuple],
        allow_duplicates: bool,
        signature_func: Optional[Callable] = None,
    ) -> List[tuple]:
        """
        Handles duplicates based on the allow_duplicates flag.

        Args:
            existing: List of existing named tuples.
            new: List of new named tuples to add.
            allow_duplicates: Flag to allow or disallow duplicates.
            signature_func: Optional function to create signatures for comparison.

        Returns:
            Combined list of named tuples with or without duplicates based on
            the flag.
        """
        if allow_duplicates:
            return existing + new

        signature_func = signature_func or NamedTupleContainer._default_signature

        def hashable(sig):
            # Signatures may contain unhashable values (lists, dicts) —
            # fall back to their repr so dedup still works.
            try:
                hash(sig)
            except TypeError:
                return repr(sig)
            return sig

        seen = {hashable(signature_func(nt)) for nt in existing}
        unique_new = []
        for nt in new:
            sig = hashable(signature_func(nt))
            if sig not in seen:
                unique_new.append(nt)
                seen.add(sig)
        return existing + unique_new

    def _coerce_to_existing_class(self, new_named_tuples: List[tuple]) -> List[tuple]:
        """Convert incoming rows to the class of the already-stored tuples.

        Rows carrying the same field names in a different order are realigned
        by name; anything else converts positionally.

        Args:
            new_named_tuples: Rows to coerce.

        Returns:
            The coerced rows (unchanged when no coercion applies).
        """
        if not (self.named_tuples and new_named_tuples):
            return new_named_tuples
        existing_class = type(self.named_tuples[0])
        if not hasattr(existing_class, "_fields"):
            return new_named_tuples

        coerced = []
        for nt in new_named_tuples:
            if type(nt) is existing_class:
                coerced.append(nt)
            elif hasattr(nt, "_asdict") and set(nt._fields) == set(
                existing_class._fields
            ):
                coerced.append(existing_class(**nt._asdict()))
            else:
                coerced.append(existing_class(*nt))
        return coerced

    def extend(
        self, objects: Union[List[tuple], Any], **metadata
    ) -> None:
        """
        Extend the container with new objects while handling duplicates.

        Args:
            objects: Objects to add. A list is treated as a batch:
                - List of named tuples (added directly; a field-less container
                  adopts their fields)
                - List of plain tuples (converted to named tuples)
                - List of other objects (processed by extender_func)
                Anything that is not a list is treated as a single row.
                None and empty lists are no-ops.
            **metadata: Additional metadata merged over the container metadata
                for this call (e.g. allow_duplicates, signature_func).

        Raises:
            ValueError: If no way to process the objects is available.
        """
        if objects is None:
            return
        if not isinstance(objects, list):
            # Single object (bare tuple / named tuple / etc.) — batch of one.
            return self.extend([objects], **metadata)
        if not objects:
            return

        merged_metadata = {**self.metadata, **metadata}
        allow_duplicates = merged_metadata.get("allow_duplicates", False)
        signature_func = merged_metadata.get("signature_func")

        first_obj = objects[0]

        # Case 1: Already named tuples
        if hasattr(first_obj, "_fields"):
            new_named_tuples = list(objects)
            if not self.fields:
                # Adopt the schema from the incoming tuples.
                self.fields = list(first_obj._fields)
                self._tuple_class = type(first_obj)

        # Case 2: Plain tuples — convert to named tuples
        elif isinstance(first_obj, tuple):
            if self._tuple_class is None:
                raise ValueError("No tuple class available to convert tuples")
            new_named_tuples = [self._tuple_class(*obj) for obj in objects]

        # Case 3: Other objects — use the extender function
        else:
            if self.extender_func is None:
                raise ValueError(
                    "No extender function provided for processing objects"
                )
            if self._tuple_class is None:
                raise ValueError(
                    "No tuple class available for extender function results"
                )
            tuple_data = self.extender_func(self, objects, **merged_metadata)
            new_named_tuples = [self._tuple_class(*data) for data in tuple_data]

        new_named_tuples = self._coerce_to_existing_class(new_named_tuples)
        self.named_tuples = self._handle_duplicates(
            self.named_tuples, new_named_tuples, allow_duplicates, signature_func
        )

    def get(
        self, return_field: Optional[str] = None, **conditions
    ) -> Union[List[Any], Any, None]:
        """
        Query the named tuples based on specified conditions.

        Args:
            return_field: The name of the field to return. If None, returns
                entire named tuples.
            **conditions: Key-value pairs representing the query conditions.

        Returns:
            With conditions and a return_field: the first matching value, or
            None if nothing matches. Otherwise a list — of matching named
            tuples, or of the return_field's values.
        """
        single = bool(conditions) and return_field is not None
        results = []
        for named_tuple in self.named_tuples:
            if all(
                getattr(named_tuple, field, _MISSING) == value
                for field, value in conditions.items()
            ):
                result = (
                    getattr(named_tuple, return_field) if return_field else named_tuple
                )
                if single:
                    return result
                results.append(result)
        return None if single else results

    def _clone(self, named_tuples: List[tuple]) -> "NamedTupleContainer":
        """Shallow-copy this container (subclass state included) with new rows.

        Args:
            named_tuples: The rows for the new container.

        Returns:
            A new container of the same (sub)class.
        """
        new = copy.copy(self)
        new.named_tuples = named_tuples
        new.fields = list(self.fields)
        new.metadata = dict(self.metadata)
        return new

    def filter(self, predicate: Callable[[tuple], bool]) -> "NamedTupleContainer":
        """
        Filter the container based on a predicate function.

        Args:
            predicate: Function that takes a named tuple and returns True/False.

        Returns:
            A new container (same subclass) with the matching rows.
        """
        return self._clone([nt for nt in self.named_tuples if predicate(nt)])

    def map(self, func: Callable[[tuple], tuple]) -> "NamedTupleContainer":
        """
        Apply a function to all named tuples in the container.

        Args:
            func: Function that takes a named tuple and returns a modified one.

        Returns:
            A new container (same subclass) with the transformed rows.
        """
        return self._clone([func(nt) for nt in self.named_tuples])

    def modify(self, index: int, **kwargs) -> tuple:
        """
        Modify a named tuple at a specific index within the container.

        Args:
            index: The index of the named tuple to modify (list semantics;
                negatives allowed).
            **kwargs: Key-value pairs representing the fields to update and
                their new values.

        Returns:
            The updated named tuple.

        Raises:
            IndexError: If the index is out of range.
            AttributeError: If trying to modify a field that doesn't exist.
        """
        index = self._normalize_index(index)
        named_tuple = self.named_tuples[index]

        for field in kwargs:
            if field not in getattr(named_tuple, "_fields", ()):
                raise AttributeError(f"Field '{field}' does not exist in named tuple")

        new_tuple = named_tuple._replace(**kwargs)
        self.named_tuples[index] = new_tuple
        return new_tuple

    def remove(self, index: int) -> tuple:
        """
        Remove a named tuple at a specific index within the container.

        Args:
            index: The index of the named tuple to remove (list semantics;
                negatives allowed).

        Returns:
            The removed named tuple.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.named_tuples.pop(self._normalize_index(index))

    def clear(self) -> None:
        """Clear all named tuples from the container."""
        self.named_tuples.clear()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert all named tuples to a list of dictionaries.

        Returns:
            List of dictionaries representing the named tuples.
        """
        return [nt._asdict() for nt in self.named_tuples]

    def to_csv(self, filename: str, **kwargs) -> None:
        """
        Export the container to a CSV file.

        Args:
            filename: Path to the CSV file to write.
            **kwargs: Additional arguments passed to `open()` (e.g. encoding).
        """
        import csv

        kwargs.setdefault("newline", "")
        with open(filename, "w", **kwargs) as csvfile:
            if self.named_tuples:
                writer = csv.DictWriter(csvfile, fieldnames=self.fields)
                writer.writeheader()
                for nt in self.named_tuples:
                    writer.writerow(nt._asdict())

    @classmethod
    def from_csv(
        cls, filename: str, tuple_class_name: str = "CsvRow", **kwargs
    ) -> "NamedTupleContainer":
        """
        Create a NamedTupleContainer from a CSV file.

        All values are read as strings (csv module semantics).

        Args:
            filename: Path to the CSV file to read.
            tuple_class_name: Name for the created tuple class.
            **kwargs: Additional arguments passed to `open()` (e.g. encoding).

        Returns:
            A new NamedTupleContainer with data from the CSV.
        """
        import csv

        kwargs.setdefault("newline", "")
        with open(filename, "r", **kwargs) as csvfile:
            reader = csv.DictReader(csvfile)
            fields = reader.fieldnames

            if not fields:
                return cls(fields=[], tuple_class_name=tuple_class_name)

            TupleClass = namedtuple(tuple_class_name, fields)
            named_tuples = [TupleClass(**row) for row in reader]

            return cls(
                named_tuples=named_tuples,
                fields=fields,
                tuple_class_name=tuple_class_name,
            )


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage demonstrations

    # Basic usage with direct tuple creation
    print("=== Basic Usage ===")
    Person = namedtuple("Person", ["name", "age", "city"])
    people = [
        Person("Alice", 30, "New York"),
        Person("Bob", 25, "London"),
        Person("Charlie", 35, "Tokyo"),
    ]

    container = NamedTupleContainer(named_tuples=people)
    print(f"Container: {container}")
    print(f"Names: {container.name}")
    print(f"Age 35: {container.get(age=35)}")
    print(f"Bob's city: {container.get(name='Bob', return_field='city')}")

    # Usage with extender function
    print("\n=== Custom Extender Function ===")

    def file_data_extender(container, objects, **metadata):
        """Example extender that processes file-like objects"""
        return [(obj, f"/path/to/{obj}", "file") for obj in objects]

    file_container = NamedTupleContainer(
        fields=["name", "path", "type"],
        extender_func=file_data_extender,
        tuple_class_name="FileInfo",
    )

    file_container.extend(["file1.txt", "file2.py", "file3.md"])
    print(f"File container: {file_container}")
    print(f"File paths: {file_container.path}")


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
#
# The container itself is domain-agnostic — the extender function supplies the
# domain-specific logic (see uitk's FileContainer for a subclass that overrides
# `extend` instead). This keeps it reusable beyond file management.
#