# !/usr/bin/python
# coding=utf-8
from collections import namedtuple
from typing import List, Dict, Any, Optional, Union, Iterator, Callable
from pythontk.core_utils.logging_mixin import LoggingMixin


class NamedTupleContainer(LoggingMixin):
    """A generic container class for managing collections of named tuples.

    The class provides methods to query, modify, extend, and remove elements within the container.
    It supports dynamic field access, duplicate handling, and flexible extension mechanisms.

    Attributes:
        named_tuples (List): The list of named tuples stored in this container.
        metadata (Dict[str, Any]): A dictionary containing additional information like "fields"
            which are the named tuple fields, and an "allow_duplicates" flag.
        fields (List[str]): List of named tuple fields derived from metadata.
        _tuple_class: The dynamically generated named tuple class based on `fields`.

    Methods:
        extend: Add new named tuples to the container while handling duplicates.
        get: Query the named tuples based on certain conditions and optionally retrieve a specific field's value.
        modify: Modify a named tuple at a specific index.
        remove: Remove a named tuple at a specific index.

    Examples:
        Basic usage:
        ```python
        container = NamedTupleContainer(
            named_tuples=[],
            fields=["name", "age", "email"],
            metadata={"allow_duplicates": False}
        )

        # Add some data
        Person = namedtuple("Person", ["name", "age", "email"])
        people = [
            Person("Alice", 30, "alice@example.com"),
            Person("Bob", 25, "bob@example.com")
        ]
        container.extend(people)

        # Query data
        adults = container.get(age=30)
        names = container.name  # Get all names via dynamic attribute access
        ```

        With custom extender function:
        ```python
        def file_extender(container, objects, **metadata):
            # Custom logic to process objects and return list of tuples
            return [(obj.name, obj.path) for obj in objects]

        container = NamedTupleContainer(
            extender_func=file_extender,
            fields=["filename", "filepath"]
        )
        container.extend(file_objects)
        ```

    Notes:
        This class utilizes internal logging. The logging level can be set during instantiation.
        The class defines custom `__iter__` and `__repr__` methods to iterate over the named tuples and represent the object.
        Uses the `namedtuple` class from Python's standard library to dynamically create tuple classes.
    """

    def __init__(
        self,
        named_tuples: Optional[List[namedtuple]] = None,
        fields: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extender_func: Optional[Callable] = None,
        tuple_class_name: str = "TupleClass",
        log_level: str = "WARNING",
    ) -> None:
        """
        Creates a container for named tuples, providing dynamic attribute access and query capabilities.

        Args:
            named_tuples: A list of named tuples to initialize the container with.
            fields: List of field names for the named tuples.
            metadata: Metadata related to the container, including field names and other settings.
            extender_func: Optional function to handle extending the container with new objects.
                Should have signature: func(container, objects, **metadata) -> List[tuple]
            tuple_class_name: Name for the dynamically created tuple class.
            log_level: Logging level. Defaults to "WARNING".
        """
        self.logger.setLevel(log_level)

        self.named_tuples = named_tuples or []
        self.metadata = metadata or {}

        # Fields can come from parameter, metadata, or inferred from existing tuples
        if fields:
            self.fields = fields
        elif "fields" in self.metadata:
            from pythontk import make_iterable

            self.fields = make_iterable(self.metadata["fields"])
        elif self.named_tuples:
            # Infer from first tuple
            self.fields = list(self.named_tuples[0]._fields)
        else:
            self.fields = []

        self.extender_func = extender_func
        self._tuple_class = (
            namedtuple(tuple_class_name, self.fields) if self.fields else None
        )

    def __iter__(self) -> Iterator[namedtuple]:
        """
        Allows iteration over the named tuples in the container.

        Returns:
            An iterator over the named tuples.
        """
        return iter(self.named_tuples)

    def __repr__(self) -> str:
        """
        Returns the string representation of the named tuples.

        Returns:
            The string representation of the named tuples.
        """
        return f"<NamedTupleContainer({len(self.named_tuples)} items, fields={self.fields})>"

    def __len__(self) -> int:
        """
        Returns the number of named tuples in the container.

        Returns:
            The count of named tuples.
        """
        return len(self.named_tuples)

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic attribute access for field names.

        Args:
            name: The attribute name.

        Returns:
            A list of values for the specified field name if found in fields.

        Raises:
            AttributeError: If the attribute is not found in fields.
        """
        if name in self.fields:
            return [getattr(nt, name) for nt in self.named_tuples]
        else:
            raise AttributeError(
                f"'NamedTupleContainer' object has no attribute '{name}'"
            )

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[namedtuple, List[namedtuple]]:
        """
        Get named tuple(s) by index or slice.

        Args:
            index: Integer index or slice object.

        Returns:
            Single named tuple or list of named tuples.
        """
        return self.named_tuples[index]

    def __setitem__(self, index: int, value: namedtuple) -> None:
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

    @staticmethod
    def _handle_duplicates(
        existing: List[namedtuple],
        new: List[namedtuple],
        allow_duplicates: bool,
        signature_func: Optional[Callable] = None,
    ) -> List[namedtuple]:
        """
        Handles duplicates based on the allow_duplicates flag.

        Args:
            existing: List of existing named tuples.
            new: List of new named tuples to add.
            allow_duplicates: Flag to allow or disallow duplicates.
            signature_func: Optional function to create signatures for comparison.

        Returns:
            Combined list of named tuples with or without duplicates based on the flag.
        """
        if allow_duplicates:
            return existing + new

        def default_signature(nt):
            """Default signature creation that handles class objects intelligently"""
            if hasattr(nt, "_fields"):
                # Named tuple
                sig = []
                for field_name in nt._fields:
                    value = getattr(nt, field_name)
                    # For class objects, use the class name and module instead of the object itself
                    if hasattr(value, "__name__") and hasattr(value, "__module__"):
                        sig.append((field_name, value.__name__, value.__module__))
                    else:
                        sig.append((field_name, value))
                return tuple(sig)
            else:
                # Regular tuple - just return as is
                return nt

        signature_func = signature_func or default_signature

        # Create signatures for existing tuples
        existing_signatures = {signature_func(nt) for nt in existing}

        # Only add new tuples that don't match existing signatures
        unique_new = []
        for nt in new:
            if signature_func(nt) not in existing_signatures:
                unique_new.append(nt)
                existing_signatures.add(signature_func(nt))

        return existing + unique_new

    def extend(
        self, objects: Union[List[namedtuple], List[tuple], Any], **metadata
    ) -> None:
        """
        Extend the container with new objects while handling duplicates properly.

        Args:
            objects: Objects to add. Can be:
                - List of named tuples (added directly)
                - List of tuples (converted to named tuples)
                - Other objects (processed by extender_func if provided)
            **metadata: Additional metadata to merge with existing metadata.

        Raises:
            ValueError: If no way to process the objects is available.
            Exception: For other errors during extension.
        """
        try:
            # Merge metadata
            merged_metadata = {**self.metadata, **metadata}
            allow_duplicates = merged_metadata.get("allow_duplicates", False)
            signature_func = merged_metadata.get("signature_func")

            new_named_tuples = []

            # Handle different types of input objects
            if isinstance(objects, list) and objects:
                first_obj = objects[0]

                # Case 1: Already named tuples
                if hasattr(first_obj, "_fields"):
                    new_named_tuples = objects

                # Case 2: Regular tuples - convert to named tuples
                elif isinstance(first_obj, tuple):
                    if self._tuple_class:
                        new_named_tuples = [self._tuple_class(*obj) for obj in objects]
                    else:
                        raise ValueError("No tuple class available to convert tuples")

                # Case 3: Other objects - use extender function
                else:
                    if self.extender_func:
                        tuple_data = self.extender_func(
                            self, objects, **merged_metadata
                        )
                        if self._tuple_class:
                            new_named_tuples = [
                                self._tuple_class(*data) for data in tuple_data
                            ]
                        else:
                            raise ValueError(
                                "No tuple class available for extender function results"
                            )
                    else:
                        raise ValueError(
                            "No extender function provided for processing objects"
                        )

            # Handle single object
            elif objects is not None:
                return self.extend([objects], **metadata)

            # Apply duplicate handling and extend the container
            if new_named_tuples:
                # Ensure we use the same tuple class as existing tuples
                if self.named_tuples and new_named_tuples:
                    existing_class = type(self.named_tuples[0])
                    new_class = type(new_named_tuples[0])

                    # Convert to same class if different
                    if existing_class != new_class:
                        new_named_tuples = [
                            existing_class(*nt) for nt in new_named_tuples
                        ]

                self.named_tuples = self._handle_duplicates(
                    self.named_tuples,
                    new_named_tuples,
                    allow_duplicates,
                    signature_func,
                )

        except Exception as e:
            self.logger.error(
                f"An error occurred while extending the container: {str(e)}"
            )
            raise

    def get(
        self, return_field: Optional[str] = None, **conditions
    ) -> Union[List[Any], Any, None]:
        """
        Query the named tuples based on specified conditions.

        Args:
            return_field: The name of the field to return. If None, returns the entire named tuple.
            **conditions: Key-value pairs representing the query conditions.

        Returns:
            A list of matching named tuples or specified field values.
            If conditions and return_field are specified, returns the first matching value or None if not found.
        """
        results = []
        for named_tuple in self.named_tuples:
            if all(
                hasattr(named_tuple, field) and getattr(named_tuple, field) == value
                for field, value in conditions.items()
            ):
                result = (
                    getattr(named_tuple, return_field) if return_field else named_tuple
                )
                # If conditions and return_field are specified, return the first match
                if conditions and return_field:
                    return result
                results.append(result)
        return results

    def filter(self, predicate: Callable[[namedtuple], bool]) -> "NamedTupleContainer":
        """
        Filter the container based on a predicate function.

        Args:
            predicate: Function that takes a named tuple and returns True/False.

        Returns:
            A new NamedTupleContainer with filtered results.
        """
        filtered_tuples = [nt for nt in self.named_tuples if predicate(nt)]
        return NamedTupleContainer(
            named_tuples=filtered_tuples,
            fields=self.fields,
            metadata=self.metadata.copy(),
            extender_func=self.extender_func,
        )

    def map(self, func: Callable[[namedtuple], namedtuple]) -> "NamedTupleContainer":
        """
        Apply a function to all named tuples in the container.

        Args:
            func: Function that takes a named tuple and returns a modified named tuple.

        Returns:
            A new NamedTupleContainer with transformed results.
        """
        mapped_tuples = [func(nt) for nt in self.named_tuples]
        return NamedTupleContainer(
            named_tuples=mapped_tuples,
            fields=self.fields,
            metadata=self.metadata.copy(),
            extender_func=self.extender_func,
        )

    def modify(self, index: int, **kwargs) -> namedtuple:
        """
        Modify a named tuple at a specific index within the container.

        Args:
            index: The index of the named tuple within the container to modify.
            **kwargs: Key-value pairs representing the fields to update and their new values.

        Returns:
            The updated named tuple.

        Raises:
            IndexError: If the index is out of range.
            AttributeError: If trying to modify a field that doesn't exist.
        """
        if not 0 <= index < len(self.named_tuples):
            raise IndexError(
                f"Index {index} is out of range for container with {len(self.named_tuples)} items"
            )

        named_tuple = self.named_tuples[index]

        # Validate that all kwargs fields exist in the named tuple
        for field in kwargs:
            if not hasattr(named_tuple, field):
                raise AttributeError(f"Field '{field}' does not exist in named tuple")

        new_tuple = named_tuple._replace(**kwargs)
        self.named_tuples[index] = new_tuple
        return new_tuple

    def remove(self, index: int) -> namedtuple:
        """
        Remove a named tuple at a specific index within the container.

        Args:
            index: The index of the named tuple within the container to remove.

        Returns:
            The removed named tuple.

        Raises:
            IndexError: If the index is out of range.
        """
        if not 0 <= index < len(self.named_tuples):
            raise IndexError(
                f"Index {index} is out of range for container with {len(self.named_tuples)} items"
            )

        return self.named_tuples.pop(index)

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
            **kwargs: Additional arguments passed to csv.writer.
        """
        import csv

        with open(filename, "w", newline="", **kwargs) as csvfile:
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

        Args:
            filename: Path to the CSV file to read.
            tuple_class_name: Name for the created tuple class.
            **kwargs: Additional arguments passed to csv.reader.

        Returns:
            A new NamedTupleContainer with data from the CSV.
        """
        import csv

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
    print(f"Adults over 30: {container.get(age=35)}")

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
# This standalone NamedTupleContainer provides:
# 1. Generic tuple management capabilities
# 2. Flexible extension mechanisms via custom extender functions
# 3. Advanced querying and filtering
# 4. Export/import capabilities (CSV)
# 5. Functional programming style methods (map, filter)
# 6. Full compatibility with the original FileManager use case
#
# The key insight is that the container itself is domain-agnostic -
# it's the extender function that provides domain-specific logic.
# This makes it reusable across many different use cases beyond file management.
#
