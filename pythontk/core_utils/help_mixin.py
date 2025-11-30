# !/usr/bin/python
# coding=utf-8
"""HelpMixin - Enhanced help system leveraging Python's built-in help infrastructure.

This module provides a mixin class that enhances class introspection by wrapping
and extending Python's built-in help() functionality with filtering, sorting,
and targeted output options.
"""
import inspect
import pydoc
from typing import TYPE_CHECKING, Any, List, Optional


class HelpMixin:
    """A mixin providing enhanced help() functionality with filtering and sorting.

    This mixin wraps Python's built-in help system while adding:
    - Targeted help for specific methods, properties, or attributes
    - Filtering by member type (methods, properties, classmethods, staticmethods)
    - Alphabetical or categorical sorting
    - Option to include/exclude inherited members
    - Brief summaries or full documentation
    - Source code viewing
    - Source location information (file:line)
    - Inheritance chain display
    - Async/generator/abstract detection

    Usage:
        class MyClass(HelpMixin):
            def my_method(self):
                '''My method docstring.'''
                pass

        # Full help (uses built-in help)
        MyClass.help()

        # Brief summary of all methods
        MyClass.help(brief=True)

        # Help for specific method
        MyClass.help("my_method")

        # Only public methods, sorted alphabetically
        MyClass.help(members="methods", sort=True)

        # Exclude inherited members
        MyClass.help(inherited=False)

        # View source code
        MyClass.source("my_method")

        # Find where method is defined
        MyClass.where("my_method")

        # Show inheritance chain
        MyClass.mro()
    """

    # Lazy-loaded module references (class-level cache)
    _inspect = None
    _pydoc = None

    @classmethod
    def _get_inspect(cls):
        """Lazily import and cache the inspect module."""
        if cls._inspect is None:
            import inspect

            HelpMixin._inspect = inspect
        return cls._inspect

    @classmethod
    def _get_pydoc(cls):
        """Lazily import and cache the pydoc module."""
        if cls._pydoc is None:
            import pydoc

            HelpMixin._pydoc = pydoc
        return cls._pydoc

    @classmethod
    def help(
        cls,
        name: Optional[str] = None,
        *,
        members: Optional[str] = None,
        inherited: bool = True,
        brief: bool = False,
        sort: bool = False,
        private: bool = False,
        returns: bool = False,
    ) -> Optional[str]:
        """Display or return help information for this class or a specific member.

        Parameters:
            name: Name of a specific method/property to get help for.
                  If None, shows help for the entire class.
            members: Filter by member type. One of:
                     "methods", "properties", "classmethods", "staticmethods", "all"
                     If None, shows all members (same as "all").
            inherited: If True (default), include inherited members.
                       If False, only show members defined in this class.
            brief: If True, show one-line summaries instead of full docstrings.
            sort: If True, sort members alphabetically.
                  If False (default), preserve definition order.
            private: If True, include private members (those starting with _).
                     If False (default), exclude private members.
            returns: If True, return the help string instead of printing.
                     If False (default), print to stdout.

        Returns:
            Help string if returns=True, otherwise None.

        Examples:
            >>> MyClass.help()                          # Full built-in help
            >>> MyClass.help("method_name")             # Help for specific method
            >>> MyClass.help(brief=True)                # Brief summaries
            >>> MyClass.help(members="methods")         # Only methods
            >>> MyClass.help(inherited=False, sort=True) # Own members, sorted
        """
        # Specific member requested - use built-in help or return targeted info
        if name is not None:
            return cls._help_for_member(name, brief=brief, returns=returns)

        # Full help requested - either built-in or filtered
        if members is None and inherited and not brief and not sort and not private:
            # Default case: just use built-in help
            if returns:
                return pydoc.render_doc(cls, title="%s")
            help(cls)
            return None

        # Filtered/customized help
        return cls._filtered_help(
            members=members,
            inherited=inherited,
            brief=brief,
            sort=sort,
            private=private,
            returns=returns,
        )

    @classmethod
    def _help_for_member(
        cls,
        name: str,
        brief: bool = False,
        returns: bool = False,
    ) -> Optional[str]:
        """Get help for a specific member by name."""
        member = getattr(cls, name, None)
        if member is None:
            msg = f"'{cls.__name__}' has no member '{name}'"
            if returns:
                return msg
            print(msg)
            return None

        if brief:
            output = cls._format_member_brief(name, member)
        else:
            # Use built-in help for full documentation
            if returns:
                return pydoc.render_doc(member, title="%s")
            help(member)
            return None

        if returns:
            return output
        print(output)
        return None

    @classmethod
    def _filtered_help(
        cls,
        members: Optional[str],
        inherited: bool,
        brief: bool,
        sort: bool,
        private: bool,
        returns: bool,
    ) -> Optional[str]:
        """Generate filtered help output."""
        lines = [
            f"class {cls.__name__}({', '.join(b.__name__ for b in cls.__bases__)})",
            "",
        ]

        # Get docstring
        if cls.__doc__:
            doc_summary = cls._get_summary(cls.__doc__)
            lines.append(f"    {doc_summary}")
            lines.append("")

        # Collect members
        collected = cls._collect_members(
            members=members,
            inherited=inherited,
            private=private,
        )

        if sort:
            collected = sorted(collected, key=lambda x: x[0])

        # Format output
        if collected:
            lines.append("Members:")
            for name, member, member_type in collected:
                if brief:
                    lines.append(cls._format_member_brief(name, member, member_type))
                else:
                    lines.append(cls._format_member_full(name, member, member_type))

        output = "\n".join(lines)
        if returns:
            return output
        print(output)
        return None

    @classmethod
    def _collect_members(
        cls,
        members: Optional[str],
        inherited: bool,
        private: bool,
    ) -> List[tuple]:
        """Collect class members based on filters.

        Returns:
            List of (name, member, member_type) tuples.
        """
        result = []
        own_attrs = set(cls.__dict__.keys()) if not inherited else None

        for name in dir(cls):
            # Skip private unless requested
            if not private and name.startswith("_"):
                continue

            # Skip if not inherited and not defined in this class
            if own_attrs is not None and name not in own_attrs:
                continue

            try:
                member = getattr(cls, name)
            except AttributeError:
                continue

            member_type = cls._get_member_type(cls, name, member)

            # Filter by member type
            if members is not None and members != "all":
                if members == "methods" and member_type not in (
                    "method",
                    "classmethod",
                    "staticmethod",
                ):
                    continue
                elif members == "properties" and member_type != "property":
                    continue
                elif members == "classmethods" and member_type != "classmethod":
                    continue
                elif members == "staticmethods" and member_type != "staticmethod":
                    continue

            result.append((name, member, member_type))

        return result

    @classmethod
    def _get_member_type(cls, klass: type, name: str, member: Any) -> str:
        """Determine the type of a class member."""
        # Check the class dict for the raw descriptor
        raw = klass.__dict__.get(name)

        if isinstance(raw, classmethod):
            return "classmethod"
        elif isinstance(raw, staticmethod):
            return "staticmethod"
        elif isinstance(raw, property):
            return "property"
        elif callable(member):
            return "method"
        else:
            return "attribute"

    @classmethod
    def _format_member_brief(
        cls,
        name: str,
        member: Any,
        member_type: Optional[str] = None,
    ) -> str:
        """Format a single member as a brief one-liner."""
        if member_type is None:
            member_type = cls._get_member_type(cls, name, member)

        # Get signature if callable
        sig = ""
        if callable(member) and member_type != "property":
            try:
                # Unwrap to get true signature
                unwrapped = inspect.unwrap(member)
                sig = str(inspect.signature(unwrapped))
            except (ValueError, TypeError):
                sig = "(...)"

        # Get first line of docstring using inspect.getdoc for cleaner output
        doc = ""
        doc_str = inspect.getdoc(member)
        if doc_str:
            doc = cls._get_summary(doc_str)

        # Build type label with flags
        flags = cls._get_member_flags(member)
        type_parts = [member_type] if member_type else []
        if flags:
            type_parts.append(flags)
        type_label = f"[{', '.join(type_parts)}]" if type_parts else ""

        return (
            f"    {name}{sig}  {type_label}\n        {doc}"
            if doc
            else f"    {name}{sig}  {type_label}"
        )

    @classmethod
    def _format_member_full(
        cls,
        name: str,
        member: Any,
        member_type: Optional[str] = None,
    ) -> str:
        """Format a single member with full documentation."""
        if member_type is None:
            member_type = cls._get_member_type(cls, name, member)

        lines = []

        # Header with signature
        sig = ""
        if callable(member) and member_type != "property":
            try:
                # Unwrap to get true signature
                unwrapped = inspect.unwrap(member)
                sig = str(inspect.signature(unwrapped))
            except (ValueError, TypeError):
                sig = "(...)"

        # Build type label with flags
        flags = cls._get_member_flags(member)
        type_parts = [member_type] if member_type else []
        if flags:
            type_parts.append(flags)
        type_label = f"[{', '.join(type_parts)}]" if type_parts else ""

        lines.append(f"\n    {name}{sig}  {type_label}")

        # Full docstring using inspect.getdoc for cleaner output
        doc_str = inspect.getdoc(member)
        if doc_str:
            doc_lines = doc_str.split("\n")
            for line in doc_lines:
                lines.append(f"        {line}")

        return "\n".join(lines)

    @staticmethod
    def _get_summary(docstring: str) -> str:
        """Extract the first line/sentence from a docstring."""
        if not docstring:
            return ""
        # Clean and get first non-empty line
        lines = inspect.cleandoc(docstring).split("\n")
        for line in lines:
            line = line.strip()
            if line:
                # Truncate at first period if reasonable
                if ". " in line:
                    return line.split(". ")[0] + "."
                return line
        return ""

    # -------------------------------------------------------------------------
    # Source Code Methods
    # -------------------------------------------------------------------------

    @classmethod
    def source(
        cls,
        name: Optional[str] = None,
        *,
        returns: bool = False,
    ) -> Optional[str]:
        """Get source code for the class or a specific member.

        Parameters:
            name: Name of a specific method/property to get source for.
                  If None, shows source for the entire class.
            returns: If True, return the source string instead of printing.

        Returns:
            Source code string if returns=True, otherwise None.

        Examples:
            >>> MyClass.source()              # Full class source
            >>> MyClass.source("my_method")   # Method source only
            >>> src = MyClass.source("my_method", returns=True)
        """
        target = cls if name is None else getattr(cls, name, None)

        if target is None:
            msg = f"'{cls.__name__}' has no member '{name}'"
            if returns:
                return msg
            print(msg)
            return None

        try:
            # Unwrap decorated functions to get actual source
            unwrapped = inspect.unwrap(target) if callable(target) else target
            source = inspect.getsource(unwrapped)
        except (OSError, TypeError) as e:
            msg = f"Cannot retrieve source: {e}"
            if returns:
                return msg
            print(msg)
            return None

        if returns:
            return source
        print(source)
        return None

    @classmethod
    def where(
        cls,
        name: Optional[str] = None,
        *,
        returns: bool = False,
    ) -> Optional[str]:
        """Get the file and line number where the class or member is defined.

        Parameters:
            name: Name of a specific method/property to locate.
                  If None, shows location of the class.
            returns: If True, return the location string instead of printing.

        Returns:
            Location string (file:line) if returns=True, otherwise None.

        Examples:
            >>> MyClass.where()              # Class location
            >>> MyClass.where("my_method")   # Method location
        """
        target = cls if name is None else getattr(cls, name, None)

        if target is None:
            msg = f"'{cls.__name__}' has no member '{name}'"
            if returns:
                return msg
            print(msg)
            return None

        try:
            # Unwrap to get actual location for decorated functions
            unwrapped = inspect.unwrap(target) if callable(target) else target
            source_file = inspect.getsourcefile(unwrapped)
            _, line_no = inspect.getsourcelines(unwrapped)
            location = f"{source_file}:{line_no}" if source_file else "Built-in"
        except (OSError, TypeError):
            # Fall back to just file if lines unavailable
            try:
                source_file = inspect.getfile(target)
                location = source_file
            except (OSError, TypeError):
                location = "Unknown location (built-in or C extension)"

        if returns:
            return location
        print(location)
        return None

    @classmethod
    def mro(
        cls,
        *,
        brief: bool = False,
        returns: bool = False,
    ) -> Optional[str]:
        """Show the method resolution order (inheritance chain) for this class.

        Parameters:
            brief: If True, show only class names.
                   If False (default), include module paths.
            returns: If True, return the MRO string instead of printing.

        Returns:
            MRO string if returns=True, otherwise None.

        Examples:
            >>> MyClass.mro()                # Full MRO with modules
            >>> MyClass.mro(brief=True)      # Just class names
        """
        mro_classes = inspect.getmro(cls)
        lines = [f"Method Resolution Order for {cls.__name__}:", ""]

        for i, klass in enumerate(mro_classes):
            indent = "  " * i
            if brief:
                lines.append(f"{indent}└── {klass.__name__}")
            else:
                module = klass.__module__
                lines.append(f"{indent}└── {module}.{klass.__name__}")

        output = "\n".join(lines)
        if returns:
            return output
        print(output)
        return None

    # -------------------------------------------------------------------------
    # Signature Inspection
    # -------------------------------------------------------------------------

    @classmethod
    def signature(
        cls,
        name: str,
        *,
        returns: bool = False,
    ) -> Optional[str]:
        """Get detailed signature information for a method.

        Shows parameter names, types, defaults, and return type if annotated.

        Parameters:
            name: Name of the method to inspect.
            returns: If True, return the signature info instead of printing.

        Returns:
            Signature info string if returns=True, otherwise None.

        Examples:
            >>> MyClass.signature("my_method")
        """
        member = getattr(cls, name, None)
        if member is None:
            msg = f"'{cls.__name__}' has no member '{name}'"
            if returns:
                return msg
            print(msg)
            return None

        if not callable(member):
            msg = f"'{name}' is not callable"
            if returns:
                return msg
            print(msg)
            return None

        try:
            # Unwrap to get true signature
            unwrapped = inspect.unwrap(member)
            sig = inspect.signature(unwrapped)
        except (ValueError, TypeError) as e:
            msg = f"Cannot retrieve signature: {e}"
            if returns:
                return msg
            print(msg)
            return None

        lines = [f"{name}{sig}", ""]
        lines.append("Parameters:")

        for param_name, param in sig.parameters.items():
            parts = [f"    {param_name}"]

            # Type annotation
            if param.annotation is not inspect.Parameter.empty:
                parts.append(f": {cls._format_annotation(param.annotation)}")

            # Default value
            if param.default is not inspect.Parameter.empty:
                parts.append(f" = {param.default!r}")

            # Parameter kind
            kind_map = {
                inspect.Parameter.POSITIONAL_ONLY: " (positional-only)",
                inspect.Parameter.VAR_POSITIONAL: " (*args)",
                inspect.Parameter.VAR_KEYWORD: " (**kwargs)",
                inspect.Parameter.KEYWORD_ONLY: " (keyword-only)",
            }
            if param.kind in kind_map:
                parts.append(kind_map[param.kind])

            lines.append("".join(parts))

        # Return type
        if sig.return_annotation is not inspect.Signature.empty:
            lines.append("")
            lines.append(
                f"Returns: {cls._format_annotation(sig.return_annotation)}"
            )

        output = "\n".join(lines)
        if returns:
            return output
        print(output)
        return None

    @staticmethod
    def _format_annotation(annotation: Any) -> str:
        """Format a type annotation for display."""
        # For typing module generic types (List[str], Optional[int], etc.)
        # use string representation to show the full type
        if hasattr(annotation, "__origin__"):
            repr_str = str(annotation)
            # Remove 'typing.' prefix for cleaner output
            if repr_str.startswith("typing."):
                repr_str = repr_str[7:]
            return repr_str

        # Simple types with __name__ (str, int, custom classes)
        if hasattr(annotation, "__name__"):
            return annotation.__name__

        # Fallback to string representation
        return str(annotation)

    # -------------------------------------------------------------------------
    # Member Classification
    # -------------------------------------------------------------------------

    @classmethod
    def classify(
        cls,
        name: Optional[str] = None,
        *,
        returns: bool = False,
    ) -> Optional[str]:
        """Classify a member or list all members with their classifications.

        Shows detailed information including:
        - Member type (method, property, classmethod, staticmethod, attribute)
        - Defining class (which class in the MRO defines this member)
        - Special flags (async, generator, abstract)

        Parameters:
            name: Name of a specific member to classify.
                  If None, classify all members.
            returns: If True, return the classification instead of printing.

        Returns:
            Classification string if returns=True, otherwise None.

        Examples:
            >>> MyClass.classify("my_method")   # Single member
            >>> MyClass.classify()              # All members
        """
        if name is not None:
            # Classify single member
            member = getattr(cls, name, None)
            if member is None:
                msg = f"'{cls.__name__}' has no member '{name}'"
                if returns:
                    return msg
                print(msg)
                return None

            info = cls._classify_member(name, member)
            if returns:
                return info
            print(info)
            return None

        # Classify all members using inspect.classify_class_attrs
        attrs = inspect.classify_class_attrs(cls)
        lines = [f"Classification of {cls.__name__} members:", ""]

        # Group by kind
        by_kind: dict = {}
        for attr in attrs:
            if attr.name.startswith("_") and not attr.name.startswith("__"):
                continue  # Skip private
            kind = attr.kind
            if kind not in by_kind:
                by_kind[kind] = []
            by_kind[kind].append(attr)

        for kind in sorted(by_kind.keys()):
            lines.append(f"{kind.upper()}:")
            for attr in sorted(by_kind[kind], key=lambda a: a.name):
                flags = cls._get_member_flags(attr.object)
                flag_str = f" {flags}" if flags else ""
                lines.append(
                    f"    {attr.name}{flag_str} (from {attr.defining_class.__name__})"
                )
            lines.append("")

        output = "\n".join(lines)
        if returns:
            return output
        print(output)
        return None

    @classmethod
    def _classify_member(cls, name: str, member: Any) -> str:
        """Get detailed classification for a single member."""
        lines = [f"Member: {name}", ""]

        # Find defining class
        defining_class = None
        for klass in inspect.getmro(cls):
            if name in klass.__dict__:
                defining_class = klass
                break

        member_type = cls._get_member_type(cls, name, member)
        lines.append(f"Type: {member_type}")

        if defining_class:
            lines.append(f"Defined in: {defining_class.__name__}")

        # Flags
        flags = cls._get_member_flags(member)
        if flags:
            lines.append(f"Flags: {flags}")

        # Signature if callable
        if callable(member) and member_type != "property":
            try:
                sig = inspect.signature(member)
                lines.append(f"Signature: {name}{sig}")
            except (ValueError, TypeError):
                pass

        # Docstring summary
        doc = inspect.getdoc(member)
        if doc:
            lines.append(f"Summary: {cls._get_summary(doc)}")

        return "\n".join(lines)

    @staticmethod
    def _get_member_flags(member: Any) -> str:
        """Get flags describing special member characteristics."""
        flags = []

        # Unwrap to check underlying function
        try:
            unwrapped = inspect.unwrap(member) if callable(member) else member
        except Exception:
            unwrapped = member

        if inspect.iscoroutinefunction(unwrapped):
            flags.append("async")
        if inspect.isgeneratorfunction(unwrapped):
            flags.append("generator")
        if inspect.isasyncgenfunction(unwrapped):
            flags.append("async-generator")
        if getattr(unwrapped, "__isabstractmethod__", False):
            flags.append("abstract")

        return ", ".join(flags) if flags else ""

    # -------------------------------------------------------------------------
    # Listing Methods
    # -------------------------------------------------------------------------

    @classmethod
    def list_members(
        cls,
        members: Optional[str] = None,
        *,
        inherited: bool = True,
        private: bool = False,
        sort: bool = True,
        returns: bool = False,
    ) -> Optional[List[str]]:
        """Get a list of member names.

        Parameters:
            members: Filter by member type. One of:
                     "methods", "properties", "classmethods", "staticmethods", "all"
            inherited: If True (default), include inherited members.
            private: If True, include private members.
            sort: If True (default), sort alphabetically.
            returns: If True, return the list instead of printing.

        Returns:
            List of member names if returns=True, otherwise None.

        Examples:
            >>> MyClass.list_members()                    # All public members
            >>> MyClass.list_members("methods")           # Only methods
            >>> names = MyClass.list_members(returns=True)
        """
        collected = cls._collect_members(
            members=members,
            inherited=inherited,
            private=private,
        )

        names = [name for name, _, _ in collected]
        if sort:
            names.sort()

        if returns:
            return names
        print("\n".join(names) if names else "No members found")
        return None

    @staticmethod
    def about(
        target,
        name=None,
        *,
        brief=False,
        returns=False,
    ):
        """Get help for any Python object (class, function, module, method, etc.).

        Unlike help() which is a classmethod for HelpMixin subclasses, about()
        is a static method that works on any object.

        Parameters:
            target: Any Python object to inspect (class, function, module, etc.)
            name: Name of a specific member to get help for (if target is a class/module).
            brief: If True, show one-line summary instead of full docstring.
            returns: If True, return the help string instead of printing.

        Returns:
            Help string if returns=True, otherwise None.

        Examples:
            >>> HelpMixin.about(some_function)           # Help for a function
            >>> HelpMixin.about(SomeClass)               # Help for any class
            >>> HelpMixin.about(SomeClass, "method")     # Help for a method
            >>> HelpMixin.about(some_module)             # Help for a module
        """
        inspect = HelpMixin._get_inspect()

        # If name provided, get that attribute from target
        if name is not None:
            member = getattr(target, name, None)
            if member is None:
                target_name = getattr(target, "__name__", type(target).__name__)
                msg = f"'{target_name}' has no member '{name}'"
                if returns:
                    return msg
                print(msg)
                return None
            target = member

        # Get the name for display
        if hasattr(target, "__name__"):
            display_name = target.__name__
        elif hasattr(target, "__class__"):
            display_name = type(target).__name__
        else:
            display_name = str(target)

        try:
            # Get signature if callable
            sig = ""
            if callable(target):
                try:
                    unwrapped = inspect.unwrap(target)
                    sig = str(inspect.signature(unwrapped))
                except (ValueError, TypeError):
                    pass

            # Get docstring
            doc = inspect.getdoc(target) or "No documentation available."

            if brief:
                # Just first line
                first_line = doc.split("\n")[0].strip()
                output = f"{display_name}{sig}: {first_line}"
            else:
                # Full output
                header = f"{display_name}{sig}"
                sep = "=" * len(header)
                output = f"{header}\n{sep}\n\n{doc}"

            if returns:
                return output
            print(output)
            return None

        except Exception as e:
            msg = f"Error inspecting {display_name}: {e}"
            if returns:
                return msg
            print(msg)
            return None


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from abc import abstractmethod

    # Example usage
    class BaseClass(HelpMixin):
        """Base class with some methods."""

        def base_method(self) -> str:
            """A method from the base class."""
            return "base"

    class ExampleClass(BaseClass):
        """An example class demonstrating HelpMixin usage."""

        def __init__(self, value: int):
            """Initialize with a value."""
            self.value = value

        def process(self, data: str, flag: bool = True) -> str:
            """Process the input data.

            Parameters:
                data: The input string to process.
                flag: Optional flag.

            Returns:
                The processed string.
            """
            return data.upper()

        async def async_method(self) -> None:
            """An async method example."""
            pass

        def generator_method(self):
            """A generator method example."""
            yield 1
            yield 2

        @property
        def doubled(self) -> int:
            """Return the value doubled."""
            return self.value * 2

        @classmethod
        def from_string(cls, s: str) -> "ExampleClass":
            """Create instance from string representation."""
            return cls(int(s))

    # Demo different help modes
    print("=== Brief help ===")
    ExampleClass.help(brief=True)

    print("\n=== Source code ===")
    ExampleClass.source("process")

    print("\n=== Where defined ===")
    ExampleClass.where("process")

    print("\n=== MRO ===")
    ExampleClass.mro()

    print("\n=== Signature details ===")
    ExampleClass.signature("process")

    print("\n=== Classify member ===")
    ExampleClass.classify("async_method")

    print("\n=== List methods ===")
    ExampleClass.list_members("methods")


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
