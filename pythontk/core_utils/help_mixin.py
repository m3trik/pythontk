# !/usr/bin/python
# coding=utf-8
"""HelpMixin - Enhanced help system leveraging Python's built-in help infrastructure.

This module provides a mixin class that enhances class introspection by wrapping
and extending Python's built-in help() functionality with filtering, sorting,
and targeted output options.

It renders; :class:`~pythontk.ApiSurface` reads. What a member is (its kind,
signature, summary, flags, where it is defined) comes from the engine every
reader of the public API shares -- the registries, the ``api.*`` operations,
the drift gate -- so ``.help()`` and those readers can never describe one
member two ways.
"""

import json
import inspect
import pydoc
from typing import Any, Optional

from pythontk.core_utils.api_surface.api_surface import ApiSurface
from pythontk.core_utils.symbol_record import SymbolRecord


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
        MyClass.show_mro()
    """

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
        as_dict: bool = False,
        as_json: bool = False,
    ) -> Any:
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
            >>> MyClass.help(as_dict=True)               # Structured dict (agent/RPC)
            >>> MyClass.help("method", as_json=True)     # One member as JSON
        """
        # Structured output for programmatic / cross-process (live-DCC) callers.
        if as_dict or as_json:
            return cls._structured_help(
                name=name,
                members=members,
                inherited=inherited,
                sort=sort,
                private=private,
                as_json=as_json,
            )

        # Specific member requested - use built-in help or return targeted info
        if name is not None:
            return cls._help_for_member(name, brief=brief, returns=returns)

        # Full help requested - either built-in or filtered
        if members is None and inherited and not brief and not sort and not private:
            # Default case: just use built-in help
            if returns:
                return cls._render_plain(cls)
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
            msg = cls._missing(name)
            if returns:
                return msg
            print(msg)
            return None

        if brief:
            output = cls._format_member_brief(ApiSurface.record(cls, name))
        else:
            # Use built-in help for full documentation
            if returns:
                return cls._render_plain(member)
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
            lines.append(f"    {ApiSurface.summary(cls.__doc__)}")
            lines.append("")

        records = ApiSurface.members(cls, members, inherited=inherited, private=private)
        if sort:
            records = sorted(records, key=lambda r: r.name)

        # Format output
        if records:
            lines.append("Members:")
            for record in records:
                if brief:
                    lines.append(cls._format_member_brief(record))
                else:
                    lines.append(cls._format_member_full(record))

        output = "\n".join(lines)
        if returns:
            return output
        print(output)
        return None

    @classmethod
    def _missing(cls, name: str) -> str:
        """What every accessor says for a name the class does not have."""
        return f"'{cls.__name__}' has no member '{name}'"

    @classmethod
    def _type_label(cls, record: SymbolRecord) -> str:
        """``[kind, flags]`` for one member's line."""
        parts = [record.kind] + ApiSurface.flags(getattr(cls, record.name, None))
        return f"[{', '.join(parts)}]"

    @classmethod
    def _format_member_brief(cls, record: SymbolRecord) -> str:
        """Format a single member as a brief one-liner."""
        head = f"    {record.name}{record.signature}  {cls._type_label(record)}"
        return f"{head}\n        {record.summary}" if record.summary else head

    @classmethod
    def _format_member_full(cls, record: SymbolRecord) -> str:
        """Format a single member with full documentation."""
        lines = [f"\n    {record.name}{record.signature}  {cls._type_label(record)}"]
        doc_str = inspect.getdoc(getattr(cls, record.name, None))
        if doc_str:
            lines += [f"        {line}" for line in doc_str.split("\n")]
        return "\n".join(lines)

    @staticmethod
    def _render_plain(target: Any) -> str:
        """``pydoc`` rendering with terminal bold-overstrike stripped.

        ``render_doc`` emits ``C\\bCl\\bla\\bas\\bss`` backspace pairs for bolding.
        The printing path feeds those to the pager, which resolves them; a
        *returned* string has no pager, and ``returns=True`` exists for
        programmatic consumers (CLI, agent tooling, cross-process RPC) where the
        escapes would surface as doubled-letter garbage.
        """
        return pydoc.plain(pydoc.render_doc(target, title="%s"))

    # -------------------------------------------------------------------------
    # Structured Output (SymbolRecord)
    # -------------------------------------------------------------------------

    @classmethod
    def _structured_help(
        cls,
        *,
        name: Optional[str],
        members: Optional[str],
        inherited: bool,
        sort: bool,
        private: bool,
        as_json: bool,
    ) -> Any:
        """Build the structured payload for ``help(as_dict=...)``.

        A single member yields one enriched record dict
        (:meth:`ApiSurface.enrich`); the whole class yields ``{class, bases,
        mro, doc, members:[...]}``, the shape agent tooling consumes.
        """
        if name is not None:
            record = ApiSurface.record(cls, name)
            payload: Any = (
                {"error": cls._missing(name)}
                if record is None
                else ApiSurface.enrich(cls, record)
            )
        else:
            records = ApiSurface.members(
                cls, members, inherited=inherited, private=private
            )
            if sort:
                records = sorted(records, key=lambda r: r.name)
            payload = {
                "class": cls.__name__,
                "bases": [b.__name__ for b in cls.__bases__],
                "mro": [k.__name__ for k in inspect.getmro(cls)],
                "doc": ApiSurface.summary(cls.__doc__),
                "members": [ApiSurface.enrich(cls, r) for r in records],
            }
        return json.dumps(payload, indent=2) if as_json else payload

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
            msg = cls._missing(name)
            if returns:
                return msg
            print(msg)
            return None

        try:
            source = ApiSurface.source_of(target)
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
            msg = cls._missing(name)
            if returns:
                return msg
            print(msg)
            return None

        location = ApiSurface.location(target) or ApiSurface.NO_LOCATION

        if returns:
            return location
        print(location)
        return None

    @classmethod
    def show_mro(
        cls,
        *,
        brief: bool = False,
        returns: bool = False,
    ) -> Optional[str]:
        """Show the method resolution order (inheritance chain) for this class.

        Named ``show_mro`` (not ``mro``) deliberately: defining ``mro`` on a
        mixin shadows Python's built-in ``type.mro()`` for every subclass.

        Parameters:
            brief: If True, show only class names.
                   If False (default), include module paths.
            returns: If True, return the MRO string instead of printing.

        Returns:
            MRO string if returns=True, otherwise None.

        Examples:
            >>> MyClass.show_mro()                # Full MRO with modules
            >>> MyClass.show_mro(brief=True)      # Just class names
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
            msg = cls._missing(name)
            if returns:
                return msg
            print(msg)
            return None

        output = cls._signature_detail(name, member)
        if returns:
            return output
        print(output)
        return None

    @classmethod
    def _signature_detail(cls, name: str, member: Any) -> str:
        """Render the parameter breakdown for an already-resolved *member*.

        Split out from :meth:`signature` (which resolves a name against ``cls``)
        so the same rendering serves objects that are not HelpMixin subclasses -
        the ``python -m pythontk`` CLI inspects arbitrary resolved targets, and
        would otherwise need its own copy of this formatting.
        """
        if not callable(member):
            return f"'{name}' is not callable"

        try:
            # Unwrap to get true signature
            unwrapped = inspect.unwrap(member)
            sig = inspect.signature(unwrapped)
        except (ValueError, TypeError) as e:
            return f"Cannot retrieve signature: {e}"

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
            lines.append(f"Returns: {cls._format_annotation(sig.return_annotation)}")

        return "\n".join(lines)

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
        as_dict: bool = False,
        as_json: bool = False,
    ) -> Any:
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
            >>> MyClass.classify(as_dict=True)  # Enriched records (type/owner/flags)
        """
        if as_dict or as_json:
            if name is not None:
                record = ApiSurface.record(cls, name)
                payload: Any = (
                    {"error": cls._missing(name)}
                    if record is None
                    else ApiSurface.enrich(cls, record)
                )
            else:
                payload = [ApiSurface.enrich(cls, r) for r in ApiSurface.members(cls)]
            return json.dumps(payload, indent=2) if as_json else payload

        if name is not None:
            # Classify single member
            member = getattr(cls, name, None)
            if member is None:
                msg = cls._missing(name)
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
                flags = ", ".join(ApiSurface.flags(attr.object))
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
        record = ApiSurface.record(cls, name)
        lines = [f"Member: {name}", "", f"Type: {record.kind}"]
        defined_in = ApiSurface.defined_in(cls, name)
        if defined_in:
            lines.append(f"Defined in: {defined_in}")
        flags = ApiSurface.flags(member)
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")
        if record.signature:
            lines.append(f"Signature: {name}{record.signature}")
        if record.summary:
            lines.append(f"Summary: {record.summary}")
        return "\n".join(lines)

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
        as_dict: bool = False,
        as_json: bool = False,
    ) -> Any:
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
            >>> MyClass.list_members(as_dict=True)         # Enriched member records
        """
        records = ApiSurface.members(cls, members, inherited=inherited, private=private)
        if sort:
            records = sorted(records, key=lambda r: r.name)
        if as_dict or as_json:
            payload = [ApiSurface.enrich(cls, r) for r in records]
            return json.dumps(payload, indent=2) if as_json else payload

        names = [r.name for r in records]

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
        signature=False,
        returns=False,
        as_dict=False,
        as_json=False,
    ):
        """Get help for any Python object (class, function, module, method, etc.).

        Unlike help() which is a classmethod for HelpMixin subclasses, about()
        is a static method that works on any object.

        Parameters:
            target: Any Python object to inspect (class, function, module, etc.)
            name: Name of a specific member to get help for (if target is a class/module).
            brief: If True, show one-line summary instead of full docstring.
            signature: If True, show its parameter breakdown instead (as
                :meth:`signature` does for a member).
            returns: If True, return the help string instead of printing.

        Returns:
            Help string if returns=True, otherwise None.

        Examples:
            >>> HelpMixin.about(some_function, signature=True)
            >>> HelpMixin.about(some_function)           # Help for a function
            >>> HelpMixin.about(SomeClass)               # Help for any class
            >>> HelpMixin.about(SomeClass, "method")     # Help for a method
            >>> HelpMixin.about(some_module)             # Help for a module
        """
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
            sig = ApiSurface.signature_of(target)
            doc = inspect.getdoc(target) or "No documentation available."

            if as_dict or as_json:
                info = {
                    "name": display_name,
                    "signature": sig,
                    "module": getattr(target, "__module__", None),
                    "doc": inspect.getdoc(target) or "",
                    "source": ApiSurface.location(target),
                }
                return json.dumps(info, indent=2) if as_json else info

            if signature:
                output = HelpMixin._signature_detail(display_name, target)
            elif brief:
                output = f"{display_name}{sig}: {ApiSurface.summary(doc)}"
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
    ExampleClass.show_mro()

    print("\n=== Signature details ===")
    ExampleClass.signature("process")

    print("\n=== Classify member ===")
    ExampleClass.classify("async_method")

    print("\n=== List methods ===")
    ExampleClass.list_members("methods")


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
