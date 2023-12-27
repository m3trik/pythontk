# !/usr/bin/python
# coding=utf-8
import inspect
from typing import get_type_hints
import textwrap


class HelpMixin:
    """A mixin class for providing dynamic help information based on different docstring formats.

    This mixin class facilitates the generation and display of formatted help information
    for methods within a class. It supports multiple established docstring formats including
    Google Style, NumPy/SciPy Style, reStructuredText, as well as a custom format. The class
    provides methods to detect the docstring format, format the docstring for display, and
    output help information for either a specific method or all methods within a class.

    Attributes:
        FORMATS (dict): A dictionary defining the structure for known docstring formats.
                        Each format is identified by a key, and the value is a list of
                        recognizable section headers for that format.
    Methods:
        help: Display help information for the current class or a specific method.
        _detect_format: Analyze a docstring to detect its format based on predefined templates.
        _format_docstring: Format a given docstring based on its detected format.

    The `help` method can be used to print formatted help information for any method
    within the class or the class itself, providing a quick and convenient way to access
    documentation. The mixin is designed to be flexible and adaptable to different
    documentation styles, making it a versatile tool for enhancing code readability and maintainability.
    """

    # Define the structure for known docstring formats
    FORMATS = {
        "google": [
            "Args:",
            "Returns:",
            "Raises:",
        ],
        "numpy": [
            "Parameters",
            "Returns",
            "Raises",
            "See Also",
            "Notes",
            "References",
            "Examples",
        ],
        "restructuredtext": [
            ":param",
            ":returns:",
            ":raises:",
        ],
        "custom": [
            "Parameters:",
            "Properties:",
            "Methods:",
            "Attributes:",
            "Example:",
            "Returns:",
            "Raises:",
            "Notes:",
            "Usage:",
        ],
    }

    @classmethod
    def help(cls, method_name=None):
        """Display help information for the current class or a specific method.

        Parameters:
            method_name: Optional. The name of the method to get help for.
        """
        entity = cls
        help_msg = f"Class: {cls.__name__}\nInherits from: {', '.join([base.__name__ for base in cls.__bases__])}\n\n"
        help_msg += f"Available Methods (use '{cls.__name__}.help('method_name')' for full docstring with examples):\n"
        print(help_msg)

        if method_name:
            method = getattr(entity, method_name, None)
            if method and inspect.isfunction(method):
                cls._display_method_help(method, method_name)
            else:
                print(f"No help available for method '{method_name}'.")
        else:
            # Display general help for methods unique to Switchboard class
            for attr_name in dir(entity):
                if attr_name.startswith("_") or hasattr(super(cls, entity), attr_name):
                    continue  # Skip private attributes and inherited methods

                attr = getattr(entity, attr_name)
                if callable(attr):
                    try:
                        signature = inspect.signature(attr)
                        method_sig = f"{attr_name}{signature}"
                        first_line_doc = ""
                        if attr.__doc__:
                            first_line_doc = (
                                textwrap.dedent(attr.__doc__).split("\n")[0].strip()
                            )
                        print(f"  - {method_sig}:\n\t{first_line_doc}\n")
                    except (TypeError, ValueError):
                        print(f"  - {attr_name}: (Method with complex signature)\n")

    @classmethod
    def _display_method_help(cls, method, method_name):
        """Display detailed help for a specific method."""
        doc = textwrap.dedent(method.__doc__).strip() if method.__doc__ else ""
        formatted_doc = cls._format_docstring(doc)

        # Get method signature and type hints
        signature = inspect.signature(method)
        type_hints = get_type_hints(method)

        print(f"Help for method '{method_name}':\n{formatted_doc}")
        print(f"\nSignature: {method_name}{signature}")

        # Display argument types and default values
        for param_name, param in signature.parameters.items():
            param_type = type_hints.get(param_name, "Any")
            default = (
                f" = {param.default}"
                if param.default is not inspect.Parameter.empty
                else ""
            )
            print(f"    {param_name}: {param_type}{default}")

        # Display return type
        return_type = type_hints.get("return", "None")
        print(f"Returns: {return_type}")

    @classmethod
    def _detect_format(cls, docstring):
        """Detect the docstring format.

        Parameters:
            docstring: The original docstring to analyze.

        Returns:
            The name of the detected format.
        """
        for format_name, headers in cls.FORMATS.items():
            if any(header in docstring for header in headers):
                return format_name
        return "custom"

    @classmethod
    def _format_docstring(cls, docstring):
        """Format the docstring based on its detected format.

        Parameters:
            docstring: The original docstring to format.
        """
        format_name = cls._detect_format(docstring)
        headers = cls.FORMATS[format_name]
        formatted_doc = ["\nDescription:"]
        lines = docstring.split("\n")

        section_header = None
        for line in lines:
            line = line.strip()
            if line in headers:
                if section_header:
                    formatted_doc.append("\n")  # Divider between sections
                section_header = line
                formatted_doc.append(f"{line}")
            elif section_header:
                formatted_doc.append(f"    {line}")
            else:
                formatted_doc.append(f"    {line}")

        return "\n".join(formatted_doc)


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
