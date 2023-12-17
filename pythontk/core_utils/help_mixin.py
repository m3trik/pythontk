# !/usr/bin/python
# coding=utf-8
import inspect
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
        detect_format: Analyze a docstring to detect its format based on predefined templates.
        format_docstring: Format a given docstring based on its detected format.
        format_custom_docstring: Special handling for formatting custom-style docstrings.

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
        ],
    }

    @classmethod
    def help(cls, method_name=None):
        """Display help information for the current class or a specific method.

        Parameters:
            method_name: Optional. The name of the method to get help for.
        """
        entity = cls
        help_msg = f"Available Methods (use '{cls.__name__}.help('method_name')' for full docstring with examples):"

        if method_name:
            # Display help for a specific method
            method = getattr(entity, method_name, None)
            if method and inspect.isfunction(method) and method.__doc__:
                doc = textwrap.dedent(method.__doc__).strip()  # Normalize indentation
                formatted_doc = cls.format_docstring(doc)
                print(f"Help for method '{method_name}':\n{formatted_doc}")
            else:
                print(f"No help available for method '{method_name}'.")
        else:
            # Display general help for all methods in the class
            print(help_msg)
            for attr_name in dir(entity):
                if not attr_name.startswith("_"):  # Filter out private attributes
                    attr = getattr(entity, attr_name)
                    if callable(attr) and attr.__doc__:
                        first_line_doc = (
                            textwrap.dedent(attr.__doc__).split("\n")[0].strip()
                        )
                        print(f"  - {attr_name}: {first_line_doc}")

    @classmethod
    def detect_format(cls, docstring):
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
    def format_docstring(cls, docstring):
        """Format the docstring based on its detected format.

        Parameters:
            docstring: The original docstring to format.
        """
        format_name = cls.detect_format(docstring)
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

    @classmethod
    def format_custom_docstring(cls, lines, headers):
        """Special formatting for the custom docstring style.

        Parameters:
            lines: List of lines in the docstring.
            headers: List of section headers for the custom format.
        """
        formatted_doc = ["Description:\n"]
        section_header = None
        for line in lines:
            line = line.strip()
            if line in headers:
                if section_header is None:
                    # Add divider only for the first section
                    formatted_doc.append("\n")
                section_header = line
                formatted_doc.append(line + "\n")
            elif section_header:
                formatted_doc.append("    " + line)
            else:
                formatted_doc.append("    " + line)
        return "\n".join(formatted_doc)


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
