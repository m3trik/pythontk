# !/usr/bin/python
# coding=utf-8
import sys
import re
import json
import subprocess
from pythontk.core_utils import help_mixin


class _PackageManagerHelperMixin:
    """A mixin class to provide package management capabilities using pip.

    This mixin class offers methods to interact with the Python Package Index (PyPI)
    using pip. It allows for the installation, uninstallation, listing, and updating
    of Python packages, as well as retrieving package details and versions.

    Methods:
        install: Install a specified Python package.
        uninstall: Uninstall a specified Python package.
        list_packages: List all installed Python packages.
        package_details: Retrieve detailed information about a specified package.
        update: Update a specified Python package to the latest version.
        package_version: Get the installed version of a specified package.
        list_outdated_packages: List all outdated Python packages in the environment.
        is_outdated: Check if a specified package is outdated.

    The class is intended to be used as a mixin, providing package management functionalities
    to other classes that deal with Python environments and package management.
    """

    def install(self, package_name):
        """Install a package.

        Example:
            pkg_mgr.install("numpy")
        """
        self.pip(f"install {package_name}")

    def uninstall(self, package_name):
        """Uninstall a package.

        Example:
            pkg_mgr.uninstall("numpy")
        """
        self.pip(f"uninstall {package_name} -y")

    def list_packages(self):
        """List installed packages.

        Example:
            installed_packages = pkg_mgr.list_packages()
        """
        return self.pip("list")

    def package_details(self, package_name):
        """Show details of a specific package.

        Example:
            details = pkg_mgr.package_details("numpy")
        """
        return self.pip(f"show {package_name}")

    def update(self, package_name):
        """Update a package to the latest version.

        Example:
            pkg_mgr.update("numpy")
        """
        self.pip(f"install --upgrade {package_name}")

    def installed_version(self, package_name):
        """Get the installed version of a package."""
        package_info = self.package_details(package_name)
        return package_info.get("version", "Not installed")

    def latest_version(self, package_name):
        """Get the latest version of a package from PyPI using the standard library."""
        import json
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        url = f"https://pypi.org/pypi/{package_name}/json"
        request = Request(url)

        try:
            with urlopen(request) as response:
                data = json.loads(response.read().decode())
                return data["info"]["version"]  # Return the latest version
        except HTTPError as e:
            raise RuntimeError(
                f"HTTP error occurred while fetching the latest version of {package_name}: {e.code} {e.reason}"
            )
        except URLError as e:
            raise RuntimeError(
                f"URL error occurred while fetching the latest version of {package_name}: {e.reason}"
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse JSON response for {package_name}: {e.msg}"
            )

    def list_outdated_packages(self):
        """List all outdated packages.

        Example:
            outdated_packages = pkg_mgr.list_outdated_packages()
        """
        outdated_packages = self.pip("list --outdated --format=json")
        if isinstance(outdated_packages, str):
            try:
                return json.loads(outdated_packages)
            except json.JSONDecodeError:
                raise ValueError("Failed to decode JSON from outdated packages list")
        elif isinstance(outdated_packages, dict):
            # If the output is already a dictionary, return it as is or extract the relevant data
            return outdated_packages
        else:
            raise TypeError("Unexpected data type for outdated packages list")

    def is_outdated(self, package_name: str) -> bool:
        """Efficiently check if a specific package is outdated."""
        return self.installed_version(package_name) != self.latest_version(package_name)


class _PkgVersionCheck:
    """A class to check package versions and detect available updates.

    This class provides functionality to check if a package is up-to-date by comparing
    the installed version against the latest version available on PyPI.

    Attributes:
        _package_name: The name of the package to check.
        _python_path: Path to the Python interpreter to use for checking.
        _installed_ver: Cached installed version of the package.
        _latest_ver: Cached latest version of the package from PyPI.
    """

    _package_name: str = None
    _python_path: str = None
    _installed_ver: str = ""
    _latest_ver: str = ""

    def __init__(self, package_name=None, python_path=None):
        """Initialize the _PkgVersionCheck.

        Parameters:
            package_name (str, optional): The name of the package to check.
            python_path (str, optional): Path to the Python interpreter to use.
                Defaults to the current Python interpreter.
        """
        if package_name:
            self._package_name = package_name
        if python_path:
            self._python_path = python_path
        else:
            self._python_path = sys.executable

    def start_version_check(self, package_name=None, python_path=None) -> None:
        """Start a version check in a background thread.

        Parameters:
            package_name (str, optional): The name of the package to check.
                If not provided, uses the value set in __init__ or the class default.
            python_path (str, optional): Path to the Python interpreter to use.
                If not provided, uses the value set in __init__ or the class default.
        """
        if package_name:
            self._package_name = package_name
        if python_path:
            self._python_path = python_path

        # Make sure we have a package name
        if not self._package_name:
            raise ValueError("Package name must be provided")

        # Start a background thread to check versions
        import threading

        threading.Thread(target=self.check_version, daemon=True).start()

    @property
    def new_version_available(self) -> bool:
        """Check if a new version of the package is available.

        Returns:
            bool: True if a newer version is available, False otherwise.
        """
        try:
            return (
                self.installed_ver != self.latest_ver
                and self.installed_ver
                and self.latest_ver
            )
        except AttributeError:
            return False

    @property
    def installed_ver(self) -> str:
        """Get the installed version of the package.

        Returns:
            str: The installed version, or an empty string if not cached.
        """
        return getattr(self, "_installed_ver", "")

    @property
    def latest_ver(self) -> str:
        """Get the latest version of the package from PyPI.

        Returns:
            str: The latest version, or an empty string if not cached.
        """
        return getattr(self, "_latest_ver", "")

    def check_version(self, package_name=None, python_path=None) -> None:
        """Check the installed and latest versions of the package.

        This method updates the _installed_ver and _latest_ver attributes.
        Network errors are handled silently to prevent blocking the application.

        Parameters:
            package_name (str, optional): The name of the package to check.
                If not provided, uses the value set in __init__ or the class default.
            python_path (str, optional): Path to the Python interpreter to use.
                If not provided, uses the value set in __init__ or the class default.
        """
        if package_name:
            self._package_name = package_name
        if python_path:
            self._python_path = python_path

        # Make sure we have a package name
        if not self._package_name:
            raise ValueError("Package name must be provided")

        # Create a package manager for the specified Python interpreter
        pkg_mgr = PackageManager(python_path=self._python_path)

        try:
            # Get the installed and latest versions
            self._installed_ver = pkg_mgr.installed_version(self._package_name)
            self._latest_ver = pkg_mgr.latest_version(self._package_name)
        except RuntimeError:
            # Silently handle network errors - version check will be skipped
            # The installed version is still retrieved, but latest version remains empty
            self._latest_ver = ""
            pass


class _PkgVersionUtils:
    """Utilities for managing package version information in files.

    This class provides static methods for updating version numbers in source files
    and managing package version specifications in requirements files.
    """

    @staticmethod
    def update_version(
        filepath: str,
        change: str = "increment",
        version_part: str = "patch",
        max_version_parts: tuple = (99, 99),
        version_regex: str = r"__version__\s*=\s*['\"](\d+)\.(\d+)\.(\d+)['\"]",
    ) -> str:
        """Update the version number in a text file.

        This function updates the version number in a text file depending on its state.
        The version number is represented as a string in the format 'x.y.z', where x, y,
        and z are integers and it matches the provided regex pattern.

        Args:
            filepath: The path to the text file containing the version number.
            change: The type of change, either 'increment' or 'decrement'. Defaults to 'increment'.
            version_part: The part of the version to update ('major', 'minor', or 'patch').
            max_version_parts: Maximum values for the minor and patch version parts.
            version_regex: Regex pattern for finding the version string.

        Returns:
            str: The new version number or empty string if not found.
        """
        import re
        from pythontk.file_utils._file_utils import FileUtils

        lines = FileUtils.get_file_contents(filepath, as_list=True)

        version_pattern = re.compile(version_regex)
        max_minor, max_patch = max_version_parts

        version = ""
        for i, line in enumerate(lines):
            match = version_pattern.search(line)
            if match:
                major, minor, patch = map(int, match.groups())

                if version_part == "patch":
                    if change == "increment":
                        patch = (patch + 1) % (max_patch + 1)
                        if patch == 0:
                            minor = (minor + 1) % (max_minor + 1)
                            major += minor == 0
                    elif change == "decrement":
                        if patch == 0:
                            patch = max_patch
                            minor = (
                                (minor - 1) % (max_minor + 1)
                                if minor > 0
                                else max_minor
                            )
                            major -= minor == max_minor
                        else:
                            patch -= 1
                    else:
                        raise ValueError(
                            "Invalid change parameter. Use either 'increment' or 'decrement'."
                        )
                elif version_part == "minor":
                    if change == "increment":
                        minor = (minor + 1) % (max_minor + 1)
                        major += minor == 0
                    elif change == "decrement":
                        minor = (minor - 1) % (max_minor + 1)
                    else:
                        raise ValueError(
                            "Invalid change parameter. Use either 'increment' or 'decrement'."
                        )
                elif version_part == "major":
                    if change == "increment":
                        major += 1
                    elif change == "decrement":
                        major = max(0, major - 1)
                    else:
                        raise ValueError(
                            "Invalid change parameter. Use either 'increment' or 'decrement'."
                        )
                else:
                    raise ValueError(
                        "Invalid version_part parameter. Use either 'major', 'minor', or 'patch'."
                    )

                version = f"{major}.{minor}.{patch}"

                # Preserve the original format of the line
                new_line = re.sub(
                    version_regex,
                    lambda m: m.group(0).replace(
                        m.group(1) + "." + m.group(2) + "." + m.group(3), version
                    ),
                    line,
                )
                lines[i] = new_line
                break

        FileUtils.write_to_file(filepath, lines)
        if not version:
            print(f"Error: No version found in {filepath}")
        return version

    @staticmethod
    def update_requirements(file_path=None, inc=None, exc=None) -> list:
        """Update the requirements.txt file with current versions of packages.

        Args:
            file_path: Path to the requirements.txt file. Defaults to the caller's directory.
                      If a relative path is given, it's relative to the caller's directory.
            inc: Patterns or objects to include in the update.
            exc: Patterns or objects to exclude from the update.

        Returns:
            list: Updated requirements with their versions.
                 Example: ['package1==1.0.0', 'package2==2.3.4']
        """
        import os
        import inspect
        import pkg_resources
        from pythontk.iter_utils._iter_utils import IterUtils

        # Determine the caller's directory
        caller_frame = inspect.stack()[1]
        caller_path = caller_frame.filename
        caller_dir = os.path.dirname(caller_path)

        if file_path is None:
            file_path = os.path.join(caller_dir, "requirements.txt")
        else:
            file_path = os.path.abspath(os.path.join(caller_dir, file_path))

        try:
            with open(file_path, "r") as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    package_name = line.strip().split("==")[0]
                    if package_name in IterUtils.filter_list([package_name], inc, exc):
                        try:
                            version = pkg_resources.get_distribution(
                                package_name
                            ).version
                            updated_lines.append(f"{package_name}=={version}\n")
                        except Exception as e:
                            print(f"Error updating version for {package_name}: {e}")
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)

            with open(file_path, "w") as file:
                file.writelines(updated_lines)

        except FileNotFoundError:
            print(f"File not found: {file_path}")

        return [
            line.strip()
            for line in updated_lines
            if line.strip() and not line.startswith("#")
        ]


class PackageManager(
    _PkgVersionCheck, _PkgVersionUtils, _PackageManagerHelperMixin, help_mixin.HelpMixin
):
    """A class that encapsulates package management functionalities using pip.

    This class combines the capabilities of _PackageManagerHelperMixin and HelpMixin to
    offer an integrated environment for managing Python packages and accessing help
    documentation for different methods. It provides an interface to execute pip commands
    and parse their output, along with handling Python environment-specific information.

    Methods:
        pip: Execute a pip command and handle its output.
        _get_startupinfo: Prepare startup information for subprocesses on Windows.
        _parse_command: Convert a pip command string into a list format.
        _process_output: Process and format the output from pip commands.
        _is_informational_message: Check if a stderr message is informational.
        _convert_output: Convert command output into a structured format.
        _is_list_format: Determine if the output is in a list format.
        _parse_list_format: Parse list format output into a dictionary.
        _is_key_value_format: Determine if the output is in key-value pair format.
        _parse_key_value_format: Parse key-value pair output into a dictionary.

    The PackageManager class serves as a comprehensive tool for package management in Python,
    supporting a range of pip functionalities and enhancing user experience with formatted
    help messages and command outputs.
    """

    def __init__(self, python_path=sys.executable):
        self.python_path = python_path

    def _get_startupinfo(self):
        """Prepare startup information for subprocess on Windows."""
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            return startupinfo
        return None

    def pip(self, command, output_as_string=False):
        """Execute a pip command and return the output."""
        full_command = [self.python_path, "-m", "pip"] + self._parse_command(command)
        startupinfo = self._get_startupinfo()

        try:
            result = subprocess.run(
                full_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                startupinfo=startupinfo,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command '{' '.join(full_command)}' failed: {e.stderr}")

        return self._process_output(result.stdout, result.stderr, output_as_string)

    def _parse_command(self, command):
        """Parse the pip command from string to list format."""
        if isinstance(command, str):
            return re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', command)
        return command

    def _process_output(self, stdout, stderr, output_as_string):
        """Process the output from the pip command."""
        if self._is_informational_message(stderr):
            stderr = ""

        output = stdout if not stderr else stdout + "\nError:\n" + stderr
        return output.strip() if output_as_string else self._convert_output(output)

    def _is_informational_message(self, stderr):
        """Check if the stderr message is informational."""
        informational_patterns = ["A new release of pip available"]
        return any(pattern in stderr for pattern in informational_patterns)

    def _convert_output(self, output):
        """Convert the output into a dictionary, list, or return as string."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            lines = output.split("\n")
            if self._is_list_format(lines):
                return self._parse_list_format(lines)
            elif self._is_key_value_format(lines):
                return self._parse_key_value_format(lines)
            return output

    def _is_list_format(self, lines):
        """Determine if the output is in 'pip list' table format."""
        return all(len(line.split()) >= 2 for line in lines if line)

    def _parse_list_format(self, lines):
        """Parse 'pip list' table format into a dictionary."""
        output_dict = {}
        data_start = False  # Flag to indicate the start of data lines
        for line in lines:
            if "---" in line:  # Detect the line of dashes
                data_start = True
                continue
            if data_start and line.strip():
                package, version = line.split(maxsplit=1)
                output_dict[package] = version
        return output_dict

    def _is_key_value_format(self, lines):
        """Determine if the output is in key-value pair format."""
        return any(":" in line for line in lines if line)

    def _parse_key_value_format(self, lines):
        """Parse key-value pair format into a dictionary."""
        output_dict = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                output_dict[key.strip().lower()] = value.strip()
        return output_dict


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pkg_mgr = PackageManager("C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe")
    # output = pkg_mgr.get_installed_packages("")

    # Test various pip commands
    output = pkg_mgr.pip("list", output_as_string=1)
    # print(output)

    output = pkg_mgr.pip("show pythontk")
    print(output["version"])

    # output = pkg_mgr.pip("freeze")
    # print(output)

    # output = pkg_mgr.pip("check")
    # print(output)

    print("Installed Packages:")
    print(pkg_mgr.list_packages())

    # Show details of a specific package (e.g., 'numpy')
    print("\nPackage Details (numpy):")
    print(pkg_mgr.package_details("numpy"))

    # List outdated packages
    print("\nOutdated Packages:")
    print(pkg_mgr.list_outdated_packages())

    # Check if a specific package (e.g., 'numpy') is outdated
    print("\nIs 'numpy' outdated?")
    print(pkg_mgr.is_outdated("numpy"))

    pkg_mgr.help()

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
