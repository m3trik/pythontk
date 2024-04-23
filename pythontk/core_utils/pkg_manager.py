# !/usr/bin/python
# coding=utf-8
import sys
import re
import json
import subprocess
from pythontk.core_utils import help_mixin


class PkgManagerHelperMixin:
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

    def is_outdated(self, package_name):
        """Check if a specific package is outdated.

        Example:
            is_outdated = pkg_mgr.is_outdated("numpy")
        """
        all_outdated = self.list_outdated_packages()
        if isinstance(all_outdated, list):
            return any(
                pkg.get("name", "").lower() == package_name.lower()
                for pkg in all_outdated
            )
        elif isinstance(all_outdated, dict):
            # Adjust logic here based on how the data is structured in the dictionary
            return package_name.lower() in (
                pkg_name.lower() for pkg_name in all_outdated.keys()
            )
        else:
            raise TypeError("Unexpected data type for outdated packages")


class PkgManager(PkgManagerHelperMixin, help_mixin.HelpMixin):
    """A class that encapsulates package management functionalities using pip.

    This class combines the capabilities of PkgManagerHelperMixin and HelpMixin to
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

    The PkgManager class serves as a comprehensive tool for package management in Python,
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
    pkg_mgr = PkgManager("C:/Program Files/Autodesk/Maya2023/bin/mayapy.exe")
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
