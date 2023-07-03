# !/usr/bin/python
# coding=utf-8
import sys
import os
import re
import json
import traceback

# from this package:
from pythontk.utils import Utils
from pythontk.iter_utils import IterUtils
from pythontk.str_utils import StrUtils


class FileUtils:
    """ """

    @staticmethod
    def is_valid(filepath: str) -> list:
        """Determine if the given file or dir is valid.

        Parameters:
            filepath (str): The path to a file.

        Returns:
            (str) The path type (ie. 'file' or 'dir') or None.
        """
        fp = os.path.expandvars(filepath)  # convert any env variables to their values.

        if os.path.isfile(fp):
            return "file"
        elif os.path.isdir(fp):
            return "dir"
        return None

    @staticmethod
    def get_file(filepath, mode="a+"):
        """Return a file object with the given mode.

        Parameters:
            filepath (str): The path to an existing file or the desired location for one to be created.
            mode (str): 'r' - Read - Default value. Opens a file for reading, error if the file does not exist.
                        'a' - Append - Opens a file for appending, creates the file if it does not exist.
                        'a+' - Read+Write - Creates a new file or opens an existing file, the file pointer position at the end of the file.
                        'w' - Write - Opens a file for writing, creates the file if it does not exist.
                        'w+' - Read+Write - Opens a file for reading and writing, creates the file if it does not exist.
                        'x' - Create - Creates a new file, returns an error if the file exists.
                        't' - Text - Default value. Text mode
                        'b' - Binary - Binary mode (e.g. images)
        Returns:
            (obj) file
        """
        try:
            with open(filepath, mode) as f:
                return f
        except OSError:
            traceback.print_exc()

    @staticmethod
    def get_file_contents(filepath: str, as_list=False) -> None:
        """Get each line of a text file as indices of a list.
        Will create a file if one doesn't exist.

        Parameters:
            filepath (str): The path to an existing text based file.
            as_list (bool): Return as a list or a string.

        Returns:
            (list)
        """
        try:
            with open(filepath, "r") as f:
                return f.readlines() if as_list else f.read()
        except OSError:
            traceback.print_exc()

    @staticmethod
    def write_to_file(filepath, lines):
        """Write the given list contents to the given file.

        Parameters:
            filepath (str): The path to an existing text based file.
            lines (list): A list of strings to write to the file.
        """
        try:
            with open(filepath, "w") as f:
                f.writelines(lines)
        except OSError:
            traceback.print_exc()

    @staticmethod
    def get_file_info(paths, returned_type):
        """Get specific file information for a list of paths.

        Parameters:
            paths (list): List of paths to retrieve file information from.
            returned_type (str): Return specific file information. Multiple types can be given using '|'.
                    ex. 'file|filename|filepath|dir|dirpath|timestamp|unixtimestamp|size|filetype'
        Returns:
            (list): List of tuples containing requested file information.
        """
        import time
        from pathlib import Path

        returnTypes = [t.strip().rstrip("s").lower() for t in returned_type.split("|")]
        results = []

        for path in paths:
            path = os.path.expandvars(path)
            path_obj = Path(path)
            if not path_obj.exists():
                continue

            is_file = path_obj.is_file()
            info = []
            for option in returnTypes:
                if option == "file":
                    info.append(path_obj.name)
                elif option == "filename":
                    info.append(path_obj.stem)
                elif option == "filepath":
                    info.append(str(path_obj))
                elif option == "dir":
                    info.append(path_obj.parent.name)
                elif option == "dirpath":
                    info.append(str(path_obj.parent))
                elif option == "timestamp":
                    info.append(time.ctime(os.path.getmtime(path)) if is_file else None)
                elif option == "unixtimestamp":
                    info.append(os.path.getmtime(path) if is_file else None)
                elif option == "size":
                    info.append(os.path.getsize(path) if is_file else None)
                elif option == "filetype":
                    info.append(path_obj.suffix if is_file else None)

            if any(x is not None for x in info):
                results.append(tuple(info))

        return results

    @staticmethod
    def get_dir_contents(
        dirPath,
        returned_type="files",
        recursive=False,
        num_threads=1,
        inc_files=[],
        exc_files=[],
        inc_dirs=[],
        exc_dirs=[],
    ):
        """Get the contents of a directory and any of its children.

        Parameters:
            dirPath (str): The path to the directory.
            returned_type (str): Return files and directories. Multiple types can be given using '|'
                            ex. 'files|dirs' (valid: 'files'(default), filenames, 'filepaths', 'dirs', 'dirpaths')
                            case insensitive. singular or plural.
            recursive (bool): When False, Return the contents of the root dir only.
            num_threads (int): The number of threads to use for processing directories and files.
                            If set to 1 or 0, multithreading will not be used.
            inc_files (str/list): Include only specific files.
            exc_files (str/list): Excluded specific files.
            inc_dirs (str/list): Include only specific child directories.
            exc_dirs (str/list): Excluded specific child directories.
                            supports using the '*' operator: startswith*, *endswith, *contains*
                            ex. *.ext will exclude all files with the given extension.
                            exclude takes precedence over include.
        Returns:
            (list): A list of files, directories, filenames, filepaths, dirs, or dirpaths based on the returned_type.

        Examples:
            get_dir_contents(dirPath, returned_type='filepaths')
            get_dir_contents(dirPath, returned_type='files|dirs')
        """
        path = os.path.expandvars(dirPath)
        returnTypes = {t.strip().rstrip("s").lower() for t in returned_type.split("|")}

        def process_directory(root, dirs, files):
            result = []
            if not recursive and root != path:
                return result

            dirs = IterUtils.filter_list(dirs, inc_dirs, exc_dirs)
            files = IterUtils.filter_list(files, inc_files, exc_files)

            if "dir" in returnTypes:
                result.extend(dirs)
            if "dirpath" in returnTypes:
                result.extend(os.path.join(root, d) for d in dirs)
            if "file" in returnTypes:
                result.extend(files)
            if "filename" in returnTypes:
                result.extend(os.path.splitext(f)[0] for f in files)
            if "filepath" in returnTypes:
                result.extend(os.path.join(root, f) for f in files)

            return result

        result = []

        if num_threads > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(process_directory, root, dirs, files): (
                        root,
                        dirs,
                        files,
                    )
                    for root, dirs, files in os.walk(path, topdown=True)
                }

                for future in as_completed(futures):
                    result.extend(future.result())
        else:
            for root, dirs, files in os.walk(path, topdown=True):
                result.extend(process_directory(root, dirs, files))

        return result

    @staticmethod
    def create_dir(filepath: str) -> None:
        """Create a directory if one doesn't already exist.

        Parameters:
            filepath (str): The path to where the file will be created.
        """
        fp = os.path.expandvars(filepath)  # convert any env variables to their values.
        try:
            if not os.path.exists(fp):
                os.makedirs(fp)
        except OSError as error:
            print(
                "{} in create_dir\n\t# Error: {}.\n\tConfirm that the following path is correct: #\n\t{}".format(
                    __file__, error, fp
                )
            )

    @staticmethod
    def get_filepath(obj, inc_filename=False):
        """Get the filepath of a class or module.

        Parameters:
            obj (obj): A python module, class, or the built-in __file__ variable.
            inc_filename (bool): Include the filename (or directory if the object is a package) in the returned path.

        Returns:
            (str)
        """
        from types import ModuleType
        import inspect

        if obj is None:
            return ""

        if isinstance(obj, str):
            filepath = obj
        elif isinstance(obj, ModuleType):
            filepath = getattr(obj, "__file__", None)
            if filepath is None and hasattr(
                obj, "__path__"
            ):  # handle namespace packages
                filepath = obj.__path__[0]
        elif callable(obj) or isinstance(obj, object):
            try:
                module = inspect.getmodule(obj)
                if module.__name__ == "__main__":
                    filepath = sys.argv[0]
                else:
                    filepath = getattr(module, "__file__", None)
                    if filepath is None and hasattr(
                        module, "__path__"
                    ):  # handle namespace packages
                        filepath = module.__path__[0]
            except AttributeError:
                raise ValueError(
                    "Unable to determine file path for object of type: ", type(obj)
                )
        else:
            raise ValueError("Invalid type for obj: ", type(obj))

        if filepath and inc_filename:
            return os.path.abspath(filepath)
        elif filepath:
            return os.path.abspath(os.path.dirname(filepath))
        else:
            return None

    @staticmethod
    @Utils.listify(threading=True)
    def format_path(p, section="", replace=""):
        """Format a given filepath(s).
        When a section arg is given, the correlating section of the string will be returned.
        If a replace arg is given, the stated section will be replaced by the given value.

        Parameters:
            p (str/list): The filepath(s) to be formatted.
            section (str): The desired subsection of the given path.
                'path' path minus filename,
                'dir'  directory name,
                'file' filename plus ext,
                'name', filename minus ext,
                'ext', file extension,
                (if '' is given, the fullpath will be returned)
        Returns:
            (str/list) List if 'strings' given as list.
        """
        if not isinstance(p, (str)):
            return p

        p = os.path.expandvars(p)  # convert any env variables to their values.
        p = re.sub(
            r"(?<!\\)\\(?!\\)", "/", p
        )  # Replace single backslashes, not followed by another backslash, with forward slashes.
        p = p.strip("/")  # strip trailing forward slashes.

        fullpath = p if "/" in p else ""
        fn = p.split("/")[-1]
        filename = fn if "." in fn and not fn.startswith(".") else ""
        path = "/".join(p.split("/")[:-1]) if filename else p
        directory = p.split("/")[-2] if (filename and path) else p.split("/")[-1]
        name = (
            "".join(filename.rsplit(".", 1)[:-1]) if filename else "" if fullpath else p
        )
        ext = filename.rsplit(".", 1)[-1]

        if section == "path":
            result = path

        elif section == "dir":
            result = directory

        elif section == "file":
            result = filename

        elif section == "name":
            result = name

        elif section == "ext":
            result = ext

        else:
            result = p

        if replace:
            result = StrUtils.rreplace(p, result, replace, 1)

        return result

    @classmethod
    def append_paths(cls, root_dir, **kwargs):
        """Append all sub-directories of the given 'root_dir' to the python path.

        Parameters:
            root_dir (str): Sub-directories of this directory will be appended to the system path.
            kwargs (optional): Any file related keyword arguments that 'get_dir_contents' allows.
                    ie. recursive, num_threads, inc_dirs, exc_dirs. But not: dirPath or returned_type.
        Returns:
            list:  the appended paths.
        """
        path = os.path.dirname(os.path.abspath(root_dir))
        return [
            sys.path.append(d) for d in cls.get_dir_contents(path, "dirs", **kwargs)
        ]

    @staticmethod
    def get_classes_from_dir(dir_path):
        """Parses the Python source files in a directory and extracts the names of all defined classes.
        This function uses the Abstract Syntax Tree (AST) module to parse the Python source files, and hence does not execute any code within these files. It only considers classes that are defined at the top level of each file, and not those defined within other classes or functions.

        Parameters:
            dir_path (str): The path to the directory containing the Python source files.

        Raises:
            Exception: If the provided directory path does not exist or is not a directory.

        Returns:
            dict: A dictionary where the keys are the names of the identified classes, and the values are the corresponding paths to the Python files where these classes are defined.

        Examples:
            >>> get_classes_from_dir('/path/to/directory')
            {'MyClass1': '/path/to/directory/my_file1.py', 'MyClass2': '/path/to/directory/my_file2.py'}

        Note:
            This function will not correctly identify classes that are defined dynamically or otherwise in non-standard ways.
        """
        import ast

        # Check if directory exists
        if not os.path.isdir(dir_path):
            raise Exception(f"Directory {dir_path} doesn't exist")

        # Create an empty dictionary to store class names and their module file paths
        widget_classes = {}

        # Iterate over each file in the directory
        for filename in os.listdir(dir_path):
            # Check if the file is a Python file
            if filename.endswith(".py"):
                with open(os.path.join(dir_path, filename), "r") as file:
                    module = ast.parse(file.read())
                    classes = [
                        node for node in module.body if isinstance(node, ast.ClassDef)
                    ]

                    for cls in classes:
                        # Check if the class inherits from QWidget
                        for base in cls.bases:
                            # Add the class name and file path to the dictionary
                            widget_classes[cls.name] = os.path.join(dir_path, filename)

        return widget_classes

    @classmethod
    @Utils.listify(threading=True)
    def time_stamp(cls, filepath, stamp="%m-%d-%Y  %H:%M"):
        """Attach or detach a modified timestamp and date to/from a given file path.

        Parameters:
            filepath (str): The full path to a file. ie. 'C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'
            stamp (str): The time stamp format.

        Returns:
            str: Filepath with attached or detached timestamp, depending on whether it initially had a timestamp.
                ie. '16:46  11-09-2021  C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb' from 'C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'
        """
        from datetime import datetime
        import os.path
        import re

        filepath = cls.format_path(filepath)

        # Check if the file path has a timestamp using regular expression
        match = re.match(r"\d{2}:\d{2}  \d{2}-\d{2}-\d{4}", filepath)
        if match:
            # If it does, return the file path without the timestamp
            return "".join(filepath.split()[2:])
        else:
            # If it doesn't, attach a timestamp
            try:
                return "{}  {}".format(
                    datetime.fromtimestamp(os.path.getmtime(filepath)).strftime(stamp),
                    filepath,
                )
            except (FileNotFoundError, OSError) as error:
                print(f"Error: {error}")
                return filepath

    @classmethod
    def update_version(
        cls,
        filepath: str,
        change: str = "increment",
        version_part: str = "patch",
        max_version_parts: tuple = (9, 9),
    ) -> None:
        """This function updates the version number in a text file depending on its state.
        The version number is defined as a line in the following format: __version__ = "0.0.0"
        The version number is represented as a string in the format 'x.y.z', where x, y, and z are integers.

        Parameters:
            filepath (str): The path to the text file containing the version number.
            change (str, optional): The type of change, either 'increment' or 'decrement'. Defaults to 'increment'.
            version_part (str, optional): The part of the version number to update, either 'major', 'minor', or 'patch'. Defaults to 'patch'.
            max_version_parts (tuple, optional): A tuple containing the maximum values for the minor and patch version parts. Defaults to (9, 9).

        Returns:
            (str): The new version number.
        """
        import re

        lines = cls.get_file_contents(filepath, as_list=True)

        version_pattern = re.compile(r"__version__\s*=\s*['\"](\d+)\.(\d+)\.(\d+)['\"]")
        max_minor, max_patch = max_version_parts

        version = ""
        for i, line in enumerate(lines):
            match = version_pattern.match(line)
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
                lines[i] = f"__version__ = '{version}'\n"
                break

        cls.write_to_file(filepath, lines)
        if not version:
            print(
                f'# Error: No version in the format: __version__ = "0.0.0" found in {filepath}'
            )
        return version

    @classmethod
    def set_json_file(cls, file):
        """Set the current json filepath.

        Parameters:
            file (str): The filepath to a json file. If a file doesn't exist, it will be created.
        """
        cls._jsonFile = file
        cls.get_file(cls._jsonFile)  # will create the file if it does not exist.

    @classmethod
    def get_json_file(cls):
        """Get the current json filepath.

        Returns:
            (str)
        """
        try:
            return cls._jsonFile
        except AttributeError:
            return ""

    @classmethod
    def set_json(cls, key, value, file=None):
        """
        Parameters:
            key () = Set the json key.
            value () = Set the json value for the given key.
            file (str): Temporarily set the filepath to a json file.
                    If no file is given, the previously set file will be used
                    if one was set.
        Example:
            set_json('hdr_map_visibility', state)
        """
        if not file:
            file = cls.get_json_file()

        assert (
            file
        ), "{} in set_json\n\t# Error: Operation requires a json file to be specified. #".format(
            __file__
        )
        assert isinstance(
            file, str
        ), "{} in set_json\n\t# Error:   Incorrect datatype: {} #".format(
            __file__, type(file).__name__
        )

        try:
            with open(file, "r") as f:
                dct = json.loads(f.read())
                dct[key] = value
        except json.decoder.JSONDecodeError:
            dct = {}
            dct[key] = value

        with open(file, "w") as f:
            f.write(json.dumps(dct))

    @classmethod
    def get_json(cls, key, file=None):
        """
        Parameters:
            key () = Set the json key.
            value () = Set the json value for the given key.
            file (str): Temporarily set the filepath to a json file.
                            If no file is given, the previously set file will
                            be used if one was set.
        Returns:
            (str)

        Example:
            get_json('hdr_map_visibility') #returns: state
        """
        if not file:
            file = cls.get_json_file()

        assert (
            file
        ), "{} in set_json\n\t# Error: Operation requires a json file to be specified. #".format(
            __file__
        )
        assert isinstance(
            file, str
        ), "{} in set_json\n\t# Error:   Incorrect datatype: {} #".format(
            __file__, type(file).__name__
        )

        try:
            with open(file, "r") as f:
                return json.loads(f.read())[key]

        except KeyError:
            # print ('# Error: {}: get_json: KeyError: {}'.format(__file__, error))
            pass
        except FileNotFoundError:
            # print ('# Error: {}: get_json: FileNotFoundError: {}'.format(__file__, error))
            pass
        except json.decoder.JSONDecodeError as error:
            print(
                "{} in get_json\n\t# Error: JSONDecodeError: {} #".format(
                    __file__, error
                )
            )


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------


# Deprecated ------------------------------------
# @staticmethod
# def get_filepath(obj, inc_filename=False):
#     """Get the filepath of a class or module.

#     Parameters:
#             obj (obj): A python module, class, or the built-in __file__ variable.
#             inc_filename (bool): Include the filename in the returned result.

#     Returns:
#             (str)
#     """
#     from types import ModuleType

#     if isinstance(obj, type(None)):
#         return ""
#     elif isinstance(obj, str):
#         filepath = obj
#     elif isinstance(obj, ModuleType):
#         filepath = obj.__file__
#     else:
#         clss = obj if callable(obj) else obj.__class__
#         try:
#             import inspect

#             filepath = inspect.getfile(clss)

#         except (
#             TypeError
#         ) as error:  # iterate over each filepath in the call frames, until a class with a matching name is found.
#             import importlib

#             filepath = ""
#             for frame_record in inspect.stack():
#                 if filepath:
#                     break
#                 frame = frame_record[0]
#                 _filepath = inspect.getframeinfo(frame).filename
#                 mod_name = os.path.splitext(os.path.basename(_filepath))[0]
#                 spec = importlib.util.spec_from_file_location(mod_name, _filepath)
#                 if not spec:
#                     continue
#                 mod = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(mod)

#                 for cls_name, clss_ in inspect.getmembers(
#                     mod, inspect.isclass
#                 ):  # get the module's classes.
#                     if cls_name == clss.__name__:
#                         filepath = _filepath
#     if inc_filename:
#         return os.path.abspath(filepath)
#     else:
#         return os.path.abspath(os.path.dirname(filepath))


# def getCallingModuleDir():
#   """Get the directory path of the module that called the function.

#   Returns:
#       (str) The directory path of the calling module.
#   """
#   import os, inspect

#   calling_frame = inspect.currentframe().f_back
#   calling_module = inspect.getmodule(calling_frame)
#   calling_module_path = os.path.abspath(calling_module.__file__)
#   calling_module_dir = os.path.dirname(calling_module_path)

#   return calling_module_dir
