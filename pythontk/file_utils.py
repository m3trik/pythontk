# !/usr/bin/python
# coding=utf-8
import sys
import os
import re
import json
import traceback

# from this package:
from pythontk import core_utils
from pythontk import iter_utils
from pythontk import str_utils


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
    def get_dir_contents(
        dirPath,
        returned_type="file",
        recursive=False,
        num_threads=1,
        inc_files=[],
        exc_files=[],
        inc_dirs=[],
        exc_dirs=[],
        group_by_type=False,
    ):
        """Get the contents of a directory and any of its children.

        Parameters:
            dirPath (str): The path to the directory.
            returned_type (str/list): Return files and directories. Can be a single string or a list of strings.
                                      (valid: 'file'(default), 'filename', 'filepath', 'dir', 'dirpath')
            recursive (bool): When False, return the contents of the root dir only. When True, includes sub-directories.
            num_threads (int): Specifies the number of threads to use for processing directories and files.
                           A value of 0 (default) means no multithreading, -1 means use all available cores.
            inc_files (str/list): Include only specific files.
            exc_files (str/list): Exclude specific files.
            inc_dirs (str/list): Include only specific child directories.
            exc_dirs (str/list): Exclude specific child directories.
            group_by_type (bool): When set to True, returns a dictionary where each key corresponds to a 'returned_type',
                                  and the value is a list of items of that type.
        Returns:
            list/dict: A list or dictionary containing the results based on the `returned_type` and `group_by_type` parameters.

        Examples:
            # Example 1: Basic usage with default `returned_type`
            result = get_dir_contents('/path/to/directory')

            # Example 2: Specifying multiple return types
            result = get_dir_contents('/path/to/directory', returned_type=['filename', 'filepath'])

            # Example 3: Using the `group_by_type` flag
            result = get_dir_contents('/path/to/directory', returned_type=['filename', 'filepath'], group_by_type=True)
            result['filename']  # ['file1', 'file2', ...],
            result['filepath']  # ['/path/to/file1', '/path/to/file2', ...]

        """
        from itertools import chain

        path = os.path.expandvars(dirPath)
        options = iter_utils.IterUtils.make_iterable(returned_type)
        grouped_result = {opt: [] for opt in options}

        def process_directory(root, dirs, files):
            temp_result = {opt: [] for opt in options}

            if not recursive and root != path:
                return temp_result

            dirs = iter_utils.IterUtils.filter_list(dirs, inc_dirs, exc_dirs)
            files = iter_utils.IterUtils.filter_list(files, inc_files, exc_files)

            for opt in options:
                if opt == "dir":
                    temp_result[opt].extend(dirs)
                elif opt == "dirpath":
                    temp_result[opt].extend([os.path.join(root, d) for d in dirs])
                elif opt == "file":
                    temp_result[opt].extend(files)
                elif opt == "filename":
                    temp_result[opt].extend([os.path.splitext(f)[0] for f in files])
                elif opt == "filepath":
                    temp_result[opt].extend([os.path.join(root, f) for f in files])

            return temp_result

        if num_threads > 1:
            import multiprocessing
            from concurrent.futures import ThreadPoolExecutor, as_completed

            num_cores = (
                multiprocessing.cpu_count() if num_threads == -1 else num_threads
            )
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = {
                    executor.submit(process_directory, root, dirs, files): (
                        root,
                        dirs,
                        files,
                    )
                    for root, dirs, files in os.walk(path, topdown=True)
                }

                for future in as_completed(futures):
                    data = future.result()
                    for opt in options:
                        grouped_result[opt].extend(data[opt])
        else:
            for root, dirs, files in os.walk(path, topdown=True):
                data = process_directory(root, dirs, files)
                for opt in options:
                    grouped_result[opt].extend(data[opt])

        return (
            grouped_result
            if group_by_type
            else list(chain.from_iterable(grouped_result.values()))
        )

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

    @classmethod
    def get_file_info(cls, paths, returned_type, hash_algo=None, force_tuples=False):
        """Returns file and directory information for a list of files based on specified parameters.

        This method will traverse each path, obtaining information as per the `returned_type` parameter.

        Parameters:
            paths (str/list): Path(s) to a file or directory.
            returned_type (str/list): A single string or a list of strings containing types of information to be returned.
                 Supported types are as follows:
                 - 'file': Returns the name of the file including extension (if it's a file).
                 - 'filename': Returns the name of the file excluding extension (if it's a file).
                 - 'filepath': Returns the full file path.
                 - 'dir': Returns the parent directory name of the file or directory.
                 - 'dirpath': Returns the full path of the parent directory of the file or directory.
                 - 'timestamp': Returns the last modification timestamp of the file (if it's a file).
                 - 'unixtimestamp': Returns the last modification Unix timestamp of the file (if it's a file).
                 - 'size': Returns the size of the file (if it's a file).
                 - 'filetype': Returns the extension of the file (if it's a file).
                 - 'permissions': Returns the file permissions (if it's a file).
                 - 'owner': Returns the owner's user ID of the file (if it's a file).
                 - 'group': Returns the group ID of the file (if it's a file).
                 - 'hash': Returns the hash of the file using the specified algorithm (if it's a file and hash_algo is provided).
            hash_algo (str, optional): A string specifying the hash algorithm to be used. Supported algorithms are those in Python's hashlib library (e.g., 'md5', 'sha1', 'sha256'). Default is None.
            force_tuples (bool, optional): If True, ensures that the result is always returned as tuples even if only one item is specified in `returned_type`. If False, returns single values as is without wrapping in a tuple. Default is False.

        Returns:
            list: A list of tuples. Each tuple contains requested information in the same order as the types specified
                in `returned_type`. If a type of information is not applicable (for instance, requesting 'size' for a directory),
                its place in the tuple will be None.
        """
        import time
        import hashlib
        from stat import filemode
        from pathlib import Path

        options = iter_utils.IterUtils.make_iterable(returned_type)
        results = []

        for _path in iter_utils.IterUtils.make_iterable(paths):
            path = os.path.expandvars(_path)
            path_obj = Path(path)
            if not path_obj.exists():
                continue

            is_file = path_obj.is_file()
            stats = path_obj.stat() if is_file else None

            def get_hash():
                if is_file and hash_algo:
                    hash_obj = hashlib.new(hash_algo)
                    with open(path, "rb") as file:
                        hash_obj.update(file.read())
                    return hash_obj.hexdigest()
                return None

            info_dict = {
                "file": path_obj.name if is_file else None,
                "filename": path_obj.stem if is_file else None,
                "filepath": str(path_obj) if is_file else None,
                "dir": path_obj.parent.name,
                "dirpath": str(path_obj.parent),
                "timestamp": time.ctime(stats.st_mtime) if is_file else None,
                "unixtimestamp": stats.st_mtime if is_file else None,
                "size": stats.st_size if is_file else None,
                "filetype": path_obj.suffix if is_file else None,
                "permissions": filemode(stats.st_mode) if is_file else None,
                "owner": stats.st_uid if is_file else None,
                "group": stats.st_gid if is_file else None,
                "hash": get_hash(),
            }

            info = [info_dict[option] for option in options]

            if any(x is not None for x in info):
                if len(options) == 1 and not force_tuples:
                    results.append(info[0])
                else:
                    results.append(tuple(info))

        return results

    @staticmethod
    @core_utils.CoreUtils.listify(threading=True)
    def format_path(p, section="", replace=""):
        """Format a given filepath(s).
        When a section arg is given, the correlating section of the string will be returned.
        If a replace arg is given, the stated section will be replaced by the given value.

        Parameters:
            p (str/list): The filepath(s) to be formatted.
            section (str): The desired subsection of the given path.
                 - path: path minus filename,
                 - dir: directory name,
                 - file: filename plus ext,
                 - name: filename minus ext,
                 - ext: file extension,
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
            result = str_utils.StrUtils.rreplace(p, result, replace, 1)

        return result

    @classmethod
    def append_path(cls, path, **kwargs):
        """Append a directory to the python path.

        Parameters:
            path (str): The directory to be appended to the system path.
            kwargs (optional): Any file related keyword arguments that 'get_dir_contents' allows.
                    ie. recursive, num_threads, inc_dirs, exc_dirs. But not: dirPath or returned_type.
        Returns:
            list:  The appended paths.

        Example:
            ptk.append_path(<path>, recursive=True, exc_dirs="_*")
        )
        """
        root_dir = os.path.dirname(os.path.abspath(path))
        appended_paths = []
        for directory in cls.get_dir_contents(root_dir, "dirpath", **kwargs):
            sys.path.append(directory)
            appended_paths.append(directory)
        return appended_paths

    @staticmethod
    def get_object_path(obj, inc_filename=False):
        """Retrieve the absolute file path associated with a Python object.

        This method can take different Python objects such as modules, classes, callable objects, or even
        the built-in __file__ variable, and tries to extract the file path associated with the object.

        Parameters:
            obj (object/str): A Python object. This can be a module, class, callable, built-in __file__ variable, or a string file path.
            inc_filename (bool, optional): Flag to decide whether to include the filename in the returned path. Defaults to False.
                If the object is a package, the package directory is considered as the filename.

        Returns:
            str: The absolute file path associated with the input object. If the input is a string, the absolute path of the string is returned.
                If include_filename is set to False, only the directory containing the file is returned.
                Returns None if the file path cannot be determined.
        """
        from types import ModuleType
        import inspect
        import importlib

        if obj is None:
            return ""

        elif isinstance(obj, str):
            filepath = obj

        elif isinstance(obj, ModuleType):
            if hasattr(obj, "__file__"):
                filepath = obj.__file__
            elif hasattr(obj, "__path__"):
                filepath = obj.__path__[0]
            else:
                filepath = None

        else:
            clss = obj if callable(obj) else obj.__class__
            try:
                filepath = inspect.getfile(clss)
            except TypeError:
                filepath = ""
                for frame_record in inspect.stack():
                    if filepath:
                        break
                    frame = frame_record[0]
                    _filepath = inspect.getframeinfo(frame).filename
                    mod_name = os.path.splitext(os.path.basename(_filepath))[0]
                    spec = importlib.util.spec_from_file_location(mod_name, _filepath)
                    if spec:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        filepath = next(
                            (
                                _filepath
                                for cls_name, _ in inspect.getmembers(
                                    mod, inspect.isclass
                                )
                                if cls_name == clss.__name__
                            ),
                            "",
                        )

        if filepath is None:
            raise ValueError(
                f"Unable to determine the file path for the given object: {obj}"
            )

        if not inc_filename:
            filepath = os.path.dirname(filepath)

        return os.path.abspath(filepath)

    @classmethod
    def get_classes_from_path(
        cls,
        path,
        returned_type=["classname", "filepath"],
        inc=[],
        exc=[],
        top_level_only=True,
        force_tuples=False,
    ):
        """Scans the specified directory or Python file, loads each file as a module, and retrieves classes from these modules.

        Parameters:
            path (str): The path to the directory or Python file to scan for classes.
            returned_type (str/list): A single string or a list of strings representing the type of information to return.
                 Supported options are:
                 - classname: Returns the name of the class.
                 - classobj: Returns the class object.
                 - file: Returns the name of the file including extension (if it's a file).
                 - filename: Returns the name of the file excluding extension (if it's a file).
                 - filepath: Returns the file path of the Python file where the class is defined.
                 - module: Returns the module object where the class is defined.
            inc (list, optional): A list of class names to include in the results.
            exc (list, optional): A list of class names to exclude from the results.
            top_level_only (bool, optional): If True, only retrieves top-level classes. If False, retrieves all classes within the specified path. Default is True.
            force_tuples (bool, optional): If True, ensures that the result is always returned as tuples even if only one item is specified in `returned_type`. If False, returns single values as is without wrapping in a tuple. Default is False.

        Returns:
            list: A list of tuples, where each tuple contains information about a class found in the Python files in the directory or Python file.
            The types of information in each tuple are determined by the `returned_type` parameter. If `force_tuples` is False and only one item is specified in `returned_type`, the results may be returned as single values instead of tuples.

        Raises:
            FileNotFoundError: If the provided path does not exist or is not a directory or Python file.
            ValueError: If an invalid option is provided in `returned_type`.
        """
        import ast
        import importlib.util

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} doesn't exist")
        elif os.path.isfile(path) and not path.endswith(".py"):
            return []
        elif os.path.isdir(path):
            filenames = [fn for fn in os.listdir(path) if fn.endswith(".py")]
        else:
            filenames = [os.path.basename(path)]
            path = os.path.dirname(path)

        options = iter_utils.IterUtils.make_iterable(returned_type)
        results = []

        valid_options = {
            "file",
            "filename",
            "filepath",
            "classobj",
            "classname",
            "module",
        }
        if not all(option in valid_options for option in options):
            raise ValueError(
                f"Invalid option in returned_type. Valid options are {', '.join(valid_options)}, got {options}"
            )

        for filename in filenames:
            filepath = os.path.join(path, filename)

            with open(filepath, "r") as file:
                module_ast = ast.parse(file.read())
                if top_level_only:
                    classes = [
                        node
                        for node in module_ast.body
                        if isinstance(node, ast.ClassDef)
                    ]
                else:
                    classes = [
                        node
                        for node in ast.walk(module_ast)
                        if isinstance(node, ast.ClassDef)
                    ]

            module_name = filename.rstrip(".py")
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module_obj = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module_obj
            try:
                spec.loader.exec_module(module_obj)
            except Exception as e:
                raise RuntimeError(
                    f"The following error occurred while loading the module {module_name} from {filepath}: {e}"
                ) from e

            for clss in classes:
                info = {
                    "file": filename,
                    "filename": module_name,
                    "filepath": filepath,
                    "classobj": module_obj.__dict__.get(clss.name),
                    "classname": clss.name,
                    "module": module_obj,
                }
                if len(options) == 1 and not force_tuples:
                    results.append(info[options[0]])
                else:
                    results.append(tuple(info[option] for option in options))

        if inc or exc:
            results = iter_utils.IterUtils.filter_list(
                results, inc=inc, exc=exc, nested_as_unit=True
            )
        return results

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

    @classmethod
    def update_version(
        cls,
        filepath: str,
        change: str = "increment",
        version_part: str = "patch",
        max_version_parts: tuple = (99, 99),
        version_regex: str = r"__version__\s*=\s*['\"](\d+)\.(\d+)\.(\d+)['\"]",
    ) -> None:
        """This function updates the version number in a text file depending on its state.
        The version number is represented as a string in the format 'x.y.z', where x, y, and z are integers and it matches the provided regex pattern.

        Parameters:
            filepath (str): The path to the text file containing the version number.
            change (str, optional): The type of change, either 'increment' or 'decrement'. Defaults to 'increment'.
            version_part (str, optional): The part of the version number to update, either 'major', 'minor', or 'patch'. Defaults to 'patch'.
            max_version_parts (tuple, optional): A tuple containing the maximum values for the minor and patch version parts. Defaults to (9, 9).
            version_regex (str, optional): A regex pattern that defines the format of the version line in the file.
                    The pattern should have three groups each representing major, minor, and patch versions respectively.
        Returns:
            str: The new version number. If the function could not find a version number that matches the provided pattern in the file, it will print an error message and return an empty string.
        """
        import re

        lines = cls.get_file_contents(filepath, as_list=True)

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

        cls.write_to_file(filepath, lines)
        if not version:
            print(f"Error: No version found in {filepath}")
        return version

    @staticmethod
    def update_requirements(file_path=None):
        """Update the requirements.txt file with the current versions of packages.

        Parameters:
            file_path (str): Path to the requirements.txt file. Defaults to the caller's directory.
                             If a relative path is given, it's relative to the caller's directory.
        """
        import inspect
        import pkg_resources

        # Determine the caller's directory
        caller_frame = inspect.stack()[1]
        caller_path = caller_frame.filename
        caller_dir = os.path.dirname(caller_path)

        if file_path is None:
            file_path = os.path.join(caller_dir, "requirements.txt")
        else:
            # Resolve relative paths relative to the caller's directory
            file_path = os.path.abspath(os.path.join(caller_dir, file_path))

        required_packages = []

        try:
            # Read the existing requirements.txt
            with open(file_path, "r") as file:
                lines = file.readlines()

            for line in lines:
                # Ignore empty lines or comments
                if not line.strip() or line.startswith("#"):
                    continue

                # Extract package name
                package_name = line.strip().split("==")[0]

                try:
                    # Get the current version of the package
                    version = pkg_resources.get_distribution(package_name).version
                    required_packages.append(f"{package_name}=={version}")
                except Exception as e:
                    print(f"Error updating version for {package_name}: {e}")

        except FileNotFoundError:
            print(f"File not found: {file_path}")

        return required_packages


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
