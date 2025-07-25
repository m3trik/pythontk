# !/usr/bin/python
# coding=utf-8
import sys
import os
import re
import json
import traceback
from typing import Union, List, Dict, Tuple, Optional, Any

# From this package:
from pythontk import core_utils
from pythontk import iter_utils
from pythontk import str_utils


class FileUtils(core_utils.HelpMixin):
    """ """

    @staticmethod
    def is_valid(filepath: str, expected_type: Optional[str] = None) -> bool:
        """Check if a path is valid, optionally requiring a specific type ('file' or 'dir').

        Parameters:
            filepath (str): Path to check.
            expected_type (str, optional): Required type ('file' or 'dir').

        Returns:
            bool: True if valid (and matches type if given), False otherwise.
        """
        fp = os.path.expandvars(filepath)

        if expected_type == "file":
            return os.path.isfile(fp)
        elif expected_type == "dir":
            return os.path.isdir(fp)
        else:
            return os.path.exists(fp)

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
        content="file",
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
            content (str/list): Return files and directories. Can be a single string or a list of strings.
                                      (valid: 'file'(default), 'filename', 'filepath', 'dir', 'dirpath')
            recursive (bool): When False, return the contents of the root dir only. When True, includes sub-directories.
            num_threads (int): Specifies the number of threads to use for processing directories and files.
                           A value of 0 (default) means no multithreading, -1 means use all available cores.
            inc_files (str/list): Include only specific files.
            exc_files (str/list): Exclude specific files.
            inc_dirs (str/list): Include only specific child directories.
            exc_dirs (str/list): Exclude specific child directories.
            group_by_type (bool): When set to True, returns a dictionary where each key corresponds to a 'content',
                                  and the value is a list of items of that type.
        Returns:
            list/dict: A list or dictionary containing the results based on the `content` and `group_by_type` parameters.

        Examples:
            # Example 1: Basic usage with default `content`
            result = get_dir_contents('/path/to/directory')

            # Example 2: Specifying multiple return types
            result = get_dir_contents('/path/to/directory', content=['filename', 'filepath'])

            # Example 3: Using the `group_by_type` flag
            result = get_dir_contents('/path/to/directory', content=['filename', 'filepath'], group_by_type=True)
            result['filename']  # ['file1', 'file2', ...],
            result['filepath']  # ['/path/to/file1', '/path/to/file2', ...]

        """
        from itertools import chain

        path = os.path.expandvars(dirPath)
        options = iter_utils.IterUtils.make_iterable(content)
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

    @staticmethod
    def copy_file(
        file_path: str,
        destination: str,
        new_name: Optional[str] = None,
        overwrite: bool = True,
        create_dir: bool = True,
    ) -> str:
        """Copies a file to a specified folder, ensuring the folder exists.

        Parameters:
            file_path (str): Path to the file to be copied.
            destination (str): Target directory.
            new_name (str, optional): New name for the copied file.
            overwrite (bool, optional): Allow overwriting an existing file.
            create_dir (bool, optional): Auto-create destination dir.

        Returns:
            str: Path to the copied file.
        """
        import shutil

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if create_dir:
            os.makedirs(destination, exist_ok=True)

        file_name = new_name or os.path.basename(file_path)
        destination_path = os.path.join(destination, file_name)

        if os.path.exists(destination_path):
            if not overwrite:
                raise FileExistsError(f"File already exists: {destination_path}")
            os.remove(destination_path)

        shutil.copy2(file_path, destination_path)
        return destination_path

    @classmethod
    def move_file(
        cls,
        file_path: Union[str, List[Union[str, Tuple[str, str]]]],
        destination: str,
        new_name: Optional[str] = None,
        overwrite: bool = True,
        create_dir: bool = True,
        verbose: bool = False,
    ) -> Union[str, List[str]]:
        """Moves one or more files to a specified folder.

        Parameters:
            file_path (str | list): File path or list of file paths (or (dir, filename) tuples).
            destination (str): Folder to move files into.
            new_name (str, optional): Rename a single file. Ignored when moving multiple files.
            overwrite (bool): Whether to overwrite existing files.
            create_dir (bool): Create the destination folder if it doesn't exist.
            verbose (bool): Print each move result.

        Returns:
            str | list: New file path(s).
        """
        import shutil

        file_paths = iter_utils.IterUtils.make_iterable(file_path)
        results = []

        if create_dir:
            os.makedirs(destination, exist_ok=True)

        for entry in file_paths:
            if isinstance(entry, tuple):
                dir_path, filename = entry
                src_path = os.path.join(dir_path, filename).replace("\\", "/")
            else:
                src_path = entry.replace("\\", "/")

            if not os.path.exists(src_path):
                raise FileNotFoundError(f"File not found: {src_path}")

            name = (
                new_name
                if isinstance(file_path, str) and new_name
                else os.path.basename(src_path)
            )
            dst_path = os.path.join(destination, name)

            if os.path.exists(dst_path):
                if overwrite:
                    os.remove(dst_path)
                else:
                    raise FileExistsError(f"File already exists: {dst_path}")

            shutil.move(src_path, dst_path)
            dst_path = dst_path.replace("\\", "/")

            if verbose:
                print(f"Moved: {src_path} -> {dst_path}")

            results.append(dst_path)

        return (
            results[0]
            if isinstance(file_path, str) and not isinstance(file_path, (list, tuple))
            else results
        )

    @classmethod
    def get_file_info(cls, paths, info, hash_algo=None, force_tuples=False):
        """Returns file and directory information for a list of file strings based on specified parameters.

        This method will traverse each path, obtaining information as per the `info` parameter.

        Parameters:
            paths (str/list): Path(s) to a file or directory.
            info (str/list): A single string or a list of strings containing types of information to be returned.
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
            force_tuples (bool, optional): If True, ensures that the result is always returned as tuples even if only one item is specified in `info`. If False, returns single values as is without wrapping in a tuple. Default is False.

        Returns:
            list: A list of tuples. Each tuple contains requested information in the same order as the types specified
                in `info`. If a type of information is not applicable (for instance, requesting 'size' for a directory),
                its place in the tuple will be None.
        """
        import time
        import hashlib
        from stat import filemode
        from pathlib import Path

        options = iter_utils.IterUtils.make_iterable(info)
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

    def get_metadata(file_path: str, *keys: str) -> Dict[str, Optional[str]]:
        """Retrieves metadata from a specified file in Windows.

        Parameters:
            file_path (str): The full path to the file from which metadata will be retrieved.
            *keys (str): The metadata keys to retrieve.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Returns:
            Dict[str, Optional[str]]: A dictionary containing the requested metadata key-value pairs.

        Example:
            metadata = get_metadata(
                "C:\\path\\to\\your\\file.txt",
                "Custom_PlaneName",
                "Custom_ModuleType",
                "Comments"
            )
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        import win32com.client

        shell = win32com.client.Dispatch("Shell.Application")
        ns = shell.NameSpace(os.path.dirname(file_path))
        item = ns.ParseName(os.path.basename(file_path))

        metadata = {}
        for key in keys:
            metadata[key] = item.ExtendedProperty(key)

        return metadata

    def set_metadata(file_path: str, **metadata: Dict[str, Optional[str]]) -> None:
        """Attaches or modifies metadata for a specified file in Windows.

        Parameters:
            file_path (str): The full path to the file to which metadata will be attached or modified.
            **metadata (Dict[str, Optional[str]]): Key-value pairs representing the metadata properties and their values.
                If a value is None, the corresponding metadata property will be removed.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Example:
            set_metadata(
                "C:\\path\\to\\your\\file.txt",
                Custom_PlaneName="Boeing 747",
                Custom_ModuleType="Engine",
                Comments="This is a sample comment."
            )
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        import win32com.client

        shell = win32com.client.Dispatch("Shell.Application")
        ns = shell.NameSpace(os.path.dirname(file_path))
        item = ns.ParseName(os.path.basename(file_path))

        for key, value in metadata.items():
            if value is None:  # Clear the metadata property
                item.ExtendedProperty(key, "")
            else:
                item.ExtendedProperty(key, value)

    @staticmethod
    @core_utils.CoreUtils.listify(threading=True)
    def format_path(
        p: Union[str, List[str]],
        section: Union[str, None] = None,
        replace: Union[str, None] = None,
    ) -> Union[str, List[str]]:
        """Format a given filepath(s).
        When a section arg is given, the correlating section of the string will be returned.
        If a replace arg is given, the stated section will be replaced by the given value.

        Parameters:
            p (str/list): The filepath(s) to be formatted.
            section (str, optional): The desired subsection of the given path.
                 - path: path minus filename,
                 - dir: directory name,
                 - file: filename plus ext,
                 - name: filename minus ext,
                 - ext: file extension,
                    (if None is given, the fullpath will be returned)
            replace (str, optional): The value to replace the section with.

        Returns:
            (str/list) List if 'strings' given as list.

        Raises:
            ValueError: If the section is not valid.
        """
        valid_sections = {"path", "dir", "file", "name", "ext", None}
        if section not in valid_sections:
            raise ValueError(
                f"Invalid section: {section}. Valid options are: {', '.join(filter(None, valid_sections))}"
            )

        if not isinstance(p, str):
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

        result = {
            "path": path,
            "dir": directory,
            "file": filename,
            "name": name,
            "ext": ext,
        }.get(section, p)

        if replace:
            result = str_utils.StrUtils.rreplace(p, result, replace, 1)

        return result

    @staticmethod
    @core_utils.CoreUtils.listify(threading=True)
    def convert_to_relative_path(
        file_path: str,
        base_dir: str,
        prepend_base: bool = True,
        check_existence: bool = False,
    ) -> str:
        """Convert an absolute file path to a relative path based on the given base directory.

        If the file path and the base directory are on different drives,
        the file path's drive letter is changed to match the base directory's drive letter.

        Parameters:
            file_path (str): The absolute file path to convert.
            base_dir (str): The base directory to which the path should be made relative.
            prepend_base (bool): Whether to prepend the base directory to the relative path.
            check_existence (bool): Whether to check the existence of the base directory.

        Returns:
            str: The relative file path.

        Raises:
            FileNotFoundError: If check_existence is True and the base directory does not exist.
        """
        if check_existence and not os.path.exists(base_dir):
            raise FileNotFoundError(f"The base directory does not exist: {base_dir}")

        # Get drive letters
        file_drive, file_path_without_drive = os.path.splitdrive(file_path)
        base_drive, base_path_without_drive = os.path.splitdrive(base_dir)

        # If the drives are different, keep only the file name
        if file_drive != base_drive:
            relative_path = os.path.basename(file_path)
        else:
            # Calculate the relative path
            relative_path = os.path.relpath(file_path, base_dir)

        if prepend_base:
            return os.path.join(os.path.basename(base_dir), relative_path).replace(
                "\\", "/"
            )

        return relative_path.replace("\\", "/")

    @staticmethod
    def remap_file_paths(
        source_paths: List[str], target_dir: str, base_dir: str
    ) -> List[Tuple[str, str, str]]:
        """
        Remap a list of file paths to a new directory while preserving their relative
        structure with respect to a base directory.

        Parameters:
            source_paths (List[str]): List of original file paths.
            target_dir (str): Target root directory to remap files into.
            base_dir (str): Directory to compute relative paths from.

        Returns:
            List[Tuple[str, str, str]]: List of tuples:
                (lookup_key, new_full_path, relative_path_or_mapped_path)
                - lookup_key: Lowercased relative path (if under base_dir) or filename
                - new_full_path: Full remapped file path under the target_dir
                - relative_path_or_mapped_path: Relative path for Maya or similar use
        """
        results = []

        base_dir_norm = os.path.normpath(base_dir).replace("\\", "/")
        base_dir_name = os.path.basename(base_dir_norm)

        for original_path in source_paths:
            original_norm = os.path.normpath(original_path).replace("\\", "/")

            if original_norm.lower().startswith(base_dir_norm.lower()):
                rel_path = os.path.relpath(original_norm, base_dir_norm).replace(
                    "\\", "/"
                )
                lookup_key = rel_path.lower()
                new_path = os.path.join(target_dir, rel_path).replace("\\", "/")
            else:
                filename = os.path.basename(original_norm)
                lookup_key = filename.lower()
                new_path = os.path.join(target_dir, filename).replace("\\", "/")

            new_path_norm = os.path.normpath(new_path).replace("\\", "/")

            if new_path_norm.lower().startswith(base_dir_norm.lower()):
                rel_path = os.path.relpath(new_path_norm, base_dir_norm).replace(
                    "\\", "/"
                )
                mapped_path = f"{base_dir_name}/{rel_path}"
            else:
                mapped_path = new_path_norm

            results.append((lookup_key, new_path_norm, mapped_path))

        return results

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
    def get_object_path(obj, inc_filename: bool = False) -> str:
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
        Raises:
            ValueError: If the file path for the given object cannot be determined.
        """
        from types import ModuleType
        import os
        import inspect
        import importlib

        def fail_msg(obj):
            return ValueError(
                f"\nUnable to determine the file path for the given object: {obj} Type: {type(obj)}"
                f"\nPerhaps the object is not a module, class, function, or was run from an interactive source."
            )

        if obj is None:
            filepath = None

        elif isinstance(obj, str):
            filepath = obj

        elif isinstance(obj, ModuleType):
            if hasattr(obj, "__file__"):
                filepath = obj.__file__
            elif hasattr(obj, "__path__"):
                filepath = obj.__path__[0]
            else:
                filepath = None

        else:  # Handle class or function objects
            try:
                clss = obj if callable(obj) else obj.__class__
            except AttributeError:
                raise fail_msg(obj)

            # Attempt to get the module directly
            mod_name = clss.__module__

            # If module is in sys.modules, retrieve file
            if mod_name in sys.modules:
                module = sys.modules[mod_name]
                filepath = getattr(module, "__file__", None)

            # Handle __main__ case
            if filepath is None and mod_name == "__main__":
                main_file = getattr(sys.modules["__main__"], "__file__", None)
                if main_file:
                    filepath = main_file

            # If we still don't have a filepath, scan the stack trace
            if not filepath:
                for frame_record in inspect.stack():
                    frame = frame_record[0]
                    _filepath = inspect.getframeinfo(frame).filename
                    mod_name = os.path.splitext(os.path.basename(_filepath))[0]

                    # Ignore interactive execution sources
                    if (
                        not _filepath
                        or _filepath.startswith("<")
                        and _filepath.endswith(">")
                    ):
                        continue

                    if _filepath in sys.modules:
                        mod = sys.modules[_filepath]
                    else:
                        spec = importlib.util.spec_from_file_location(
                            mod_name, _filepath
                        )
                        if not spec or not spec.loader:
                            continue

                        try:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            sys.modules[_filepath] = mod  # Cache it for consistency
                        except Exception as e:
                            continue

                    # Try to find the class in the loaded module
                    class_members = inspect.getmembers(mod, inspect.isclass)
                    if clss.__name__ in [cls_name for cls_name, _ in class_members]:
                        filepath = _filepath
                        break

        if filepath is None:
            raise fail_msg(obj)

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
            top_level_only (bool, optional): If True, only retrieves top-level classes. If False, retrieves all classes within the specified path.
            force_tuples (bool, optional): If True, ensures that the result is always returned as tuples even if only one item is specified in `returned_type`.

        Returns:
            list: A list of tuples, where each tuple contains information about a class found in the Python files in the directory or Python file.

        Raises:
            FileNotFoundError: If the provided path does not exist or is not a directory or Python file.
            ValueError: If an invalid option is provided in `returned_type`.
        """
        import ast
        import importlib.util
        import sys
        from pathlib import Path

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

            module_name = Path(filename).stem
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
    def update_requirements(file_path=None, inc=None, exc=None):
        """Update the requirements.txt file with the current versions of packages.

        Parameters:
            file_path (str): Path to the requirements.txt file. Defaults to the caller's directory.
                             If a relative path is given, it's relative to the caller's directory.
            inc (str/list, optional): Patterns or objects to include in the update.
            exc (str/list, optional): Patterns or objects to exclude from the update.

        Returns:
            list: A list of updated requirements with their versions.
            example: ['package1==1.0.0', 'package2==2.3.4']
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
            file_path = os.path.abspath(os.path.join(caller_dir, file_path))

        try:
            with open(file_path, "r") as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    package_name = line.strip().split("==")[0]
                    if package_name in iter_utils.IterUtils.filter_list(
                        [package_name], inc, exc
                    ):
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


# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
