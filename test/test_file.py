#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk FileUtils.

Comprehensive edge case coverage for:
- format_path
- is_valid
- write_to_file / get_file_contents
- create_dir
- get_file_info
- get_dir_contents
- get_object_path
- JSON operations
- Path edge cases (unicode, special chars, etc.)

Run with:
    python -m pytest test_file.py -v
    python test_file.py
"""
import os
import tempfile
import unittest
from pathlib import Path

from pythontk import FileUtils

from conftest import BaseTestCase, TestPaths


class FileTest(BaseTestCase):
    """File utilities test class with comprehensive edge case coverage."""

    @classmethod
    def setUpClass(cls):
        """Set up test paths used across file tests."""
        cls.test_base_path = TestPaths.BASE_DIR
        cls.test_files_path = TestPaths.TEST_FILES_DIR
        cls.file1_path = cls.test_files_path / "file1.txt"
        cls.file2_path = cls.test_files_path / "file2.txt"

    # -------------------------------------------------------------------------
    # format_path Tests
    # -------------------------------------------------------------------------

    def test_format_path_basic_normalization(self):
        """Test format_path normalizes path separators."""
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3"), "X:/n/dir1/dir3")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3", "path"), "X:/n/dir1/dir3"
        )

    def test_format_path_vscode_directory(self):
        """Test format_path with .vscode directory."""
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "path"),
            "X:/n/dir1/dir3/.vscode",
        )
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "path"),
            "X:/n/dir1/dir3/.vscode",
        )

    def test_format_path_unc_path(self):
        """Test format_path with UNC paths."""
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "path"),
            r"\\192.168.1.240/nas/lost+found",
        )

    def test_format_path_environment_variable(self):
        """Test format_path expands environment variables."""
        self.assertEqual(
            FileUtils.format_path(r"%programfiles%", "path"), "C:/Program Files"
        )

    def test_format_path_dir_extraction(self):
        """Test format_path extracts directory names."""
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "dir"), "dir3")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "dir"), ".vscode"
        )
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "dir"),
            ".vscode",
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "dir"),
            "lost+found",
        )
        self.assertEqual(
            FileUtils.format_path(r"%programfiles%", "dir"), "Program Files"
        )

    def test_format_path_file_extraction(self):
        """Test format_path extracts file names."""
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "file"), "")
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "file"), "")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "file"),
            "tasks.json",
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "file"),
            "file.ext",
        )
        self.assertEqual(FileUtils.format_path(r"%programfiles%", "file"), "")

    def test_format_path_name_extraction(self):
        """Test format_path extracts file names without extension."""
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "name"), "")
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "name"), "")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "name"), "tasks"
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "name"),
            "file",
        )
        self.assertEqual(FileUtils.format_path(r"%programfiles%", "name"), "")

    def test_format_path_ext_extraction(self):
        """Test format_path extracts extensions."""
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "ext"), "")
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "ext"), "")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "ext"), "json"
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "ext"),
            "ext",
        )
        self.assertEqual(FileUtils.format_path(r"%programfiles%", "ext"), "")

    def test_format_path_edge_cases(self):
        """Test format_path edge cases."""
        self.assertEqual(FileUtils.format_path(r"programfiles", "name"), "programfiles")
        self.assertEqual(FileUtils.format_path(r"programfiles", "path"), "programfiles")
        self.assertEqual(
            FileUtils.format_path(r"programfiles/", "path"), "programfiles"
        )

    def test_format_path_list_input(self):
        """Test format_path with list input."""
        self.assertEqual(
            FileUtils.format_path(
                [r"X:\n/dir1/dir3", r"X:\n/dir1/dir3/.vscode"], "dir"
            ),
            ["dir3", ".vscode"],
        )

    def test_format_path_empty_string(self):
        """Test format_path with empty string."""
        result = FileUtils.format_path("", "path")
        self.assertEqual(result, "")

    def test_format_path_trailing_separator(self):
        """Test format_path with trailing separators."""
        result = FileUtils.format_path("C:/test/path/", "dir")
        self.assertEqual(result, "path")

    def test_format_path_multiple_extensions(self):
        """Test format_path with multiple extensions."""
        result = FileUtils.format_path("file.tar.gz", "ext")
        # Should return last extension
        self.assertEqual(result, "gz")

    def test_format_path_hidden_files(self):
        """Test format_path with hidden files - files starting with . are NOT recognized as files."""
        # Hidden files starting with '.' are treated as directories, not files
        result = FileUtils.format_path("/path/to/.gitignore", "file")
        self.assertEqual(
            result, ""
        )  # Empty because .gitignore is not considered a file
        # Use 'dir' to get the hidden file name instead
        result_dir = FileUtils.format_path("/path/to/.gitignore", "dir")
        self.assertEqual(result_dir, ".gitignore")

    # -------------------------------------------------------------------------
    # is_valid Tests
    # -------------------------------------------------------------------------

    def test_is_valid_file(self):
        """Test is_valid checks file existence."""
        self.assertTrue(FileUtils.is_valid(str(self.file1_path), "file"))

    def test_is_valid_directory(self):
        """Test is_valid checks directory existence."""
        self.assertTrue(FileUtils.is_valid(str(self.test_files_path), "dir"))

    def test_is_valid_nonexistent_file(self):
        """Test is_valid returns False for nonexistent file."""
        self.assertFalse(FileUtils.is_valid("/nonexistent/file.txt", "file"))

    def test_is_valid_nonexistent_directory(self):
        """Test is_valid returns False for nonexistent directory."""
        self.assertFalse(FileUtils.is_valid("/nonexistent/directory", "dir"))

    def test_is_valid_empty_path(self):
        """Test is_valid with empty path."""
        self.assertFalse(FileUtils.is_valid("", "file"))

    def test_is_valid_file_as_dir(self):
        """Test is_valid returns False when file checked as dir."""
        self.assertFalse(FileUtils.is_valid(str(self.file1_path), "dir"))

    def test_is_valid_dir_as_file(self):
        """Test is_valid returns False when dir checked as file."""
        self.assertFalse(FileUtils.is_valid(str(self.test_files_path), "file"))

    # -------------------------------------------------------------------------
    # write_to_file / get_file_contents Tests
    # -------------------------------------------------------------------------

    def test_write_to_file_basic(self):
        """Test write_to_file writes content correctly."""
        result = FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')
        self.assertIsNone(result)

    def test_get_file_contents_as_list(self):
        """Test get_file_contents reads file content as list."""
        FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')
        content = FileUtils.get_file_contents(str(self.file1_path), as_list=True)
        self.assertEqual(content, ['__version__ = "0.9.0"'])

    def test_write_and_read_multiline(self):
        """Test write and read with multiline content - newlines are preserved."""
        multiline = "line1\nline2\nline3"
        FileUtils.write_to_file(str(self.file1_path), multiline)
        content = FileUtils.get_file_contents(str(self.file1_path), as_list=True)
        # readlines() preserves newlines except on last line
        self.assertEqual(content, ["line1\n", "line2\n", "line3"])

    def test_write_and_read_empty(self):
        """Test write and read empty content."""
        FileUtils.write_to_file(str(self.file1_path), "")
        content = FileUtils.get_file_contents(str(self.file1_path), as_list=True)
        self.assertEqual(content, [])

    def test_write_and_read_unicode(self):
        """Test write and read with unicode content."""
        unicode_content = "日本語テスト αβγδ ñoño"
        FileUtils.write_to_file(str(self.file1_path), unicode_content)
        content = FileUtils.get_file_contents(str(self.file1_path), as_list=True)
        self.assertEqual(content, [unicode_content])

    def test_write_and_read_special_chars(self):
        """Test write and read with special characters."""
        special = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        FileUtils.write_to_file(str(self.file1_path), special)
        content = FileUtils.get_file_contents(str(self.file1_path), as_list=True)
        self.assertEqual(content, [special])

    # -------------------------------------------------------------------------
    # create_dir Tests
    # -------------------------------------------------------------------------

    def test_create_directory(self):
        """Test create_dir creates directories."""
        sub_dir = str(self.test_files_path / "sub-directory")
        result = FileUtils.create_dir(sub_dir)
        self.assertIsNone(result)
        self.assertTrue(os.path.isdir(sub_dir))

    def test_create_nested_directory(self):
        """Test create_dir creates nested directories."""
        nested = str(self.test_files_path / "a" / "b" / "c")
        FileUtils.create_dir(nested)
        self.assertTrue(os.path.isdir(nested))
        # Cleanup
        os.rmdir(nested)
        os.rmdir(str(self.test_files_path / "a" / "b"))
        os.rmdir(str(self.test_files_path / "a"))

    def test_create_existing_directory(self):
        """Test create_dir on existing directory is idempotent."""
        existing = str(self.test_files_path)
        result = FileUtils.create_dir(existing)
        self.assertIsNone(result)
        self.assertTrue(os.path.isdir(existing))

    # -------------------------------------------------------------------------
    # get_file_info Tests
    # -------------------------------------------------------------------------

    def test_get_file_info_basic(self):
        """Test get_file_info extracts file metadata."""
        files = [str(self.file1_path), str(self.file2_path)]

        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "filename", "filepath"]),
            [
                ("file1.txt", "file1", str(self.file1_path)),
                ("file2.txt", "file2", str(self.file2_path)),
            ],
        )

    def test_get_file_info_file_and_type(self):
        """Test get_file_info with file and filetype."""
        files = [str(self.file1_path), str(self.file2_path)]
        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "filetype"]),
            [("file1.txt", ".txt"), ("file2.txt", ".txt")],
        )

    def test_get_file_info_filename_and_type(self):
        """Test get_file_info with filename and filetype."""
        files = [str(self.file1_path), str(self.file2_path)]
        self.assertEqual(
            FileUtils.get_file_info(files, ["filename", "filetype"]),
            [("file1", ".txt"), ("file2", ".txt")],
        )

    def test_get_file_info_with_size(self):
        """Test get_file_info with file size."""
        files = [str(self.file1_path), str(self.file2_path)]
        result = FileUtils.get_file_info(files, ["file", "size"])
        self.assertEqual(
            result,
            [
                ("file1.txt", os.path.getsize(str(self.file1_path))),
                ("file2.txt", os.path.getsize(str(self.file2_path))),
            ],
        )

    def test_get_file_info_single_file(self):
        """Test get_file_info with single info item returns value directly."""
        result = FileUtils.get_file_info([str(self.file1_path)], ["filename"])
        # Single info item returns value directly, not tuple (unless force_tuples=True)
        self.assertEqual(result, ["file1"])

    def test_get_file_info_empty_list(self):
        """Test get_file_info with empty list."""
        result = FileUtils.get_file_info([], ["filename"])
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # get_dir_contents Tests
    # -------------------------------------------------------------------------

    def test_get_dir_contents_dirpath(self):
        """Test get_dir_contents returns directory paths."""
        path = str(self.test_files_path)
        base_path = str(self.test_base_path)
        imgtk_test_dirpath = os.path.join(base_path, "test_files\\imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files\\sub-directory")
        self.assertEqual(
            FileUtils.get_dir_contents(path, "dirpath"),
            [imgtk_test_dirpath, sub_directory_dirpath],
        )

    def test_get_dir_contents_filenames_recursive(self):
        """Test get_dir_contents returns filenames recursively."""
        path = str(self.test_files_path)
        result = FileUtils.get_dir_contents(path, "filename", recursive=True)
        self.assertEqual(
            result,
            [
                "file1",
                "file2",
                "test",
                "im_Base_color",
                "im_h",
                "im_Height",
                "im_Metallic",
                "im_Mixed_AO",
                "im_n",
                "im_Normal_DirectX",
                "im_Normal_OpenGL",
                "im_Roughness",
            ],
        )

    def test_get_dir_contents_file_and_dir(self):
        """Test get_dir_contents returns both files and dirs."""
        path = str(self.test_files_path)
        self.assertEqual(
            sorted(FileUtils.get_dir_contents(path, ["file", "dir"])),
            sorted(
                [
                    "imgtk_test",
                    "sub-directory",
                    "file1.txt",
                    "file2.txt",
                    "test.json",
                ]
            ),
        )

    def test_get_dir_contents_exc_dirs(self):
        """Test get_dir_contents with excluded directories."""
        path = str(self.test_files_path)
        self.assertEqual(
            sorted(
                FileUtils.get_dir_contents(path, ["file", "dir"], exc_dirs=["sub*"])
            ),
            sorted(["imgtk_test", "file1.txt", "file2.txt", "test.json"]),
        )

    def test_get_dir_contents_inc_files(self):
        """Test get_dir_contents with included files filter."""
        path = str(self.test_files_path)
        self.assertEqual(
            FileUtils.get_dir_contents(path, "filename", inc_files="*.txt"),
            ["file1", "file2"],
        )
        self.assertEqual(
            FileUtils.get_dir_contents(path, "file", inc_files="*.txt"),
            ["file1.txt", "file2.txt"],
        )

    def test_get_dir_contents_dirpath_and_dir(self):
        """Test get_dir_contents with both dirpath and dir types."""
        path = str(self.test_files_path)
        base_path = str(self.test_base_path)
        imgtk_test_dirpath = os.path.join(base_path, "test_files\\imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files\\sub-directory")
        self.assertEqual(
            sorted(FileUtils.get_dir_contents(path, ["dirpath", "dir"])),
            [
                imgtk_test_dirpath,
                sub_directory_dirpath,
                "imgtk_test",
                "sub-directory",
            ],
        )

    def test_get_dir_contents_group_by_type(self):
        """Test get_dir_contents with group_by_type functionality."""
        path = str(self.test_files_path)
        base_path = str(self.test_base_path)
        imgtk_test_dirpath = os.path.join(base_path, "test_files\\imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files\\sub-directory")
        result = FileUtils.get_dir_contents(
            path, ["dirpath", "file"], group_by_type=True
        )
        self.assertIsInstance(result, dict)
        self.assertIn("dirpath", result)
        self.assertIn("file", result)
        self.assertEqual(
            sorted(result["dirpath"]),
            sorted([imgtk_test_dirpath, sub_directory_dirpath]),
        )
        self.assertEqual(
            sorted(result["file"]),
            sorted(["file1.txt", "file2.txt", "test.json"]),
        )

    def test_get_dir_contents_empty_directory(self):
        """Test get_dir_contents on empty directory."""
        empty_dir = str(self.test_files_path / "sub-directory")
        FileUtils.create_dir(empty_dir)
        result = FileUtils.get_dir_contents(empty_dir, "file")
        self.assertEqual(result, [])

    def test_get_dir_contents_nonexistent(self):
        """Test get_dir_contents on nonexistent directory."""
        result = FileUtils.get_dir_contents("/nonexistent/path", "file")
        self.assertEqual(result, [])

    # -------------------------------------------------------------------------
    # get_object_path Tests
    # -------------------------------------------------------------------------

    def test_get_object_path_from_file(self):
        """Test get_object_path with __file__ variable."""
        path = str(self.test_base_path)
        self.assertEqual(FileUtils.get_object_path(__file__), path)

    def test_get_object_path_with_filename(self):
        """Test get_object_path including filename."""
        self.assertEqual(
            FileUtils.get_object_path(__file__, inc_filename=True),
            os.path.abspath(__file__),
        )

    def test_get_object_path_from_module(self):
        """Test get_object_path with a module."""
        import pythontk

        self.assertEqual(
            FileUtils.get_object_path(pythontk), os.path.dirname(pythontk.__file__)
        )

    def test_get_object_path_from_class(self):
        """Test get_object_path with a class."""
        path = str(self.test_base_path)

        class TestClass:
            pass

        self.assertEqual(FileUtils.get_object_path(TestClass), path)

    def test_get_object_path_from_function(self):
        """Test get_object_path with a function."""
        path = str(self.test_base_path)

        def test_function():
            pass

        self.assertEqual(FileUtils.get_object_path(test_function), path)

    def test_get_object_path_none(self):
        """Test get_object_path with None."""
        self.assertEqual(FileUtils.get_object_path(None), "")

    # -------------------------------------------------------------------------
    # get_file Tests
    # -------------------------------------------------------------------------

    def test_get_file_opens_handle(self):
        """Test get_file opens file handle."""
        file_handle = FileUtils.get_file(str(self.file1_path))
        self.assertIn("TextIOWrapper", str(type(file_handle)))
        file_handle.close()

    # -------------------------------------------------------------------------
    # get_classes_from_path Tests
    # -------------------------------------------------------------------------

    def test_get_classes_from_dir(self):
        """Test get_classes_from_path discovers classes in Python files."""
        path = str(self.test_base_path)
        result = FileUtils.get_classes_from_path(path, "classname")
        self.assertIn("BaseTestCase", result)

    # -------------------------------------------------------------------------
    # Version Management Tests
    # -------------------------------------------------------------------------

    def test_update_version(self):
        """Test PackageManager version management."""
        from pythontk.core_utils import PackageManager

        FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')
        result = PackageManager.update_version(str(self.file1_path), "increment")
        self.assertEqual(str(result), "0.9.1")

        result = PackageManager.update_version(str(self.file1_path), "decrement")
        self.assertEqual(str(result), "0.9.0")

    # -------------------------------------------------------------------------
    # JSON Tests
    # -------------------------------------------------------------------------

    def test_json_set_and_get_file(self):
        """Test JSON file operations."""
        json_path = str(self.test_files_path / "test.json")
        FileUtils.set_json_file(json_path)
        self.assertEqual(FileUtils.get_json_file(), json_path)

    def test_json_set_and_get_value(self):
        """Test JSON value operations."""
        json_path = str(self.test_files_path / "test.json")
        FileUtils.set_json_file(json_path)
        FileUtils.set_json("key", "value")
        self.assertEqual(FileUtils.get_json("key"), "value")

    def test_json_nested_values(self):
        """Test JSON with nested values."""
        json_path = str(self.test_files_path / "test.json")
        FileUtils.set_json_file(json_path)
        nested = {"a": {"b": {"c": 1}}}
        FileUtils.set_json("nested", nested)
        result = FileUtils.get_json("nested")
        self.assertEqual(result, nested)

    def test_json_array_values(self):
        """Test JSON with array values."""
        json_path = str(self.test_files_path / "test.json")
        FileUtils.set_json_file(json_path)
        array = [1, 2, 3, "four", None]
        FileUtils.set_json("array", array)
        result = FileUtils.get_json("array")
        self.assertEqual(result, array)

    def test_json_nonexistent_key(self):
        """Test JSON get with nonexistent key."""
        json_path = str(self.test_files_path / "test.json")
        FileUtils.set_json_file(json_path)
        result = FileUtils.get_json("nonexistent_key_xyz")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(exit=False)
