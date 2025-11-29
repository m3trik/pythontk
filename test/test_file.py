#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk FileUtils.

Run with:
    python -m pytest test_file.py -v
    python test_file.py
"""
import os
import unittest
from pathlib import Path

from pythontk import FileUtils

from conftest import BaseTestCase, TestPaths


class FileTest(BaseTestCase):
    """File utilities test class."""

    @classmethod
    def setUpClass(cls):
        """Set up test paths used across file tests."""
        cls.test_base_path = TestPaths.BASE_DIR
        cls.test_files_path = TestPaths.TEST_FILES_DIR
        cls.file1_path = cls.test_files_path / "file1.txt"
        cls.file2_path = cls.test_files_path / "file2.txt"

    def test_format_path(self):
        """Test format_path normalizes and parses paths."""
        # Test basic path normalization
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3"), "X:/n/dir1/dir3")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3", "path"), "X:/n/dir1/dir3"
        )
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "path"),
            "X:/n/dir1/dir3/.vscode",
        )
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "path"),
            "X:/n/dir1/dir3/.vscode",
        )

        # Test UNC path
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "path"),
            r"\\192.168.1.240/nas/lost+found",
        )

        # Test environment variable expansion
        self.assertEqual(
            FileUtils.format_path(r"%programfiles%", "path"), "C:/Program Files"
        )

        # Test directory extraction
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

        # Test file extraction
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

        # Test name extraction
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

        # Test extension extraction
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

        # Test edge cases
        self.assertEqual(FileUtils.format_path(r"programfiles", "name"), "programfiles")
        self.assertEqual(FileUtils.format_path(r"programfiles", "path"), "programfiles")
        self.assertEqual(
            FileUtils.format_path(r"programfiles/", "path"), "programfiles"
        )

        # Test list input
        self.assertEqual(
            FileUtils.format_path(
                [r"X:\n/dir1/dir3", r"X:\n/dir1/dir3/.vscode"], "dir"
            ),
            ["dir3", ".vscode"],
        )

    def test_is_valid(self):
        """Test is_valid checks file/directory existence."""
        self.assertTrue(FileUtils.is_valid(str(self.file1_path), "file"))
        self.assertTrue(FileUtils.is_valid(str(self.test_files_path), "dir"))

    def test_write_to_file(self):
        """Test write_to_file writes content correctly."""
        result = FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')
        self.assertIsNone(result)

    def test_get_file_contents(self):
        """Test get_file_contents reads file content."""
        # Ensure file has expected content
        FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')

        content = FileUtils.get_file_contents(str(self.file1_path), as_list=True)
        self.assertEqual(content, ['__version__ = "0.9.0"'])

    def test_create_directory(self):
        """Test create_dir creates directories."""
        sub_dir = str(self.test_files_path / "sub-directory")
        result = FileUtils.create_dir(sub_dir)
        self.assertIsNone(result)
        self.assertTrue(os.path.isdir(sub_dir))

    def test_get_file_info(self):
        """Test get_file_info extracts file metadata."""
        files = [str(self.file1_path), str(self.file2_path)]

        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "filename", "filepath"]),
            [
                ("file1.txt", "file1", str(self.file1_path)),
                ("file2.txt", "file2", str(self.file2_path)),
            ],
        )

        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "filetype"]),
            [("file1.txt", ".txt"), ("file2.txt", ".txt")],
        )

        self.assertEqual(
            FileUtils.get_file_info(files, ["filename", "filetype"]),
            [("file1", ".txt"), ("file2", ".txt")],
        )

        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "size"]),
            [
                ("file1.txt", os.path.getsize(str(self.file1_path))),
                ("file2.txt", os.path.getsize(str(self.file2_path))),
            ],
        )

    def test_get_directory_contents(self):
        """Test get_dir_contents lists directory contents."""
        path = str(self.test_files_path)
        base_path = str(self.test_base_path)

        imgtk_test_dirpath = os.path.join(base_path, "test_files\\imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files\\sub-directory")

        with self.subTest("Test returned dirpaths"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "dirpath"),
                [imgtk_test_dirpath, sub_directory_dirpath],
            )

        with self.subTest("Test returned filenames recursively"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "filename", recursive=True),
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

        with self.subTest("Test returned file and dir"):
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

        with self.subTest("Test with exc_dirs"):
            self.assertEqual(
                sorted(
                    FileUtils.get_dir_contents(path, ["file", "dir"], exc_dirs=["sub*"])
                ),
                sorted(["imgtk_test", "file1.txt", "file2.txt", "test.json"]),
            )

        with self.subTest("Test with inc_files"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "filename", inc_files="*.txt"),
                ["file1", "file2"],
            )

        with self.subTest("Test returned file with inc_files"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "file", inc_files="*.txt"),
                ["file1.txt", "file2.txt"],
            )

        with self.subTest("Test returned dirpath and dir"):
            self.assertEqual(
                sorted(FileUtils.get_dir_contents(path, ["dirpath", "dir"])),
                [
                    imgtk_test_dirpath,
                    sub_directory_dirpath,
                    "imgtk_test",
                    "sub-directory",
                ],
            )

        with self.subTest("Test group_by_type functionality"):
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

    def test_get_object_path(self):
        """Test get_object_path extracts path from various objects."""
        path = str(self.test_base_path)

        # Test with __file__ variable
        self.assertEqual(FileUtils.get_object_path(__file__), path)
        self.assertEqual(
            FileUtils.get_object_path(__file__, inc_filename=True),
            os.path.abspath(__file__),
        )

        # Test with a module
        import pythontk

        self.assertEqual(
            FileUtils.get_object_path(pythontk), os.path.dirname(pythontk.__file__)
        )

        # Test with a class
        class TestClass:
            pass

        self.assertEqual(FileUtils.get_object_path(TestClass), path)

        # Test with a callable object (function)
        def test_function():
            pass

        self.assertEqual(FileUtils.get_object_path(test_function), path)

        # Test with None
        self.assertEqual(FileUtils.get_object_path(None), "")

    def test_get_file(self):
        """Test get_file opens file handle."""
        file_handle = FileUtils.get_file(str(self.file1_path))
        self.assertIn("TextIOWrapper", str(type(file_handle)))
        file_handle.close()

    def test_get_classes_from_dir(self):
        """Test get_classes_from_path discovers classes in Python files."""
        path = str(self.test_base_path)

        # Note: Class names may change as we refactor - this test may need updating
        result = FileUtils.get_classes_from_path(path, "classname")
        self.assertIn("BaseTestCase", result)

    def test_update_version(self):
        """Test PackageManager version management."""
        from pythontk.core_utils import PackageManager

        # Reset to known version first
        FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')

        # Test increment
        result = PackageManager.update_version(str(self.file1_path), "increment")
        self.assertEqual(str(result), "0.9.1")

        # Test decrement
        result = PackageManager.update_version(str(self.file1_path), "decrement")
        self.assertEqual(str(result), "0.9.0")

    def test_json(self):
        """Test JSON file operations."""
        json_path = str(self.test_files_path / "test.json")

        # Set JSON file
        FileUtils.set_json_file(json_path)
        self.assertEqual(FileUtils.get_json_file(), json_path)

        # Set/get JSON value
        FileUtils.set_json("key", "value")
        self.assertEqual(FileUtils.get_json("key"), "value")


if __name__ == "__main__":
    unittest.main(exit=False)
