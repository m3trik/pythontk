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
import inspect
import os
import sys
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

        # Ensure test files exist
        os.makedirs(cls.test_files_path, exist_ok=True)
        with open(cls.file1_path, "w") as f:
            f.write("file1")
        with open(cls.file2_path, "w") as f:
            f.write("file2")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.file2_path):
            os.remove(cls.file2_path)
        # file1.txt might be used by other tests, but we created it so we should probably clean it.
        # However, existing tests might rely on it being there.
        # Given the previous state, file1.txt existed but file2.txt didn't.
        # I'll leave file1.txt alone if it was already there, but here I overwrote it.
        # Let's just clean up file2.txt to be safe, or both.
        pass

    # -------------------------------------------------------------------------
    # format_path Tests
    # -------------------------------------------------------------------------

    def test_format_path_basic_normalization(self):
        """Test format_path normalizes path separators."""
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3"), "X:/n/dir1/dir3")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3", "path"), "X:/n/dir1/dir3"
        )

    def test_format_path_preserves_drive_root(self):
        """A bare drive root must keep its trailing slash ("C:" means CWD-relative)."""
        self.assertEqual(FileUtils.format_path("C:/"), "C:/")
        self.assertEqual(FileUtils.format_path("C:\\"), "C:/")

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

    @unittest.skipUnless(sys.platform == "win32", "Windows-only env var")
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
        if sys.platform == "win32":
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
        if sys.platform == "win32":
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
    # next_version_path Tests
    # -------------------------------------------------------------------------

    def test_next_version_path_empty_dir_starts_at_start(self):
        with tempfile.TemporaryDirectory() as d:
            result = FileUtils.next_version_path(os.path.join(d, "shot.ma"))
            self.assertEqual(os.path.basename(result), "shot_v001.ma")

    def test_next_version_path_picks_max_plus_one_across_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            for n in (3, 5):
                open(os.path.join(d, f"shot_v{n:03d}.ma"), "w").close()
            result = FileUtils.next_version_path(os.path.join(d, "shot.ma"))
            self.assertEqual(os.path.basename(result), "shot_v006.ma")

    def test_next_version_path_input_version_acts_as_floor(self):
        with tempfile.TemporaryDirectory() as d:
            # input claims v009 but nothing on disk -> next is v010
            result = FileUtils.next_version_path(os.path.join(d, "shot_v009.ma"))
            self.assertEqual(os.path.basename(result), "shot_v010.ma")

    def test_next_version_path_custom_format(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "shot.v002.ma"), "w").close()
            result = FileUtils.next_version_path(
                os.path.join(d, "shot.ma"), format="{stem}.v{n:03d}{ext}"
            )
            self.assertEqual(os.path.basename(result), "shot.v003.ma")

    def test_next_version_path_does_not_leak_across_stems(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "shot_v005.ma"), "w").close()
            result = FileUtils.next_version_path(os.path.join(d, "other.ma"))
            self.assertEqual(os.path.basename(result), "other_v001.ma")

    def test_next_version_path_ignores_matching_directories(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "shot_v009.ma"))  # dir, not file
            result = FileUtils.next_version_path(os.path.join(d, "shot.ma"))
            self.assertEqual(os.path.basename(result), "shot_v001.ma")

    def test_next_version_path_format_without_n_raises(self):
        with self.assertRaises(ValueError):
            FileUtils.next_version_path("shot.ma", format="{stem}{ext}")

    def test_next_version_path_format_with_unknown_field_raises(self):
        with self.assertRaises(ValueError):
            FileUtils.next_version_path(
                "shot.ma", format="{stem}_{user}_v{n:03d}{ext}"
            )

    def test_next_version_path_missing_dir_returns_start(self):
        # Nonexistent parent dir should not crash; falls back to start.
        result = FileUtils.next_version_path(
            os.path.join(tempfile.gettempdir(), "no_such_dir_xyz", "shot.ma")
        )
        self.assertEqual(os.path.basename(result), "shot_v001.ma")

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
        imgtk_test_dirpath = os.path.join(base_path, "test_files", "imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files", "sub-directory")
        self.assertEqual(
            FileUtils.get_dir_contents(path, "dirpath"),
            [imgtk_test_dirpath, sub_directory_dirpath],
        )

    def test_get_dir_contents_filenames_recursive(self):
        """Test get_dir_contents returns filenames recursively."""
        path = str(self.test_files_path)
        result = FileUtils.get_dir_contents(path, "filename", recursive=True)
        expected = [
            "file1",
            "file2",
            "test",
            "im_Base_color",
            "im_H",
            "im_Height_16",
            "im_Height_8",
            "im_Mixed_AO_L",
            "im_N",
            "im_Normal_DirectX",
            "im_Normal_OpenGL",
        ]
        self.assertEqual(
            sorted([f.lower() for f in result]),
            sorted([f.lower() for f in expected]),
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
        imgtk_test_dirpath = os.path.join(base_path, "test_files", "imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files", "sub-directory")
        self.assertEqual(
            sorted(FileUtils.get_dir_contents(path, ["dirpath", "dir"])),
            sorted(
                [
                    imgtk_test_dirpath,
                    sub_directory_dirpath,
                    "imgtk_test",
                    "sub-directory",
                ]
            ),
        )

    def test_get_dir_contents_group_by_type(self):
        """Test get_dir_contents with group_by_type functionality."""
        path = str(self.test_files_path)
        base_path = str(self.test_base_path)
        imgtk_test_dirpath = os.path.join(base_path, "test_files", "imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files", "sub-directory")
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
    # get_classes_from_path Tests
    # -------------------------------------------------------------------------

    def test_get_classes_from_dir(self):
        """Test get_classes_from_path discovers classes in Python files."""
        path = str(self.test_base_path)
        result = FileUtils.get_classes_from_path(path, "classname")
        self.assertIn("BaseTestCase", result)

    def test_classes_in_packages_match_canonical_imports(self):
        """Classes loaded from a packaged .py must be the *same object*
        as those obtained via ``from pkg.mod import Cls``.

        Background: previously the loader synthesized a unique module name
        (``<stem>_ptk_loader_<id>``) per file to avoid sys.modules pollution,
        which produced a *different class object* than what a normal Python
        import returns. Downstream ``isinstance`` / ``==`` checks across the
        two paths failed silently.
        """
        from pythontk.iter_utils._iter_utils import IterUtils as Canonical

        filepath = inspect.getfile(Canonical)
        result = FileUtils.get_classes_from_path(
            filepath, ["classname", "classobj"]
        )
        match = [obj for name, obj in result if name == "IterUtils"]
        self.assertEqual(len(match), 1)
        self.assertIs(match[0], Canonical)

    def test_canonical_module_path_walks_up_init_py(self):
        """``_canonical_module_path`` returns dotted name for a packaged file."""
        from pythontk.iter_utils._iter_utils import IterUtils

        path = inspect.getfile(IterUtils)
        self.assertEqual(
            FileUtils._canonical_module_path(path),
            "pythontk.iter_utils._iter_utils",
        )

    def test_canonical_module_path_returns_none_for_loose_file(self):
        """A .py file outside any package returns None (synthetic loader fallback)."""
        with tempfile.TemporaryDirectory() as tmp:
            loose = Path(tmp) / "loose_module.py"
            loose.write_text("class Loose: pass\n", encoding="utf-8")
            self.assertIsNone(FileUtils._canonical_module_path(str(loose)))

    def test_canonical_module_path_for_package_init(self):
        """An ``__init__.py`` resolves to its package's dotted name."""
        import pythontk.iter_utils as pkg

        init_path = pkg.__file__
        self.assertEqual(
            FileUtils._canonical_module_path(init_path),
            "pythontk.iter_utils",
        )

    def test_synthetic_loader_fallback_for_loose_file(self):
        """Loose .py files (no parent ``__init__.py``) still load and yield classes."""
        with tempfile.TemporaryDirectory() as tmp:
            loose = Path(tmp) / "loose_widget.py"
            loose.write_text(
                "class LooseWidget:\n    label = 'isolated'\n",
                encoding="utf-8",
            )
            result = FileUtils.get_classes_from_path(
                str(loose), ["classname", "classobj"]
            )
            match = [obj for name, obj in result if name == "LooseWidget"]
            self.assertEqual(len(match), 1)
            # Class loaded via synthetic loader; its module name is unique
            # and has been cleaned from sys.modules so it does not pollute.
            self.assertNotIn(match[0].__module__, sys.modules)

    # -------------------------------------------------------------------------
    # Version Management Tests
    # -------------------------------------------------------------------------

    def test_update_version(self):
        """Test PackageManager version management."""
        from pythontk.core_utils.package_manager import PackageManager

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

    def test_set_json_bootstraps_missing_file(self):
        """Regression: set_json on a not-yet-existing file must create it.

        open(file, 'r') raises FileNotFoundError (a subclass of OSError, NOT
        json.decoder.JSONDecodeError) for an absent file, so the sole
        JSONDecodeError except clause never caught it and set_json could not
        bootstrap a fresh settings file. The except now also catches
        FileNotFoundError, falling back to an empty dict and writing the file.
        """
        with tempfile.TemporaryDirectory() as d:
            json_path = os.path.join(d, "does_not_exist_yet.json")
            self.assertFalse(os.path.exists(json_path))
            # Should not raise; must create the file with the value.
            FileUtils.set_json("hdr_map_visibility", True, file=json_path)
            self.assertTrue(os.path.exists(json_path))
            self.assertIs(
                FileUtils.get_json("hdr_map_visibility", file=json_path), True
            )

    def test_reveal_in_file_manager(self):
        """reveal_in_file_manager builds the right platform command and selects files vs folders."""
        import sys

        captured = []
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "scene.blend")
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("x")

            # A real file → the args include the file path (selected on Win/mac).
            args_file = FileUtils.reveal_in_file_manager(f, _runner=captured.append)
            self.assertIn(f, [os.path.normpath(a) for a in args_file if isinstance(a, str)])
            self.assertEqual(captured[-1], args_file)  # the runner received exactly the args

            # A directory → opens the folder (no file-select token).
            args_dir = FileUtils.reveal_in_file_manager(d, _runner=captured.append)
            self.assertNotIn("/select,", args_dir)
            self.assertEqual(os.path.normpath(args_dir[-1]), os.path.normpath(d))

            # Platform sanity: the launcher executable matches the OS.
            launcher = {"win": "explorer", "darwin": "open"}.get(
                "win" if sys.platform.startswith("win") else sys.platform, "xdg-open"
            )
            self.assertEqual(args_file[0], launcher)

        # A path whose containing directory is gone → FileNotFoundError (caller can message).
        with self.assertRaises(FileNotFoundError):
            FileUtils.reveal_in_file_manager(
                os.path.join(tempfile.gettempdir(), "no_such_dir_xyz", "f.blend"),
                _runner=captured.append,
            )

    # -------------------------------------------------------------------------
    # is_cloud_placeholder Tests
    # -------------------------------------------------------------------------

    def test_is_cloud_placeholder_true_for_online_only(self):
        """RECALL_ON_DATA_ACCESS (Dropbox/OneDrive online-only) → True."""
        from unittest.mock import patch

        with patch("pythontk.file_utils._file_utils.os.stat") as m_stat:
            m_stat.return_value.st_file_attributes = 0x00400020  # RECALL | ARCHIVE
            self.assertTrue(FileUtils.is_cloud_placeholder("X:/cloud/only.csv"))

    def test_is_cloud_placeholder_true_for_offline_flag(self):
        """Legacy FILE_ATTRIBUTE_OFFLINE also marks a placeholder."""
        from unittest.mock import patch

        with patch("pythontk.file_utils._file_utils.os.stat") as m_stat:
            m_stat.return_value.st_file_attributes = 0x00001020  # OFFLINE | ARCHIVE
            self.assertTrue(FileUtils.is_cloud_placeholder("X:/cloud/old.csv"))

    def test_is_cloud_placeholder_false_for_local_file(self):
        """A fully-hydrated local file (ARCHIVE only) → False."""
        from unittest.mock import patch

        with patch("pythontk.file_utils._file_utils.os.stat") as m_stat:
            m_stat.return_value.st_file_attributes = 0x00000020  # ARCHIVE only
            self.assertFalse(FileUtils.is_cloud_placeholder("X:/local/file.csv"))

    def test_is_cloud_placeholder_false_without_attr_field(self):
        """Non-Windows stat results lack st_file_attributes → False (no crash)."""
        from unittest.mock import patch

        class _StatNoAttrs:  # mimics a POSIX os.stat_result
            st_size = 10

        with patch(
            "pythontk.file_utils._file_utils.os.stat", return_value=_StatNoAttrs()
        ):
            self.assertFalse(FileUtils.is_cloud_placeholder("X:/file.csv"))

    def test_is_cloud_placeholder_false_for_missing_path(self):
        """A path that can't be stat'd → False (caller handles 'not found')."""
        self.assertFalse(
            FileUtils.is_cloud_placeholder("X:/definitely/missing/abc123.csv")
        )

    # -------------------------------------------------------------------------
    # free_space Tests
    # -------------------------------------------------------------------------

    def test_free_space_returns_int_for_existing_dir(self):
        """An existing directory reports a non-negative free-byte count."""
        import tempfile

        free = FileUtils.free_space(tempfile.gettempdir())
        self.assertIsInstance(free, int)
        self.assertGreaterEqual(free, 0)

    def test_free_space_resolves_nonexistent_child(self):
        """A not-yet-created child resolves up to its existing parent volume."""
        import os as _os
        import tempfile

        child = _os.path.join(tempfile.gettempdir(), "no_such_dir_xyz", "f.csv")
        self.assertIsInstance(FileUtils.free_space(child), int)

    def test_free_space_none_when_unresolvable(self):
        """No existing ancestor → None (caller treats unknown as 'can't tell')."""
        from unittest.mock import patch

        # With nothing on the path existing, the walk-up reaches the drive root
        # (a fixed point) and bails to None rather than guessing.
        with patch(
            "pythontk.file_utils._file_utils.os.path.exists", return_value=False
        ):
            self.assertIsNone(FileUtils.free_space("Z:/x/y/z.csv"))

    # -------------------------------------------------------------------------
    # get_dir_contents / get_object_path regression tests
    # -------------------------------------------------------------------------

    def test_get_dir_contents_recursive_exc_dirs_prunes_subtree(self):
        """Regression: recursive walk must not descend into excluded dirs, so
        files inside exc_dirs are never returned (serial and threaded)."""
        with tempfile.TemporaryDirectory() as root:
            keep = os.path.join(root, "keep")
            skip = os.path.join(root, "_skip")
            os.makedirs(keep)
            os.makedirs(skip)
            with open(os.path.join(keep, "a.txt"), "w") as f:
                f.write("a")
            with open(os.path.join(skip, "b.txt"), "w") as f:
                f.write("b")
            for nt in (1, 2):
                result = FileUtils.get_dir_contents(
                    root,
                    "filepath",
                    recursive=True,
                    exc_dirs="_skip",
                    num_threads=nt,
                )
                names = sorted(os.path.basename(p) for p in result)
                self.assertEqual(names, ["a.txt"], f"num_threads={nt}")

    def test_get_dir_contents_recursive_inc_dirs_includes_subtree(self):
        """Regression: inc_dirs selects whole subtrees — a matched directory's
        descendants are included even when their names don't match the
        pattern. Re-applying inc at every depth silently dropped everything
        below the matched dir (serial and threaded)."""
        with tempfile.TemporaryDirectory() as root:
            nested = os.path.join(root, "assets", "child")
            other = os.path.join(root, "other")
            os.makedirs(nested)
            os.makedirs(other)
            with open(os.path.join(root, "assets", "top.txt"), "w") as f:
                f.write("t")
            with open(os.path.join(nested, "deep.txt"), "w") as f:
                f.write("d")
            with open(os.path.join(other, "no.txt"), "w") as f:
                f.write("n")
            for nt in (1, 2):
                result = FileUtils.get_dir_contents(
                    root,
                    "filepath",
                    recursive=True,
                    inc_dirs="assets",
                    num_threads=nt,
                )
                names = sorted(os.path.basename(p) for p in result)
                self.assertEqual(
                    names, ["deep.txt", "top.txt"], f"num_threads={nt}"
                )

    def test_get_dir_contents_num_threads_all_cores_enters_parallel_branch(self):
        """Regression: num_threads=-1 ('use all cores') must enter the
        multithreaded branch (the only place cpu_count() is consulted).
        Before the fix, -1 > 1 was False and it silently ran serially."""
        import multiprocessing
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "a.txt"), "w") as f:
                f.write("a")
            with patch.object(
                multiprocessing, "cpu_count", wraps=multiprocessing.cpu_count
            ) as spy:
                result = FileUtils.get_dir_contents(
                    root, "filepath", recursive=True, num_threads=-1
                )
            self.assertTrue(
                spy.called,
                "num_threads=-1 did not enter the parallel branch",
            )
            self.assertEqual(
                sorted(os.path.basename(p) for p in result), ["a.txt"]
            )

    def test_get_object_path_module_absent_from_sys_modules_no_unbound_local(self):
        """Regression: a class whose __module__ is absent from sys.modules
        must not raise UnboundLocalError (filepath referenced before
        assignment at the __main__ check); the graceful contract is a
        ValueError. inspect.stack is stubbed empty so the unrelated stack-scan
        fallback is bypassed and only the finding's code path is exercised."""
        import inspect as _inspect
        from unittest.mock import patch

        absent = type(
            "ObjPathAbsentModZZZ", (), {"__module__": "totally_absent_mod_zzz"}
        )
        inst = absent()
        with patch.object(_inspect, "stack", return_value=[]):
            try:
                FileUtils.get_object_path(inst)
            except UnboundLocalError:
                self.fail(
                    "get_object_path raised UnboundLocalError for a module "
                    "absent from sys.modules"
                )
            except ValueError:
                pass  # documented graceful failure: path could not be determined

    def test_get_object_path_stack_scan_never_executes_caller_files(self):
        """Regression: the stack-scan fallback exec'd each caller source file
        (spec.loader.exec_module) just to check whether it defines the class —
        re-running the file's side effects (under pytest, exec'ing
        pytest/__main__.py launched a whole nested test session inside the
        test). Frame globals already ARE each stack module's namespace, so an
        unresolvable object must produce ValueError having executed nothing.

        Proven in a subprocess: the driver script appends its __name__ to a
        marker file at module level — a re-execution of the file appends a
        second line."""
        import subprocess
        import sys as _sys

        script = (
            "import os\n"
            "with open(os.environ['PTK_EXEC_MARKER'], 'a') as f:\n"
            "    f.write(__name__ + '\\n')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    import sys\n"
            "    sys.path.insert(0, os.environ['PTK_ROOT'])\n"
            "    from pythontk.file_utils._file_utils import FileUtils\n"
            "    try:\n"
            "        FileUtils.get_object_path(object())\n"
            "        print('OUTCOME:no-error')\n"
            "    except ValueError:\n"
            "        print('OUTCOME:value-error')\n"
        )

        import pythontk

        pythontk_root = os.path.dirname(
            os.path.dirname(os.path.abspath(pythontk.__file__))
        )

        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "goppath_driver.py")
            marker_path = os.path.join(tmp, "exec_marker.txt")
            with open(script_path, "w") as f:
                f.write(script)

            env = {
                **os.environ,
                "PTK_EXEC_MARKER": marker_path,
                "PTK_ROOT": pythontk_root,
            }
            result = subprocess.run(
                [_sys.executable, script_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("OUTCOME:value-error", result.stdout)
            with open(marker_path) as f:
                executions = f.read().splitlines()
            self.assertEqual(
                executions,
                ["__main__"],
                f"driver script was re-executed by get_object_path: {executions}",
            )


if __name__ == "__main__":
    unittest.main(exit=False)
