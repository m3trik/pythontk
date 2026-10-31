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
import shutil
import tempfile
import unittest
from pathlib import Path

from pythontk import FileUtils

from conftest import BaseTestCase, TestPaths


class FileTest(BaseTestCase):
    """File utilities test class with comprehensive edge case coverage."""

    @classmethod
    def setUpClass(cls):
        """Lay out the tree the tests read: ``file1.txt`` (committed -- the tests
        rewrite it, so its bytes are kept to put back), and ``file2.txt`` and
        ``sub-directory``, made here."""
        cls.test_base_path = TestPaths.BASE_DIR
        cls.test_files_path = TestPaths.TEST_FILES_DIR
        cls.file1_path = cls.test_files_path / "file1.txt"
        cls.file2_path = cls.test_files_path / "file2.txt"
        cls.sub_dir_path = cls.test_files_path / "sub-directory"

        os.makedirs(cls.sub_dir_path, exist_ok=True)
        cls._file1_committed = (
            cls.file1_path.read_bytes() if cls.file1_path.exists() else None
        )
        with open(cls.file1_path, "w") as f:
            f.write("file1")
        with open(cls.file2_path, "w") as f:
            f.write("file2")

    @classmethod
    def tearDownClass(cls):
        """Leave the tree as committed, whichever tests ran: a subset run left
        ``file1.txt`` rewritten, and a release commits the working tree."""
        if cls._file1_committed is not None:
            cls.file1_path.write_bytes(cls._file1_committed)
        if os.path.exists(cls.file2_path):
            os.remove(cls.file2_path)
        shutil.rmtree(cls.sub_dir_path, ignore_errors=True)

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

    def test_a_failed_read_returns_the_default(self):
        """The failure mode belongs in the signature, not in folklore.

        Measured 2026-08-20: the swallowed OSError handed None to
        ``_PkgVersionUtils.update_version``, which fed it straight to
        ``enumerate`` and raised ``TypeError: 'NoneType' object is not
        iterable`` mid version-bump -- a failure reported nowhere near the
        read that actually failed.
        """
        missing = os.path.join(str(self.test_files_path), "no_such_file.txt")
        self.assertIsNone(FileUtils.get_file_contents(missing))
        self.assertEqual(FileUtils.get_file_contents(missing, default=""), "")
        self.assertEqual(
            FileUtils.get_file_contents(missing, as_list=True, default=[]), []
        )

    def test_the_default_is_ignored_when_the_read_succeeds(self):
        FileUtils.write_to_file(str(self.file1_path), "real")
        self.assertEqual(
            FileUtils.get_file_contents(str(self.file1_path), default="fallback"),
            "real",
        )

    def test_write_and_read_unicode(self):
        """Test write and read with unicode content."""
        unicode_content = "Café ★ 🚀 naïve résumé"
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
        new_dir = str(self.test_files_path / "created-directory")
        self.addCleanup(shutil.rmtree, new_dir, ignore_errors=True)
        self.assertFalse(os.path.exists(new_dir))
        result = FileUtils.create_dir(new_dir)
        self.assertIsNone(result)
        self.assertTrue(os.path.isdir(new_dir))

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
            FileUtils.next_version_path("shot.ma", format="{stem}_{user}_v{n:03d}{ext}")

    def test_next_version_path_missing_dir_returns_start(self):
        # Nonexistent parent dir should not crash; falls back to start.
        result = FileUtils.next_version_path(
            os.path.join(tempfile.gettempdir(), "no_such_dir_xyz", "shot.ma")
        )
        self.assertEqual(os.path.basename(result), "shot_v001.ma")

    def test_next_version_path_counts_every_sibling_extension(self):
        """One deliverable written as several files versions as ONE: a GLB-only
        export leaves no .fbx behind, so scanning its own extension alone numbered
        every export v001 and overwrote the last one."""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "shot_v004.glb"), "w").close()
            result = FileUtils.next_version_path(
                os.path.join(d, "shot.fbx"), extensions=(".glb",)
            )
            self.assertEqual(os.path.basename(result), "shot_v005.fbx")

    def test_next_version_number_scans_a_folder_for_a_whole_name(self):
        """No stem to split off: the resolved name IS the template, counter anywhere."""
        with tempfile.TemporaryDirectory() as d:
            for name in ("v002_hero.fbx", "v007_hero.glb", "v009_other.fbx"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual(
                FileUtils.next_version_number(d, "v{n:03d}_hero{ext}", ext=".fbx"), 3
            )
            self.assertEqual(
                FileUtils.next_version_number(
                    d, "v{n:03d}_hero{ext}", ext=".fbx", extensions=(".glb",)
                ),
                8,
            )

    def test_next_version_number_starts_at_start_and_reads_escaped_braces(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                FileUtils.next_version_number(d, "a_v{n}{ext}", ext=".ma", start=5), 5
            )
            open(os.path.join(d, "a{b}_v7.ma"), "w").close()
            self.assertEqual(
                FileUtils.next_version_number(d, "a{{b}}_v{n}{ext}", ext=".ma"), 8
            )
        missing = os.path.join(tempfile.gettempdir(), "no_such_dir_xyz")
        self.assertEqual(FileUtils.next_version_number(missing, "x_v{n}{ext}"), 1)

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
        result = FileUtils.get_classes_from_path(filepath, ["classname", "classobj"])
        match = [obj for name, obj in result if name == "IterUtils"]
        self.assertEqual(len(match), 1)
        self.assertIs(match[0], Canonical)

    def test_canonical_module_path_walks_up_init_py(self):
        """``canonical_module_path`` returns dotted name for a packaged file."""
        from pythontk.iter_utils._iter_utils import IterUtils

        path = inspect.getfile(IterUtils)
        self.assertEqual(
            FileUtils.canonical_module_path(path),
            "pythontk.iter_utils._iter_utils",
        )

    def test_canonical_module_path_returns_none_for_loose_file(self):
        """A .py file outside any package returns None (synthetic loader fallback)."""
        with tempfile.TemporaryDirectory() as tmp:
            loose = Path(tmp) / "loose_module.py"
            loose.write_text("class Loose: pass\n", encoding="utf-8")
            self.assertIsNone(FileUtils.canonical_module_path(str(loose)))

    def test_canonical_module_path_for_package_init(self):
        """An ``__init__.py`` resolves to its package's dotted name."""
        import pythontk.iter_utils as pkg

        init_path = pkg.__file__
        self.assertEqual(
            FileUtils.canonical_module_path(init_path),
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

    def test_a_file_defining_no_class_is_never_executed(self):
        """Scanning must not run a module that cannot yield a class.

        Switchboard slot discovery scans a whole package tree with this, so a
        class-less script there ran at every DCC start: a scratch probe left
        inside mayatk launched a fresh Maya on import, whose own startup scan
        imported it again -- a new instance every ~20 s, each one hung.
        """
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "executed.txt"
            script = Path(tmp) / "run_probe.py"
            script.write_text(f"open(r'{marker}', 'w').close()\n", encoding="utf-8")
            (Path(tmp) / "widget.py").write_text(
                "class Widget: pass\n", encoding="utf-8"
            )

            by_file = FileUtils.get_classes_from_path(str(script), "classname")
            by_dir = FileUtils.get_classes_from_path(tmp, "classname")

            self.assertEqual(by_file, [])
            self.assertEqual(by_dir, ["Widget"])
            self.assertFalse(marker.exists(), "the class-less script was executed")

    def test_a_file_saved_with_a_bom_is_scanned(self):
        """Python imports a UTF-8 file with a BOM (Windows PowerShell 5.1
        writes one); the scan must read it too, not die in ``ast.parse`` --
        one such file in a slots tree took a whole ``Switchboard`` down."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bom_widget.py").write_text(
                "class BomWidget:\n    pass\n", encoding="utf-8-sig"
            )
            self.assertEqual(
                FileUtils.get_classes_from_path(tmp, "classname"), ["BomWidget"]
            )

    def test_an_unchanged_file_is_parsed_once(self):
        """A Switchboard re-scans every slot file on each construction: parsing
        and walking the same unchanged files was 95% of uitk's switchboard tests
        (measured 2026-10-21: 22,757 parses in one module, 299 of 315 s)."""
        import ast
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cached_widget.py").write_text(
                "class CachedWidget:\n    class Inner:\n        pass\n", encoding="utf-8"
            )
            with mock.patch.object(ast, "parse", wraps=ast.parse) as parse:
                first = FileUtils.get_classes_from_path(tmp, "classname", top_level_only=False)
                second = FileUtils.get_classes_from_path(tmp, "classname", top_level_only=False)
                top = FileUtils.get_classes_from_path(tmp, "classname")
            self.assertEqual(first, ["CachedWidget", "Inner"])
            self.assertEqual(second, first)
            self.assertEqual(top, ["CachedWidget"])  # the mode is part of the key
            self.assertEqual(parse.call_count, 2)  # once per mode, not once per scan

    def test_an_edited_file_is_parsed_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edited_widget.py"
            path.write_text("class Before:\n    pass\n", encoding="utf-8")
            self.assertEqual(FileUtils.get_classes_from_path(tmp, "classname"), ["Before"])
            path.write_text("class AfterTheEdit:\n    pass\n", encoding="utf-8")
            self.assertEqual(
                FileUtils.get_classes_from_path(tmp, "classname"), ["AfterTheEdit"]
            )

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

    def test_reveal_in_file_manager(self):
        """reveal_in_file_manager builds the right platform command and selects files vs folders."""
        import sys

        captured = []

        def answered(args):  # a file manager that answers (Linux: the D-Bus call)
            captured.append(args)
            return True

        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "scene.blend")
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("x")

            # A real file → selected: by path on Windows/macOS, by URI through
            # freedesktop FileManager1 on Linux.
            args_file = FileUtils.reveal_in_file_manager(f, _runner=answered)
            self.assertEqual(captured[-1], args_file)  # the runner got exactly these
            launcher = {"win": "explorer", "darwin": "open"}.get(
                "win" if sys.platform.startswith("win") else sys.platform, "dbus-send"
            )
            self.assertEqual(args_file[0], launcher)
            if launcher == "dbus-send":
                self.assertIn(f"array:string:{Path(f).as_uri()}", args_file)
                # No file manager on the bus → the containing folder opens.
                captured.clear()
                fallback = FileUtils.reveal_in_file_manager(f, _runner=captured.append)
                self.assertEqual(fallback, ["xdg-open", os.path.normpath(d)])
                self.assertEqual([a[0] for a in captured], ["dbus-send", "xdg-open"])
            else:
                self.assertIn(
                    f, [os.path.normpath(a) for a in args_file if isinstance(a, str)]
                )

            # A directory → opens the folder (no file-select token).
            args_dir = FileUtils.reveal_in_file_manager(d, _runner=captured.append)
            self.assertNotIn("/select,", args_dir)
            self.assertNotIn("dbus-send", args_dir)
            self.assertEqual(os.path.normpath(args_dir[-1]), os.path.normpath(d))

        # A path whose containing directory is gone → FileNotFoundError (caller can message).
        with self.assertRaises(FileNotFoundError):
            FileUtils.reveal_in_file_manager(
                os.path.join(tempfile.gettempdir(), "no_such_dir_xyz", "f.blend"),
                _runner=captured.append,
            )

    def test_reveal_in_file_manager_resolves_a_relative_path(self):
        """A path relative to the cwd is made absolute first: Linux's file URI
        refuses a relative path (``ValueError``), a bare file name (dirname
        ``""``) raised FileNotFoundError on every OS, and a file manager would
        resolve a relative one against its OWN cwd."""
        import sys

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "sub"))
            names = (os.path.join("sub", "scene.blend"), "top.blend")
            for name in names:
                with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
                    fh.write("x")
            cwd = os.getcwd()
            os.chdir(d)
            try:
                for name in names:
                    absolute = os.path.join(os.getcwd(), name)
                    args = FileUtils.reveal_in_file_manager(
                        name, _runner=lambda a: True
                    )
                    if sys.platform.startswith("win") or sys.platform == "darwin":
                        self.assertEqual(args[-1], absolute)
                    else:
                        self.assertIn(f"array:string:{Path(absolute).as_uri()}", args)
            finally:
                os.chdir(cwd)

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
    # is_locked / locking_processes / describe_lock Tests
    # -------------------------------------------------------------------------

    @staticmethod
    def _hold_open(path):
        """Open *path* with a share mode that denies writers, as a viewer does.

        Windows share modes are per-HANDLE, not per-process, so holding this
        in the test process blocks the test process' own writes -- the real
        lock, with no subprocess to spawn or reap.
        """
        import ctypes
        from ctypes import wintypes

        create = ctypes.windll.kernel32.CreateFileW
        create.restype = wintypes.HANDLE
        create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING.
        handle = create(path, 0x80000000, 0x00000001, None, 3, 0, None)
        assert handle != wintypes.HANDLE(-1).value, "could not open the probe file"
        return handle

    def _locked_file(self):
        """A real, currently-locked file; released at test teardown."""
        import ctypes
        import os as _os
        import tempfile
        from ctypes import wintypes

        path = _os.path.join(tempfile.mkdtemp(), "held.glb")
        with open(path, "wb") as fh:
            fh.write(b"glTF")
        handle = self._hold_open(path)
        self.addCleanup(ctypes.windll.kernel32.CloseHandle, wintypes.HANDLE(handle))
        return path

    def test_is_locked_false_for_a_writable_file(self):
        """Nothing holding it -> replaceable -> False."""
        import os as _os
        import tempfile

        path = _os.path.join(tempfile.mkdtemp(), "free.glb")
        with open(path, "wb") as fh:
            fh.write(b"glTF")
        self.assertFalse(FileUtils.is_locked(path))

    def test_is_locked_false_for_missing_path_and_for_a_directory(self):
        """Neither can be 'held'; both answer False so callers need no guard."""
        import tempfile

        self.assertFalse(FileUtils.is_locked("X:/no/such/file_abc123.glb"))
        self.assertFalse(FileUtils.is_locked(tempfile.gettempdir()))

    @unittest.skipUnless(os.name == "nt", "file locking is a Windows behavior")
    def test_is_locked_true_while_another_handle_denies_writes(self):
        """The case this exists for: a viewer has the deliverable open."""
        path = self._locked_file()
        self.assertTrue(FileUtils.is_locked(path))

    @unittest.skipUnless(os.name == "nt", "file locking is a Windows behavior")
    def test_is_locked_agrees_with_the_operations_it_gates(self):
        """A True verdict must mean os.remove and os.replace really do fail.

        The probe opens for append while the caller deletes or renames; if
        those diverged, the check would be a superstition.
        """
        import os as _os

        path = self._locked_file()
        # A real source for the rename, or the raise would be a missing file.
        source = path + ".src"
        with open(source, "wb") as fh:
            fh.write(b"replacement")

        self.assertTrue(FileUtils.is_locked(path))
        with self.assertRaises(OSError):
            _os.remove(path)
        with self.assertRaises(OSError):
            _os.replace(source, path)
        self.assertTrue(_os.path.isfile(source), "the failed rename consumed it")

    @unittest.skipUnless(os.name == "nt", "Restart Manager is Windows-only")
    def test_locking_processes_names_the_holder(self):
        """The holder is this very process, so its PID must be in the list."""
        import os as _os

        path = self._locked_file()
        holders = FileUtils.locking_processes(path)
        self.assertTrue(holders, "Restart Manager named no holder for a held file")
        self.assertTrue(
            any(f"PID {_os.getpid()}" in h for h in holders),
            f"this process ({_os.getpid()}) is missing from {holders}",
        )

    def test_locking_processes_is_empty_off_windows(self):
        """Every non-Windows caller gets a clean, allocation-free no-op."""
        from unittest.mock import patch

        with patch("pythontk.file_utils._file_utils.os.name", "posix"):
            self.assertEqual(FileUtils.locking_processes(__file__), [])

    def test_describe_lock_is_falsy_when_the_file_is_free(self):
        """It doubles as the test, so a free file must read as empty."""
        self.assertEqual(FileUtils.describe_lock(__file__), "")

    @unittest.skipUnless(os.name == "nt", "file locking is a Windows behavior")
    def test_describe_lock_reports_the_holder(self):
        path = self._locked_file()
        self.assertIn("in use by", FileUtils.describe_lock(path))

    @unittest.skipUnless(os.name == "nt", "file locking is a Windows behavior")
    def test_describe_lock_still_reports_a_lock_it_cannot_attribute(self):
        """An unattributable lock is still a lock - never silently 'free'."""
        from unittest.mock import patch

        path = self._locked_file()
        with patch.object(FileUtils, "locking_processes", return_value=[]):
            self.assertEqual(FileUtils.describe_lock(path), "in use by another process")

    def test_describe_lock_says_read_only_rather_than_in_use(self):
        """REGRESSION (2026-10-19): a read-only destination (a version-control
        checkout's) cannot be replaced either, but read as "in use by another
        process" -- and the Scene Exporter's writability check, which every
        export now makes, told the user to close a program that held nothing."""
        import stat

        import pythontk as ptk

        temp = ptk.TempArtifacts("describe_lock", policy="scoped")
        self.addCleanup(temp.cleanup)
        path = temp.path(extension=".fbx")
        with open(path, "wb") as fh:
            fh.write(b"Kaydara")
        os.chmod(path, stat.S_IREAD)
        self.addCleanup(os.chmod, path, stat.S_IREAD | stat.S_IWRITE)
        self.assertEqual(FileUtils.describe_lock(path), "read-only")

    # -------------------------------------------------------------------------
    # describe_unwritable Tests
    # -------------------------------------------------------------------------

    def _scratch_folder(self):
        """A throwaway folder, removed at teardown (after any denial is lifted)."""
        import pythontk as ptk

        temp = ptk.TempArtifacts("describe_unwritable", policy="scoped")
        self.addCleanup(temp.cleanup)
        folder = temp.dir_path()
        os.makedirs(folder, exist_ok=True)
        return folder

    def _deny_writes(self, folder):
        """Deny this user the right to create files in *folder* -- a real ACL
        denial (a folder's read-only attribute does not stop a write on
        Windows), lifted at teardown before the folder is removed."""
        import getpass
        import subprocess

        user = getpass.getuser()
        denied = subprocess.run(
            ["icacls", folder, "/deny", f"{user}:(W)"], capture_output=True
        )
        if denied.returncode != 0:
            self.skipTest(f"icacls could not deny writes: {denied.stderr!r}")
        self.addCleanup(
            subprocess.run,
            ["icacls", folder, "/remove:d", user],
            capture_output=True,
        )

    @unittest.skipUnless(os.name == "nt", "the ACL denial is made with icacls")
    def test_describe_unwritable_names_a_folder_this_user_cannot_write(self):
        """REGRESSION (2026-10-19): the Scene Exporter's writability check asked
        only whether the destination FILE was held, so an existing folder the
        user cannot write to passed it and the run failed at the write."""
        folder = self._scratch_folder()
        self._deny_writes(folder)
        path = os.path.join(folder, "asset.fbx")
        self.assertEqual(FileUtils.describe_lock(path), "", "nothing holds it")
        why = FileUtils.describe_unwritable(path)
        self.assertTrue(why.startswith("the folder is not writable: "), why)
        self.assertGreater(len(why), len("the folder is not writable: "), why)

    @unittest.skipUnless(os.name == "nt", "the ACL denial is made with icacls")
    def test_describe_unwritable_asks_the_nearest_folder_that_exists(self):
        """A missing output folder is made at the write, inside its nearest
        existing ancestor -- which is the folder that must take it."""
        folder = self._scratch_folder()
        path = os.path.join(folder, "new", "deeper", "asset.fbx")
        self.assertEqual(FileUtils.describe_unwritable(path), "")
        self.assertFalse(os.path.exists(os.path.join(folder, "new")), "made nothing")
        self._deny_writes(folder)
        self.assertTrue(
            FileUtils.describe_unwritable(path).startswith("the folder is not writable")
        )

    def test_describe_unwritable_leaves_nothing_in_a_folder_that_takes_it(self):
        folder = self._scratch_folder()
        open(os.path.join(folder, "kept.txt"), "w").close()
        self.assertEqual(
            FileUtils.describe_unwritable(os.path.join(folder, "asset.glb")), ""
        )
        self.assertEqual(os.listdir(folder), ["kept.txt"], "the probe is removed")

    def test_describe_unwritable_reports_why_the_probe_was_refused(self):
        """Any platform: whatever refuses the probe file is the reason given."""
        from unittest.mock import patch

        folder = self._scratch_folder()
        refused = PermissionError(13, "Permission denied")
        with patch("pythontk.file_utils._file_utils.os.open", side_effect=refused):
            why = FileUtils.describe_unwritable(os.path.join(folder, "a.fbx"))
        self.assertEqual(why, "the folder is not writable: Permission denied")

    @unittest.skipUnless(os.name == "nt", "file locking is a Windows behavior")
    def test_describe_unwritable_still_names_what_holds_the_file(self):
        """The file's own answer comes first: closing the holder is the fix."""
        path = self._locked_file()
        self.assertIn("in use by", FileUtils.describe_unwritable(path))

    # -------------------------------------------------------------------------
    # is_under Tests
    # -------------------------------------------------------------------------

    def test_is_under_true_for_a_descendant(self):
        root = os.path.normpath("/proj")
        self.assertTrue(FileUtils.is_under(os.path.join(root, "a", "b.png"), root))

    def test_is_under_requires_a_separator_boundary(self):
        """A bare startswith puts '/proj2/x.png' inside '/proj'."""
        self.assertFalse(
            FileUtils.is_under(
                os.path.normpath("/proj2/x.png"), os.path.normpath("/proj")
            )
        )

    def test_is_under_normalizes_separators_and_trailing_slash(self):
        self.assertTrue(FileUtils.is_under("O:/proj/sub/x.png", "O:/proj/"))
        if os.name == "nt":  # mixed separators only occur on Windows
            self.assertTrue(FileUtils.is_under("O:/proj/sub/x.png", "O:\\proj"))

    def test_is_under_inclusive_controls_the_identity_case(self):
        root = os.path.normpath("/proj")
        self.assertTrue(FileUtils.is_under(root, root))
        self.assertFalse(FileUtils.is_under(root, root, inclusive=False))

    def test_is_under_case_follows_the_platform(self):
        under = FileUtils.is_under(
            os.path.normpath("/PROJ/x.png"), os.path.normpath("/proj")
        )
        self.assertEqual(under, os.name == "nt")

    def test_is_under_empty_operands_are_false(self):
        self.assertFalse(FileUtils.is_under("", "/proj"))
        self.assertFalse(FileUtils.is_under("/proj/x.png", ""))

    def test_is_under_does_not_touch_the_filesystem_or_the_cwd(self):
        """No abspath: a relative path must not be resolved against the CWD."""
        self.assertFalse(FileUtils.is_under("x.png", os.getcwd()))

    def test_remap_file_paths_needs_a_folder_boundary(self):
        """``/proj2/b.png`` is not under ``/proj``: a bare startswith took it
        for inside, and relpath then mapped it to ``../proj2/b.png`` -- out of
        the target folder."""
        root = os.path.abspath(os.sep)
        base, target = os.path.join(root, "proj"), os.path.join(root, "out")
        inside = os.path.join(base, "tex", "a.png")
        sibling = os.path.join(root, "proj2", "b.png")
        (k1, new1, _), (k2, new2, _) = FileUtils.remap_file_paths(
            [inside, sibling], target, base
        )
        self.assertEqual(k1, "tex/a.png")
        self.assertEqual(os.path.normpath(new1), os.path.join(target, "tex", "a.png"))
        self.assertEqual(k2, "b.png")  # not under base: flattened by name
        self.assertEqual(os.path.normpath(new2), os.path.join(target, "b.png"))

    # -------------------------------------------------------------------------
    # is_rooted_path / resolve_output_dir Tests
    # -------------------------------------------------------------------------

    def test_is_rooted_path_needs_a_drive_or_posix_root(self):
        if os.name == "nt":
            self.assertTrue(FileUtils.is_rooted_path("C:/bakes"))
            self.assertTrue(FileUtils.is_rooted_path(r"\\server\share\bakes"))
        else:  # no drives here: 'C:/bakes' is a folder named 'C:'
            self.assertTrue(FileUtils.is_rooted_path("/bakes"))
            self.assertFalse(FileUtils.is_rooted_path("C:/bakes"))
        self.assertFalse(FileUtils.is_rooted_path("bakes"))
        self.assertFalse(FileUtils.is_rooted_path(""))

    def test_is_rooted_path_rejects_a_driveless_root_on_windows(self):
        """os.path.isabs('/new') is True on Windows and resolves to the CURRENT
        drive's root -- which is why this test exists rather than isabs."""
        self.assertEqual(FileUtils.is_rooted_path("/new"), os.sep == "/")

    @unittest.skipUnless(os.name == "nt", "drive letters are a Windows spelling")
    def test_is_rooted_path_rejects_a_drive_relative_entry(self):
        """'C:new' carries a drive but no root: Windows resolves it against the
        CWD *on that drive*, so it is the same trap as '/new' in another
        spelling -- a full path only when the drive is followed by a root."""
        self.assertFalse(FileUtils.is_rooted_path("C:new"))
        self.assertFalse(FileUtils.is_rooted_path("C:"))
        self.assertTrue(FileUtils.is_rooted_path("C:/new"))
        self.assertTrue(FileUtils.is_rooted_path(r"C:\new"))

    def test_resolve_output_dir_empty_entry_is_the_base(self):
        base = os.path.normpath("/proj/sourceimages")
        self.assertEqual(FileUtils.resolve_output_dir("", base), base)
        self.assertEqual(FileUtils.resolve_output_dir("   ", base), base)

    def test_resolve_output_dir_joins_a_subdirectory_entry(self):
        base = os.path.normpath("/proj/sourceimages")
        self.assertEqual(
            FileUtils.resolve_output_dir("lightmaps", base),
            os.path.join(base, "lightmaps"),
        )
        self.assertEqual(
            FileUtils.resolve_output_dir("bake/lm", base),
            os.path.normpath(os.path.join(base, "bake/lm")),
        )

    def test_resolve_output_dir_treats_a_separator_spelled_entry_as_a_subdir(self):
        """On Windows '/new/' and 'new' name the same folder -- the driveless-root
        trap. On POSIX '/new/' is a full path, and a full path wins."""
        base = os.path.normpath("/proj/sourceimages")
        expected = (
            os.path.join(base, "lightmaps")
            if os.name == "nt"
            else os.path.normpath("/lightmaps")
        )
        self.assertEqual(FileUtils.resolve_output_dir("/lightmaps/", base), expected)

    def test_resolve_output_dir_keeps_a_full_path(self):
        base = os.path.normpath("/proj/sourceimages")
        full = "C:/bakes/lm" if os.name == "nt" else "/bakes/lm"
        self.assertEqual(
            FileUtils.resolve_output_dir(full, base), os.path.normpath(full)
        )

    def test_resolve_output_dir_strips_quotes_and_expands(self):
        base = os.path.normpath("/proj/sourceimages")
        self.assertEqual(
            FileUtils.resolve_output_dir('  " lightmaps "  ', base),
            os.path.join(base, "lightmaps"),
        )
        from unittest.mock import patch

        with patch.dict(os.environ, {"PTK_OUT": "lightmaps"}):
            var = "%PTK_OUT%" if os.name == "nt" else "$PTK_OUT"
            self.assertEqual(
                FileUtils.resolve_output_dir(var, base),
                os.path.join(base, "lightmaps"),
            )

    def test_resolve_output_dir_never_returns_a_relative_path(self):
        """A relative result would be created against the process CWD by the
        caller's makedirs -- in a DCC, wherever the app was launched from."""
        self.assertIsNone(FileUtils.resolve_output_dir("lightmaps", ""))
        self.assertIsNone(FileUtils.resolve_output_dir("", None))
        full = "C:/bakes" if os.name == "nt" else "/bakes"
        self.assertEqual(
            FileUtils.resolve_output_dir(full, None), os.path.normpath(full)
        )

    def test_resolve_output_dir_drive_relative_entry_is_a_subdir(self):
        """'C:new' must not pass through verbatim: makedirs would create it
        against the CWD on drive C:, which is the relative-path failure this
        promises never to return."""
        base = os.path.normpath("/proj/sourceimages")
        for entry in ("C:new", "C:bake/lm"):
            resolved = FileUtils.resolve_output_dir(entry, base)
            self.assertTrue(
                os.path.isabs(resolved), f"{entry!r} resolved to {resolved!r}"
            )
            self.assertTrue(FileUtils.is_under(resolved, base))

    # -------------------------------------------------------------------------
    # relativize_output_dir Tests -- the inverse of resolve_output_dir
    # -------------------------------------------------------------------------

    def test_relativize_output_dir_is_the_portable_spelling(self):
        base = os.path.normpath("/proj/sourceimages")
        self.assertEqual(
            FileUtils.relativize_output_dir(os.path.join(base, "uv_transfer"), base),
            "uv_transfer",
        )
        self.assertEqual(
            FileUtils.relativize_output_dir(os.path.join(base, "bakes", "v2"), base),
            "bakes/v2",
        )

    def test_relativize_output_dir_base_itself_is_the_empty_default(self):
        """The field reads empty as *base*, so the base must not come back as
        '.' -- that would resolve to the same place while reading as an edit."""
        base = os.path.normpath("/proj/sourceimages")
        self.assertEqual(FileUtils.relativize_output_dir(base, base), "")
        self.assertEqual(FileUtils.relativize_output_dir("", base), "")

    def test_relativize_output_dir_keeps_an_outside_path_verbatim(self):
        """That is what the user picked, and '../..' is not portable either.
        Verbatim, not normalized: re-spelling the separators of the value the
        dialog just wrote would show the user an edit they did not make."""
        base = os.path.normpath("/proj/sourceimages")
        outside = "D:/bakes" if os.name == "nt" else "/bakes"
        self.assertEqual(FileUtils.relativize_output_dir(outside, base), outside)
        # A sibling of base must not become "../sourceimages_old".
        sibling = os.path.normpath("/proj/sourceimages_old")
        self.assertEqual(FileUtils.relativize_output_dir(sibling, base), sibling)
        # No base to measure against -> nothing to shorten.
        self.assertEqual(FileUtils.relativize_output_dir(sibling, None), sibling)

    def test_relativize_output_dir_round_trips_through_resolve(self):
        """The pair is what a browse dialog + a stored field rely on: the text
        written back must resolve to the folder that was picked."""
        base = os.path.normpath("/proj/sourceimages")
        for picked in (
            os.path.join(base, "uv_transfer"),
            os.path.join(base, "bakes", "v2"),
            base,
        ):
            entry = FileUtils.relativize_output_dir(picked, base)
            self.assertEqual(FileUtils.resolve_output_dir(entry, base), picked, entry)

    @unittest.skipUnless(os.name == "nt", "drive letters are a Windows path shape")
    def test_relativize_output_dir_drive_letter_case_is_one_drive(self):
        """'o:' and 'O:' are the same drive. is_under folds the case, so the
        relative spelling must too: a case-sensitive drive compare treated
        them as two drives and kept only the leaf ('hero'), which resolves to
        a different folder than the one picked."""
        base = r"O:\proj\sourceimages"
        picked = r"o:\proj\sourceimages\lightmaps\hero"
        self.assertEqual(
            FileUtils.convert_to_relative_path(picked, base, prepend_base=False),
            "lightmaps/hero",
        )
        entry = FileUtils.relativize_output_dir(picked, base)
        self.assertEqual(entry, "lightmaps/hero")
        self.assertEqual(
            os.path.normcase(FileUtils.resolve_output_dir(entry, base)),
            os.path.normcase(picked),
        )

    # -------------------------------------------------------------------------
    # portable_path / resolve_portable_path Tests -- scene-record file paths
    # -------------------------------------------------------------------------

    def test_portable_path_is_relative_to_the_base(self):
        """A scene record stores no machine's drive layout: under the project
        root the spelling is root-relative, forward slashes on every OS."""
        root = os.path.abspath("/proj")
        self.assertEqual(
            FileUtils.portable_path(os.path.join(root, "sound", "vo.wav"), root),
            "sound/vo.wav",
        )

    def test_portable_path_spells_a_path_beside_the_base_as_a_chain(self):
        """Maintainer rule, 2026-09-23: relative to the SCENE'S OWN project
        wherever a relative spelling reaches -- a shared library beside the
        project is a ``../`` chain, so a teammate who mounts the project with
        its neighbours resolves the same file.  (Spelled from the SESSION's
        project the chain drifted -- measured 2026-09-22 -- which is why the
        base is the scene's and a Save As re-spells it.)"""
        root = os.path.abspath("/work/proj")
        lib = os.path.join(os.path.abspath("/work/library"), "hit.wav")
        self.assertEqual(FileUtils.portable_path(lib, root), "../library/hit.wav")
        self.assertEqual(
            FileUtils.resolve_portable_path("../library/hit.wav", root),
            os.path.normpath(lib).replace("\\", "/"),
        )
        # The project itself (a folder record) is ".", never "" -- an empty
        # spelling reads back as nothing at all.
        self.assertEqual(FileUtils.portable_path(root, root), ".")
        self.assertEqual(
            FileUtils.resolve_portable_path(".", root),
            os.path.normpath(root).replace("\\", "/"),
        )

    def test_rebase_portable_path_follows_a_scene_into_another_project(self):
        """A Save As into another project re-spells what the record names:
        the same files, spelled from the new project."""
        old = os.path.abspath("/work/proj")
        new = os.path.abspath("/work/shows/copy")
        for picked in (
            os.path.join(old, "sound", "vo.wav"),
            os.path.join(os.path.abspath("/work/library"), "hit.wav"),
        ):
            stored = FileUtils.portable_path(picked, old)
            moved = FileUtils.rebase_portable_path(stored, old, new)
            self.assertEqual(
                FileUtils.resolve_portable_path(moved, new),
                os.path.normpath(picked).replace("\\", "/"),
                moved,
            )
        self.assertEqual(
            FileUtils.rebase_portable_path("../library/hit.wav", old, new),
            "../../library/hit.wav",
        )
        # Equal bases normalize: an absolute value (written before the rule, or
        # arrived across a hand-off) becomes relative where it can.
        legacy = os.path.join(old, "sound", "vo.wav")
        self.assertEqual(
            FileUtils.rebase_portable_path(legacy, old, old), "sound/vo.wav"
        )
        # A relative value with no base to read it from stays as it is.
        self.assertEqual(
            FileUtils.rebase_portable_path("sound/vo.wav", None, new), "sound/vo.wav"
        )
        self.assertEqual(FileUtils.rebase_portable_path("", old, new), "")

    def test_portable_path_keeps_a_path_no_relative_spelling_reaches(self):
        """Across drives no relative spelling exists; nor without a base."""
        if os.name == "nt":
            self.assertEqual(
                FileUtils.portable_path("D:\\lib\\hit.wav", "C:\\proj"),
                "D:/lib/hit.wav",
            )
        full = os.path.abspath("/lib/hit.wav")
        self.assertEqual(
            FileUtils.portable_path(full, ""), os.path.normpath(full).replace("\\", "/")
        )
        self.assertEqual(FileUtils.portable_path("", "/proj"), "")

    def test_resolve_portable_path_round_trips(self):
        root = os.path.abspath("/work/proj")
        for picked in (
            os.path.join(root, "sound", "vo.wav"),
            os.path.join(os.path.abspath("/work/library"), "hit.wav"),
        ):
            stored = FileUtils.portable_path(picked, root)
            self.assertEqual(
                FileUtils.resolve_portable_path(stored, root),
                os.path.normpath(picked).replace("\\", "/"),
                stored,
            )

    def test_resolve_portable_path_leaves_an_absolute_value_alone(self):
        """Maps written before the rule hold absolute paths; they still read."""
        full = os.path.abspath("/old/machine/vo.wav")
        self.assertEqual(
            FileUtils.resolve_portable_path(full, os.path.abspath("/proj")),
            os.path.normpath(full).replace("\\", "/"),
        )
        self.assertEqual(FileUtils.resolve_portable_path("", "/proj"), "")
        # Starting at a root, a value is relative to nothing: a Windows join
        # would keep only the base's DRIVE and silently re-home it there.
        self.assertEqual(
            FileUtils.resolve_portable_path(
                "/audio/foot.wav", os.path.abspath("/proj")
            ),
            "/audio/foot.wav",
        )

    # -------------------------------------------------------------------------
    # is_same_file / has_same_content Tests -- the checks before a copy reuses
    # -------------------------------------------------------------------------

    def test_is_same_file_asks_the_filesystem_not_the_spelling(self):
        store = __import__("pythontk").TempArtifacts("ptk_same_file")
        self.addCleanup(store.cleanup)
        folder = store.dir_path()
        path = os.path.join(folder, "Map.exr")
        with open(path, "wb") as fh:
            fh.write(b"x")
        # Case is the FILESYSTEM's call: one file on NTFS/APFS, two names
        # (the second naming nothing) on a case-sensitive Linux filesystem.
        other_case = os.path.join(folder, "MAP.EXR")
        self.assertEqual(
            FileUtils.is_same_file(path, other_case), os.path.exists(other_case)
        )
        self.assertTrue(
            FileUtils.is_same_file(path, os.path.join(folder, "sub", "..", "Map.exr"))
        )
        self.assertFalse(FileUtils.is_same_file(path, os.path.join(folder, "Other")))
        # Neither on disk: the normalized spellings decide ('\\' separates only
        # on Windows; elsewhere it is a filename character).
        self.assertEqual(
            FileUtils.is_same_file("gone/a.exr", "gone\\a.exr"), os.name == "nt"
        )

    def test_has_same_content_is_the_bytes_not_the_size(self):
        store = __import__("pythontk").TempArtifacts("ptk_same_content")
        self.addCleanup(store.cleanup)
        folder = store.dir_path()

        def write(name, data):
            path = os.path.join(folder, name)
            with open(path, "wb") as fh:
                fh.write(data)
            return path

        old, new, twin = (
            write("old.exr", b"OLD-BAKE"),
            write("new.exr", b"NEW-BAKE"),
            write("twin.exr", b"OLD-BAKE"),
        )
        self.assertFalse(FileUtils.has_same_content(old, new), "same size, not same")
        self.assertTrue(FileUtils.has_same_content(old, twin))
        self.assertFalse(FileUtils.has_same_content(old, os.path.join(folder, "gone")))

    # -------------------------------------------------------------------------
    # path_length_limit / exceeds_path_length Tests
    # -------------------------------------------------------------------------

    def test_path_length_limit_is_a_plausible_os_cap(self):
        """Windows reports MAX_PATH or the extended max; POSIX reports PATH_MAX."""
        limit = FileUtils.path_length_limit()
        self.assertIsInstance(limit, int)
        if os.name == "nt":
            self.assertIn(limit, (260, 32767))
        else:
            self.assertGreaterEqual(limit, 255)

    def test_exceeds_path_length_honors_an_explicit_budget(self):
        """An explicit limit overrides the OS cap — callers leave headroom."""
        path = os.path.abspath("a/b/c.png")
        self.assertTrue(FileUtils.exceeds_path_length(path, len(path) - 1))
        self.assertFalse(FileUtils.exceeds_path_length(path, len(path)))

    def test_exceeds_path_length_measures_the_absolute_form(self):
        """A short relative path resolving to a long absolute one still counts."""
        rel = "t.png"
        absolute = os.path.abspath(rel)
        self.assertGreater(len(absolute), len(rel))
        self.assertTrue(FileUtils.exceeds_path_length(rel, len(rel)))

    def test_exceeds_path_length_empty_never_exceeds(self):
        self.assertFalse(FileUtils.exceeds_path_length(""))
        self.assertFalse(FileUtils.exceeds_path_length(None))

    def test_exceeds_path_length_defaults_to_the_os_limit(self):
        limit = FileUtils.path_length_limit()
        long_path = os.path.abspath("x/" * (limit // 2 + 8) + "t.png")
        self.assertGreater(len(long_path), limit)
        self.assertTrue(FileUtils.exceeds_path_length(long_path))

    def test_path_length_limit_is_cached_as_the_os_constant_it_is(self):
        """A per-call registry read would answer the same thing every time."""
        self.assertIs(FileUtils.path_length_limit(), FileUtils.path_length_limit())
        self.assertGreater(FileUtils.path_length_limit.cache_info().hits, 0)

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
                self.assertEqual(names, ["deep.txt", "top.txt"], f"num_threads={nt}")

    def test_get_dir_contents_lists_in_one_order_on_every_file_system(self):
        """A Linux file system lists a directory in no particular order, NTFS
        sorted (upper-cased, then as written), so on Linux the listing, and
        which of two same-named presets won, changed with the disk; the
        threaded walk also took results as they completed. Every listing is
        now in NTFS order. Simulated with a ``scandir`` that lists in reverse,
        which ``os.walk`` goes through too."""
        from unittest.mock import patch

        real_scandir = os.scandir

        class ReversedListing:
            def __init__(self, path="."):
                with real_scandir(path) as entries:
                    self._entries = iter(list(entries)[::-1])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._entries)

            def close(self):
                pass

        files = ("b.txt", "A.txt", "_c.txt")
        with tempfile.TemporaryDirectory() as root:
            for sub in ("B", "a"):
                os.makedirs(os.path.join(root, sub))
                for name in files:
                    open(os.path.join(root, sub, name), "w").close()
            for name in files:
                open(os.path.join(root, name), "w").close()
            with patch.object(os, "scandir", ReversedListing):
                flat = FileUtils.get_dir_contents(root, ["dir", "file"])
                walked = {
                    n: FileUtils.get_dir_contents(
                        root, "filepath", recursive=True, num_threads=n
                    )
                    for n in (1, 4)
                }

        ordered = ["A.txt", "b.txt", "_c.txt"]
        self.assertEqual(flat, ["a", "B"] + ordered)
        expected = [
            os.path.join(root, *parts)
            for parts in [(n,) for n in ordered]
            + [("a", n) for n in ordered]
            + [("B", n) for n in ordered]
        ]
        for n, result in walked.items():
            self.assertEqual(result, expected, f"num_threads={n}")

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
            self.assertEqual(sorted(os.path.basename(p) for p in result), ["a.txt"])

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


class RetiredSurfaceTest(unittest.TestCase):
    """Deleted in 0.11.0 after a release of warnings: ``ptk.Git`` (no callers,
    no tests) and the JSON key-value cluster (``set_json_file`` /
    ``get_json_file`` / ``set_json`` / ``get_json`` -- a process-global path,
    ``assert`` validation, a non-atomic write).  The names must not come back
    on a new primitive: a caller pinned to an old release would silently get
    a different contract under the same name."""

    def test_the_retired_names_are_gone(self):
        import pythontk as ptk

        for name in ("get_json", "set_json", "get_json_file", "set_json_file"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(FileUtils, name))
        self.assertFalse(hasattr(ptk, "Git"))


class JsonPrimitiveTest(unittest.TestCase):
    """``read_json`` / ``write_json`` -- the halves the class was missing.

    ``atomic_write_text`` existed and was composed at exactly two sites, while
    five JSON writers wrote non-atomically and tolerant reads were re-derived
    with at least six different catch sets: ``json.JSONDecodeError`` alone
    (x4), ``ValueError`` (x3), ``(OSError, ValueError)`` (x2),
    ``FileNotFoundError``, ``(json.decoder.JSONDecodeError, FileNotFoundError)``,
    ``(TypeError, ValueError)``. Those are not equivalent -- ``JSONDecodeError``
    does not catch a missing file -- so half the call sites crashed where the
    other half returned a default.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ptk_json_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def path(self, name="doc.json"):
        return os.path.join(self.dir, name)

    # -- read ----------------------------------------------------------

    def test_round_trip(self):
        p = self.path()
        data = {"a": [1, 2, {"b": None}], "c": "text"}
        FileUtils.write_json(p, data)
        self.assertEqual(FileUtils.read_json(p), data)

    def test_a_missing_file_is_the_default_not_an_exception(self):
        self.assertIsNone(FileUtils.read_json(self.path("absent.json")))
        self.assertEqual(FileUtils.read_json(self.path("absent.json"), default={}), {})

    def test_malformed_content_is_the_default(self):
        p = self.path()
        with open(p, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(
            FileUtils.read_json(p, default={"fallback": True}), {"fallback": True}
        )

    def test_a_directory_in_place_of_a_file_is_the_default(self):
        """An OSError that is not FileNotFoundError -- the case the
        ``JSONDecodeError``-only call sites crashed on."""
        self.assertEqual(FileUtils.read_json(self.dir, default="d"), "d")

    def test_an_empty_file_is_the_default(self):
        p = self.path()
        open(p, "w").close()
        self.assertEqual(FileUtils.read_json(p, default={}), {})

    # -- write ---------------------------------------------------------

    def test_write_is_atomic_leaving_no_temp_behind(self):
        p = self.path()
        FileUtils.write_json(p, {"k": 1})
        self.assertEqual(os.listdir(self.dir), [os.path.basename(p)])

    def test_an_unencodable_value_leaves_the_existing_file_intact(self):
        """The failure mode ``set_json`` had: it truncated the file, THEN
        serialised, so a value json could not encode destroyed the previous
        contents. Serialise first, write second."""
        p = self.path()
        FileUtils.write_json(p, {"good": 1})
        with self.assertRaises(TypeError):
            FileUtils.write_json(p, {"bad": object()})
        self.assertEqual(
            FileUtils.read_json(p), {"good": 1}, "the old file was destroyed"
        )
        self.assertEqual(
            os.listdir(self.dir), [os.path.basename(p)], "a temp file was stranded"
        )

    def test_write_creates_missing_parent_directories(self):
        p = os.path.join(self.dir, "nested", "deep", "doc.json")
        FileUtils.write_json(p, {"k": 1})
        self.assertEqual(FileUtils.read_json(p), {"k": 1})

    def test_indent_and_sort_keys_are_honoured(self):
        p = self.path()
        FileUtils.write_json(p, {"b": 1, "a": 2}, indent=None, sort_keys=True)
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"a": 2, "b": 1}')

    def test_the_deprecated_names_are_not_what_was_added(self):
        """The retirement plan's constraint, restated as a live check: the new
        primitive must NOT be called get_json/set_json."""
        self.assertTrue(hasattr(FileUtils, "read_json"))
        self.assertTrue(hasattr(FileUtils, "write_json"))


class UserConfigSaveTest(unittest.TestCase):
    """``UserConfig`` published a loader and no writer, so every caller that
    needed to persist a config hand-rolled one -- non-atomically."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ptk_uc_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_save_file_round_trips_through_load_file(self):
        from pythontk.core_utils.user_config import UserConfig

        p = os.path.join(self.dir, "cfg.json")
        UserConfig.save_file(p, {"root": "x", "tuning": {"quality": 2}})
        self.assertEqual(
            UserConfig.load_file(p), {"root": "x", "tuning": {"quality": 2}}
        )

    def test_save_file_creates_the_parent_directory(self):
        from pythontk.core_utils.user_config import UserConfig

        p = os.path.join(self.dir, "pkg", "cfg.json")
        UserConfig.save_file(p, {"a": 1})
        self.assertEqual(UserConfig.load_file(p), {"a": 1})

    def test_save_file_refuses_a_non_object(self):
        """``load_file`` returns {} for anything that is not a JSON object, so
        writing one would create a file its own loader silently discards."""
        from pythontk.core_utils.user_config import UserConfig

        with self.assertRaises(TypeError):
            UserConfig.save_file(os.path.join(self.dir, "c.json"), [1, 2, 3])


class AtomicJsonWriterAdoptionTest(unittest.TestCase):
    """The five non-atomic JSON writers are down to the deprecated one.

    ``atomic_write_text`` existed and was composed at two sites while five
    writers truncated their target and serialised into it. Four are migrated
    to ``write_json``; the fifth is ``set_json``, which is on its way out.

    ``QcLog.finalize`` had NO test of any kind, so its migration is covered
    here rather than assumed -- it is also the one whose truncation is worst,
    since a half-written QC log reads as a completed run with fields missing.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ptk_adopt_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_qc_log_writes_atomically_and_creates_its_parent(self):
        from pythontk import QcLog

        path = os.path.join(self.dir, "nested", "qc.json")
        QcLog(path).finalize(True)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.listdir(os.path.dirname(path)), ["qc.json"])
        data = FileUtils.read_json(path)
        self.assertIs(data["success"], True)
        self.assertIn("total_duration_sec", data)

    def test_the_metadata_sidecar_round_trips_a_non_ascii_value(self):
        """The sidecar's write is now atomic; this pins that the migration did
        not change what it stores.

        The old ``open(path, "w")`` carried no ``encoding=``, which LOOKS like
        a cp1252 bug on Windows and is not: ``json.dump`` defaults to
        ``ensure_ascii=True``, so the bytes were pure ASCII escapes either
        way -- verified before claiming otherwise. The real fix here is
        truncation, not encoding.
        """
        from pythontk.file_utils.metadata import MetadataInternal

        target = os.path.join(self.dir, "asset.fbx")
        open(target, "w").close()
        # The sidecar writer directly: Metadata.set's default path needs
        # win32com.propsys, which is not the code this migration touched.
        MetadataInternal._save_sidecar(target, {"note": "café ☕", "who": "Zoë"})
        self.assertEqual(MetadataInternal._load_sidecar(target)["note"], "café ☕")

    def test_no_module_writes_json_by_truncating_first(self):
        """Drift guard: a new ``json.dump(x, open(p, "w"))`` is the pattern
        this pass removed, and it reintroduces silently."""
        import pythontk as ptk

        root = os.path.dirname(os.path.abspath(ptk.__file__))
        offenders = []
        for folder, _dirs, files in os.walk(root):
            if "__pycache__" in folder:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as fh:
                    source = fh.read()
                for lineno, line in enumerate(source.splitlines(), 1):
                    if "json.dump(" in line or "write(json.dumps" in line:
                        rel = os.path.relpath(path, root).replace(os.sep, "/")
                        offenders.append(rel)
        # By FILE, not line: line numbers churn on every edit above them, and
        # the fact worth guarding is "which module still does this".
        # The last one, the JSON key-value set_json, was deleted in 0.11.0.
        self.assertEqual(
            sorted(set(offenders)),
            [],
            "a non-atomic JSON writer was added. Use FileUtils.write_json.",
        )


class AtomicWriteTest(unittest.TestCase):
    """``FileUtils.atomic_write``: a replacement lands whole or not at all."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ptk_atomic_write_test_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.target = os.path.join(self.dir, "deliverable.glb")
        with open(self.target, "wb") as fh:
            fh.write(b"previous")

    @staticmethod
    def _read(path):
        with open(path, "rb") as fh:
            return fh.read()

    @staticmethod
    def _writer(data):
        def write(part):
            with open(part, "wb") as fh:
                fh.write(data)

        return write

    def test_a_finished_write_replaces_the_target_leaving_nothing_beside_it(self):
        written = FileUtils.atomic_write(self.target, self._writer(b"new"))
        self.assertEqual(written, self.target)
        self.assertEqual(self._read(self.target), b"new")
        self.assertEqual(os.listdir(self.dir), ["deliverable.glb"])

    def test_a_write_that_fails_part_way_leaves_the_target_whole(self):
        """Disk full as the writer meets it: part of the file lands first."""

        def disk_full(part):
            self._writer(b"ne")(part)
            raise OSError("No space left on device")

        with self.assertRaises(OSError):
            FileUtils.atomic_write(self.target, disk_full)
        self.assertEqual(self._read(self.target), b"previous")
        self.assertEqual(
            os.listdir(self.dir), ["deliverable.glb"], "a partial file was left"
        )

    def test_a_refused_promotion_leaves_the_target_whole(self):
        """Windows refuses to replace a file another process holds open."""
        from unittest import mock

        with mock.patch("os.replace", side_effect=PermissionError("held open")):
            with self.assertRaises(PermissionError):
                FileUtils.atomic_write(self.target, self._writer(b"new"))
        self.assertEqual(self._read(self.target), b"previous")
        self.assertEqual(
            os.listdir(self.dir), ["deliverable.glb"], "the written part was left"
        )

    def test_unpromoted_the_part_waits_beside_the_target_for_its_caller(self):
        """For a caller that must close a handle on the target first."""
        part = FileUtils.atomic_write(self.target, self._writer(b"new"), promote=False)
        self.assertTrue(part.endswith(".part"), part)
        self.assertTrue(
            os.path.samefile(os.path.dirname(part), self.dir),
            "beside the target, or the caller's replace is not atomic",
        )
        self.assertEqual(self._read(self.target), b"previous")
        os.replace(part, self.target)
        self.assertEqual(self._read(self.target), b"new")

    @unittest.skipUnless(os.name == "nt", "Windows lock codes")
    def test_a_momentarily_held_target_is_still_replaced(self):
        from unittest import mock

        replace, calls = ReplaceFileTest.held(2)
        with mock.patch("os.replace", side_effect=replace), mock.patch("time.sleep"):
            FileUtils.atomic_write(self.target, self._writer(b"new"))
        self.assertEqual(self._read(self.target), b"new")
        self.assertEqual(len(calls), 3)


class DeliverFileTest(unittest.TestCase):
    """``FileUtils.deliver_file``: a finished file lands once, or is parked
    beside its destination -- never lost with the stage it was finished in."""

    _read = staticmethod(AtomicWriteTest._read)
    _writer = staticmethod(AtomicWriteTest._writer)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ptk_deliver_test_")
        self.stage = tempfile.mkdtemp(prefix="ptk_deliver_stage_")
        for folder in (self.dir, self.stage):
            self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.target = os.path.join(self.dir, "deliverable.glb")
        self._writer(b"previous")(self.target)
        self.staged = os.path.join(self.stage, "deliverable.glb")
        self._writer(b"finished")(self.staged)

    def test_it_lands_whole_with_nothing_beside_it(self):
        placed, reason = FileUtils.deliver_file(self.staged, self.target)
        self.assertEqual((placed, reason), (self.target, None))
        self.assertEqual(self._read(self.target), b"finished")
        self.assertEqual(os.listdir(self.dir), ["deliverable.glb"])

    def test_a_held_destination_parks_it_beside_and_keeps_the_old_one(self):
        """The sync client held the destination (production, 2026-10-08)."""
        held = PermissionError(13, "Access is denied")
        from unittest.mock import patch

        with patch.object(FileUtils, "replace_file", side_effect=held):
            placed, reason = FileUtils.deliver_file(self.staged, self.target)
        self.assertEqual(os.path.basename(placed), "deliverable_unplaced_v001.glb")
        self.assertTrue(reason)
        self.assertEqual(self._read(placed), b"finished")
        self.assertEqual(self._read(self.target), b"previous")
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".part")])

    def test_a_park_cut_short_leaves_no_truncated_copy(self):
        """A full disk refuses the delivery, then the park: across volumes a
        move copies first, and the copy it gave up on must not stay beside the
        deliverable under a name that reads as the finished output.
        Added: 2026-10-10
        """
        from unittest.mock import patch

        def disk_full(src, dst):
            self._writer(b"fin")(dst)  # what a cross-volume copy had written
            raise OSError(28, "No space left on device")

        held = PermissionError(13, "Access is denied")
        with (
            patch.object(FileUtils, "replace_file", side_effect=held),
            patch("shutil.move", side_effect=disk_full),
        ):
            placed, reason = FileUtils.deliver_file(self.staged, self.target)
        self.assertEqual(placed, self.staged)
        self.assertTrue(reason)
        self.assertEqual(self._read(self.staged), b"finished")
        self.assertEqual(os.listdir(self.dir), ["deliverable.glb"])


@unittest.skipUnless(os.name == "nt", "Windows lock codes")
class ReplaceFileTest(unittest.TestCase):
    """``FileUtils.replace_file``: ``os.replace`` that waits out a momentary lock.

    A sync client, a virus scanner or the search indexer opens a file right
    after it changes; for that moment Windows refuses to replace or rename it
    (``WinError 32`` sharing violation, ``WinError 5`` access denied). Captured
    live on a synced drive: ``[WinError 5] Access is denied: '.x.tmp' ->
    '.Unity.preset'`` on the third quick rewrite of a preset's sidecar.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ptk_replace_file_test_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.src = os.path.join(self.dir, "new.txt")
        self.dst = os.path.join(self.dir, "target.txt")
        for path, text in ((self.src, "new"), (self.dst, "old")):
            with open(path, "w") as fh:
                fh.write(text)

    @staticmethod
    def held(times, winerror=32):
        """An ``os.replace`` refused *times* times, as while another process
        holds the file; ``winerror=None`` refuses the way a real permission
        problem does."""
        real, calls = os.replace, []

        def replace(src, dst):
            calls.append(dst)
            if len(calls) <= times:
                if winerror is None:
                    raise PermissionError(13, "Permission denied", dst)
                raise PermissionError(13, "held by another process", dst, winerror)
            return real(src, dst)

        return replace, calls

    def _replace(self, replace):
        from unittest import mock

        with mock.patch("os.replace", side_effect=replace), mock.patch("time.sleep"):
            FileUtils.replace_file(self.src, self.dst)

    def test_a_momentary_lock_is_waited_out(self):
        replace, calls = self.held(2)
        self._replace(replace)
        self.assertEqual(len(calls), 3)
        with open(self.dst) as fh:
            self.assertEqual(fh.read(), "new")

    def test_a_lock_that_outlasts_the_wait_raises(self):
        replace, calls = self.held(10**6, winerror=5)
        with self.assertRaises(PermissionError):
            self._replace(replace)
        self.assertGreater(len(calls), 1)
        with open(self.dst) as fh:
            self.assertEqual(fh.read(), "old")

    def test_a_large_file_earns_a_longer_wait(self):
        """A reader that opens a file just after it changes reads it THROUGH,
        so the hold grows with the file: a sync client still held a freshly
        written 218 MB FBX past the flat second, and two in-place passes were
        lost (production, 2026-10-07). The same refusals that outlast a small
        file's wait are waited out for a large one.
        Added: 2026-10-07
        """
        from unittest import mock

        attempts = len(FileUtils._REPLACE_RETRY_DELAYS) + 6
        replace, _calls = self.held(attempts)
        with self.assertRaises(PermissionError):
            self._replace(replace)  # a few bytes: the second it always had
        replace, calls = self.held(attempts)
        with mock.patch("os.path.getsize", return_value=200 * 1024 * 1024):
            self._replace(replace)
        self.assertEqual(len(calls), attempts + 1)
        with open(self.dst) as fh:
            self.assertEqual(fh.read(), "new")

    def test_any_other_refusal_raises_at_once(self):
        replace, calls = self.held(10**6, winerror=None)
        with self.assertRaises(PermissionError):
            self._replace(replace)
        self.assertEqual(len(calls), 1)


class TestUniquePath(unittest.TestCase):
    """``FileUtils.unique_path`` -- the naming rule a batch of outputs goes through."""

    def test_the_bare_name_when_nothing_stands_in_the_way(self):
        self.assertEqual(
            FileUtils.unique_path("out", "Crate", "exr"),
            os.path.join("out", "Crate.exr"),
        )

    def test_taken_names_step_to_the_next_index_and_are_recorded(self):
        taken = set()
        first = FileUtils.unique_path("out", "Crate", ".exr", taken)
        second = FileUtils.unique_path("out", "crate", ".exr", taken)
        self.assertEqual(os.path.basename(first), "Crate.exr")
        # Case-insensitive on a case-insensitive filesystem, as the disk compares.
        if os.path.normcase("A") == os.path.normcase("a"):
            self.assertEqual(os.path.basename(second), "crate_1.exr")
        self.assertEqual(taken, {first, second})

    def test_a_name_another_reader_claims_is_passed_over(self):
        claims = {"crate.exr": frozenset({"|other"})}
        path = FileUtils.unique_path(
            "out", "Crate", ".exr", claims=claims, owners=["|me"]
        )
        self.assertEqual(os.path.basename(path), "Crate_1.exr")

    def test_a_name_only_the_owners_read_stays_theirs(self):
        """Re-writing an owner's file keeps its name."""
        claims = {"crate.exr": frozenset({"|me"})}
        path = FileUtils.unique_path(
            "out", "Crate", ".exr", claims=claims, owners=["|me"]
        )
        self.assertEqual(os.path.basename(path), "Crate.exr")

    def test_a_name_shared_with_someone_else_is_not_one_owners_to_take(self):
        """A file two readers share is not free for one of them alone."""
        claims = {"crate.exr": frozenset({"|me", "|sibling"})}
        path = FileUtils.unique_path(
            "out", "Crate", ".exr", claims=claims, owners=["|me"]
        )
        self.assertEqual(os.path.basename(path), "Crate_1.exr")
        both = FileUtils.unique_path(
            "out", "Crate", ".exr", claims=claims, owners=["|me", "|sibling"]
        )
        self.assertEqual(os.path.basename(both), "Crate.exr")

    def test_a_plain_collection_claims_names_outright(self):
        path = FileUtils.unique_path(
            "out", "Crate", ".exr", claims=["CRATE.exr"], owners=["|me"]
        )
        self.assertEqual(os.path.basename(path), "Crate_1.exr")

    def test_avoided_paths_are_passed_over(self):
        avoid = {os.path.abspath(os.path.join("out", "Crate.exr"))}
        path = FileUtils.unique_path("out", "Crate", ".exr", avoid=avoid)
        self.assertEqual(os.path.basename(path), "Crate_1.exr")


class TestMoveToTrash(unittest.TestCase):
    """``FileUtils.move_to_trash`` -- the platform trash, never a permanent delete.

    Windows' own recycle call deletes a file PERMANENTLY, without a word, where
    the Recycle Bin will not keep it (a network share, a removable or SUBST
    drive, a bin set to delete at once or too small), so every one of those is
    refused up front and the file left alone. Each backend is pinned here
    rather than discovered: the freedesktop and macOS ones run on any host
    against a scratch trash, the real Recycle Bin only on Windows.
    """

    def setUp(self):
        from pythontk import TempArtifacts, TestSandbox

        # The sandbox keeps every other test out of the machine's trash; these
        # test the call itself (and purge what they put there).
        real = TestSandbox.real_trash()
        real.__enter__()
        self.addCleanup(real.__exit__, None, None, None)
        self.store = TempArtifacts("ptk_move_to_trash", policy="scoped")
        self.addCleanup(self.store.cleanup)
        self.tmp = self.store.dir_path()

    def _file(self, *parts, data=b"lightmap"):
        path = os.path.join(self.tmp, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    @staticmethod
    def _read(path):
        with open(path, "rb") as fh:
            return fh.read()

    def test_a_missing_file_or_a_folder_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            FileUtils.move_to_trash(os.path.join(self.tmp, "nope.exr"))
        folder = os.path.join(self.tmp, "maps")
        os.makedirs(folder)
        with self.assertRaises(IsADirectoryError):
            FileUtils.move_to_trash(folder)
        self.assertTrue(os.path.isdir(folder))

    def test_each_platform_takes_its_own_trash(self):
        from unittest import mock

        from pythontk.file_utils._trash import _TrashInternal

        path = self._file("a.exr")
        for platform, backend in (
            ("win32", "_windows"),
            ("darwin", "_macos"),
            ("linux", "_freedesktop"),
        ):
            with (
                mock.patch.object(sys, "platform", platform),
                mock.patch.object(
                    _TrashInternal, backend, return_value="there"
                ) as chosen,
            ):
                self.assertEqual(FileUtils.move_to_trash(path), "there")
            chosen.assert_called_once_with(os.path.abspath(path))

    # -- freedesktop.org (Linux) ------------------------------------------------

    def test_freedesktop_writes_the_record_then_moves_the_file(self):
        from unittest import mock
        from urllib.parse import quote

        from pythontk.file_utils._trash import _TrashInternal

        data_home = os.path.join(self.tmp, "xdg")
        first = self._file("maps", "Crate_Lightmap.exr", data=b"one")
        second = self._file("other", "Crate_Lightmap.exr", data=b"two")
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": data_home}):
            moved = _TrashInternal._freedesktop(first)
            again = _TrashInternal._freedesktop(second)
        trash = os.path.join(data_home, "Trash")
        self.assertEqual(moved, os.path.join(trash, "files", "Crate_Lightmap.exr"))
        self.assertEqual(again, os.path.join(trash, "files", "Crate_Lightmap.1.exr"))
        self.assertFalse(os.path.exists(first))
        self.assertEqual(self._read(moved), b"one")
        self.assertEqual(self._read(again), b"two")
        with open(
            os.path.join(trash, "info", "Crate_Lightmap.exr.trashinfo"),
            encoding="utf-8",
        ) as fh:
            record = fh.read().splitlines()
        self.assertEqual(record[:2], ["[Trash Info]", f"Path={quote(first)}"])
        self.assertTrue(record[2].startswith("DeletionDate="), record)

    def test_freedesktop_a_trash_that_cannot_be_made_touches_nothing(self):
        from unittest import mock

        from pythontk.file_utils._trash import _TrashInternal

        data_home = os.path.join(self.tmp, "xdg")
        # A FILE where the trash's folder must go: nothing can be made there.
        self._file("xdg", "Trash")
        path = self._file("maps", "Crate_Lightmap.exr")
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": data_home}):
            self.assertIsNone(_TrashInternal._freedesktop(path))
        self.assertTrue(os.path.isfile(path))

    # -- macOS --------------------------------------------------------------------

    def test_macos_moves_into_the_trash_under_finders_free_name(self):
        from pythontk.file_utils._trash import _TrashInternal

        trash = os.path.join(self.tmp, ".Trash")
        os.makedirs(trash)
        first = self._file("maps", "Crate.exr", data=b"one")
        second = self._file("other", "Crate.exr", data=b"two")
        self.assertEqual(
            _TrashInternal._macos(first, [trash]), os.path.join(trash, "Crate.exr")
        )
        self.assertEqual(
            _TrashInternal._macos(second, [trash]), os.path.join(trash, "Crate 2.exr")
        )
        self.assertEqual(self._read(os.path.join(trash, "Crate 2.exr")), b"two")

    def test_macos_with_no_trash_on_the_volume_touches_nothing(self):
        from pythontk.file_utils._trash import _TrashInternal

        path = self._file("maps", "Crate.exr")
        missing = os.path.join(self.tmp, "no_trash_here")
        self.assertIsNone(_TrashInternal._macos(path, [missing]))
        self.assertTrue(os.path.isfile(path))

    # -- Windows ------------------------------------------------------------------

    @unittest.skipUnless(sys.platform.startswith("win"), "the Recycle Bin is Windows'")
    def test_windows_the_recycle_bin_keeps_it_restorably(self):
        """The file leaves its folder for the Recycle Bin, whole, and the
        Shell's own view of the bin lists it under its original folder. The
        item is purged again afterwards."""
        import subprocess

        name = f"ptk_trash_{os.getpid()}_{id(self)}.exr"
        path = self._file(name, data=b"recycled lightmap")
        item = FileUtils.move_to_trash(path)
        if item == "":
            # The documented "recycled, but no record names it" answer: this
            # machine's bin keeps nothing to restore (a GitHub runner's D:).
            self.skipTest("this volume's Recycle Bin keeps no record")
        self.assertTrue(item, "the bin's own record names it")
        self.addCleanup(self._purge, item)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self._read(item), b"recycled lightmap")
        script = (
            "$bin = (New-Object -ComObject Shell.Application).Namespace(10); "
            f"$bin.Items() | Where-Object {{ $_.Name -eq '{name}' }} | "
            "ForEach-Object { $bin.GetDetailsOf($_, 1) }"
        )
        try:
            shown = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            self.skipTest(f"no PowerShell to ask the Shell ({error})")
        self.assertIn(
            os.path.normcase(os.path.dirname(path)),
            [os.path.normcase(line.strip()) for line in shown.splitlines()],
        )

    @staticmethod
    def _purge(item):
        """Remove a test's own Recycle Bin item: its ``$R`` file and ``$I`` record."""
        record = os.path.join(os.path.dirname(item), "$I" + os.path.basename(item)[2:])
        for path in (item, record):
            try:
                os.remove(path)
            except OSError:
                pass

    @unittest.skipUnless(sys.platform.startswith("win"), "the Recycle Bin is Windows'")
    def test_windows_a_file_held_open_stays_and_raises(self):
        path = self._file("held.exr")
        with open(path, "rb"):
            with self.assertRaises(OSError):
                FileUtils.move_to_trash(path)
        self.assertTrue(os.path.isfile(path))

    def test_the_trash_is_named_as_the_platform_names_it(self):
        from unittest import mock

        for platform, name in (
            ("win32", "Recycle Bin"),
            ("darwin", "Trash"),
            ("linux", "Trash"),
        ):
            with mock.patch.object(sys, "platform", platform):
                self.assertEqual(FileUtils.trash_name(), name)

    def test_can_trash_answers_before_anything_moves(self):
        """What a prompt needs before it asks: whether the file would go to the
        trash, or could only be deleted for good. Nothing is moved."""
        from unittest import mock

        from pythontk.file_utils._trash import _TrashInternal

        path = self._file("maps", "Crate.exr")
        self.assertFalse(FileUtils.can_trash(os.path.join(self.tmp, "nope.exr")))
        self.assertFalse(FileUtils.can_trash(os.path.dirname(path)), "a folder")
        if sys.platform.startswith("win"):
            self.assertTrue(FileUtils.can_trash(path), "a fixed drive keeps it")
            with mock.patch.object(
                _TrashInternal, "_windows_bin_keeps", return_value=False
            ):
                self.assertFalse(FileUtils.can_trash(path))
        with mock.patch.object(sys, "platform", "darwin"):
            trash = os.path.join(self.tmp, ".Trash")
            with mock.patch.object(
                _TrashInternal, "_macos_candidates", return_value=[trash]
            ):
                self.assertFalse(FileUtils.can_trash(path), "no ~/.Trash here")
                os.makedirs(trash)
                self.assertTrue(FileUtils.can_trash(path))
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(
                os.environ, {"XDG_DATA_HOME": os.path.join(self.tmp, "xdg")}
            ),
        ):
            self.assertTrue(FileUtils.can_trash(path), "the home trash, made on use")
        self.assertTrue(os.path.isfile(path), "nothing moved")

    @unittest.skipUnless(sys.platform.startswith("win"), "the Recycle Bin is Windows'")
    def test_windows_a_bin_that_would_not_keep_it_is_never_called(self):
        """Every case FOF_ALLOWUNDO deletes in silently is refused up front."""
        import ctypes
        from unittest import mock

        from pythontk.file_utils._trash import _TrashInternal

        path = self._file("keep.exr", data=b"x" * 4096)
        kernel32 = ctypes.windll.kernel32

        def keeps(**patches):
            if not patches:
                return _TrashInternal._windows_bin_keeps(path)
            with mock.patch.multiple(_TrashInternal, **patches):
                return _TrashInternal._windows_bin_keeps(path)

        self.assertTrue(keeps(), "the scratch folder's own fixed drive")
        for drive_type in (2, 4, 5, 6):  # removable, network, optical, RAM disk
            with mock.patch.object(kernel32, "GetDriveTypeW", return_value=drive_type):
                self.assertFalse(_TrashInternal._windows_bin_keeps(path), drive_type)

        def setting(value):
            return lambda _hive, _key, name: value.get(name)

        self.assertFalse(keeps(_registry_int=setting({"NoRecycleFiles": 1})))
        self.assertFalse(keeps(_registry_int=setting({"NukeOnDelete": 1})))
        self.assertFalse(keeps(_registry_int=setting({"MaxCapacity": 0})))
        self.assertTrue(keeps(_registry_int=setting({"MaxCapacity": 1})))
        self.assertFalse(
            _TrashInternal._windows_bin_keeps(path[:3] + "x" * 260 + ".exr"),
            "longer than MAX_PATH",
        )
        with mock.patch.object(
            _TrashInternal, "_windows_bin_keeps", return_value=False
        ):
            with mock.patch.object(ctypes.windll.shell32, "SHFileOperationW") as call:
                self.assertIsNone(FileUtils.move_to_trash(path))
        call.assert_not_called()
        self.assertTrue(os.path.isfile(path))

    @unittest.skipUnless(sys.platform.startswith("win"), "SUBST is Windows'")
    def test_windows_a_subst_drive_is_read_on_its_host_volume(self):
        """A SUBST drive's bin deletes permanently; its host volume's keeps."""
        import ctypes
        from unittest import mock

        from pythontk.file_utils._trash import _TrashInternal

        devices = {
            "X:": "\\??\\C:\\projects\\show",
            "Y:": "\\??\\UNC\\server\\share\\show",
            "C:": "\\Device\\HarddiskVolume3",
        }

        def query(drive, buffer, _size):
            buffer.value = devices.get(drive, "")
            return len(buffer.value) + 2 if buffer.value else 0

        with mock.patch.object(ctypes.windll.kernel32, "QueryDosDeviceW", query):
            self.assertEqual(
                _TrashInternal._windows_real_path("X:\\maps\\a.exr"),
                "C:\\projects\\show\\maps\\a.exr",
            )
            self.assertEqual(
                _TrashInternal._windows_real_path("Y:\\maps\\a.exr"),
                "\\\\server\\share\\show\\maps\\a.exr",
            )
            self.assertEqual(
                _TrashInternal._windows_real_path("C:\\maps\\a.exr"), "C:\\maps\\a.exr"
            )


class TestImportFiles(unittest.TestCase):
    """``FileUtils.import_files`` -- bring files into a project folder (the HDR
    Manager's Add: copy / move into sourceimages, or link in place)."""

    def setUp(self):
        self.src = tempfile.mkdtemp(prefix="ptk_import_src_")
        self.folder = tempfile.mkdtemp(prefix="ptk_import_dst_")
        self.addCleanup(shutil.rmtree, self.src, True)
        self.addCleanup(shutil.rmtree, self.folder, True)

    @staticmethod
    def _write(path, data=b"good"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    @staticmethod
    def _read(path):
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def _validate(path):
        """A file holding ``bad`` is incomplete; a missing one is not found."""
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return False, "file not found"
        return (False, "truncated") if data == b"bad" else (True, "")

    def _in_folder(self, name):
        return os.path.normpath(os.path.join(self.folder, name))

    def test_link_takes_each_usable_file_where_it_is(self):
        good = self._write(os.path.join(self.src, "good.hdr"))
        bad = self._write(os.path.join(self.src, "bad.hdr"), b"bad")
        result = FileUtils.import_files(
            [good, bad], None, mode="link", validate=self._validate
        )
        self.assertEqual(result["added"], [good])
        self.assertEqual(result["skipped"], [(bad, "truncated")])
        self.assertFalse(result["io"])
        self.assertIsNone(result["failed"])
        self.assertFalse(result["cancelled"])

    def test_each_file_left_out_is_logged(self):
        """Bug: a script caller learnt of a skipped file only by reading the
        result's ``skipped`` -- nothing said it. Fixed: 2026-10-31"""
        bad = self._write(os.path.join(self.src, "bad.hdr"), b"bad")
        good = self._write(os.path.join(self.src, "good.hdr"))
        with self.assertLogs(
            "pythontk.file_utils._file_utils", level="WARNING"
        ) as logged:
            result = FileUtils.import_files(
                [bad, good], self.folder, validate=self._validate
            )
        self.assertEqual(result["added"], [self._in_folder("good.hdr")])
        self.assertEqual(len(logged.output), 1)
        self.assertIn(bad, logged.output[0])
        self.assertIn("truncated", logged.output[0])

    def test_copy_brings_each_file_into_the_folder(self):
        good = self._write(os.path.join(self.src, "good.hdr"))
        result = FileUtils.import_files([good], self.folder, mode="copy")
        self.assertEqual(result["added"], [self._in_folder("good.hdr")])
        self.assertTrue(result["io"])
        self.assertTrue(os.path.isfile(good))  # a copy keeps the original
        self.assertEqual(self._read(self._in_folder("good.hdr")), b"good")

    def test_move_relocates_the_file(self):
        good = self._write(os.path.join(self.src, "good.hdr"))
        result = FileUtils.import_files([good], self.folder, mode="move")
        self.assertEqual(result["added"], [self._in_folder("good.hdr")])
        self.assertFalse(os.path.exists(good))
        self.assertTrue(os.path.isfile(self._in_folder("good.hdr")))

    def test_a_file_already_under_the_folder_is_used_in_place(self):
        """Regression (2026-06-16): Copy duplicated sourceimages/hdr/x.hdr into
        the sourceimages root instead of using it where it lives."""
        inside = self._write(os.path.join(self.folder, "hdr", "env.hdr"))
        result = FileUtils.import_files([inside], self.folder, mode="copy")
        self.assertEqual(result["added"], [os.path.normpath(inside)])
        self.assertFalse(result["io"])
        self.assertFalse(os.path.exists(self._in_folder("env.hdr")))

    def test_a_refused_source_is_left_out_in_every_mode(self):
        bad = self._write(os.path.join(self.src, "bad.hdr"), b"bad")
        for mode in ("copy", "move", "link"):
            with self.subTest(mode=mode):
                result = FileUtils.import_files(
                    [bad], self.folder, mode=mode, validate=self._validate
                )
                self.assertEqual(result["added"], [])
                self.assertEqual(result["skipped"], [(bad, "truncated")])
                self.assertTrue(os.path.isfile(bad))
                self.assertFalse(os.path.exists(self._in_folder("bad.hdr")))

    def test_reuse_takes_a_usable_existing_file_and_never_clobbers(self):
        new = self._write(os.path.join(self.src, "x.hdr"), b"new")
        existing = self._write(self._in_folder("x.hdr"), b"old")
        result = FileUtils.import_files(
            [new], self.folder, validate=self._validate, on_conflict="reuse"
        )
        self.assertEqual(result["added"], [existing])
        self.assertFalse(result["io"])
        self.assertEqual(self._read(existing), b"old")

    def test_reuse_leaves_out_a_source_whose_existing_file_is_unusable(self):
        new = self._write(os.path.join(self.src, "x.hdr"), b"new")
        existing = self._write(self._in_folder("x.hdr"), b"bad")
        result = FileUtils.import_files([new], self.folder, validate=self._validate)
        self.assertEqual(result["added"], [])
        self.assertEqual([s for s, _ in result["skipped"]], [new])
        self.assertIn("truncated", result["skipped"][0][1])
        self.assertEqual(self._read(existing), b"bad")  # left as it was

    def test_overwrite_replaces_the_existing_file(self):
        new = self._write(os.path.join(self.src, "x.hdr"), b"new")
        existing = self._write(self._in_folder("x.hdr"), b"old")
        result = FileUtils.import_files([new], self.folder, on_conflict="overwrite")
        self.assertEqual(result["added"], [existing])
        self.assertTrue(result["io"])
        self.assertEqual(self._read(existing), b"new")

    def test_a_conflict_callable_asks_and_a_no_cancels_the_rest(self):
        first = self._write(os.path.join(self.src, "x.hdr"), b"new")
        second = self._write(os.path.join(self.src, "y.hdr"))
        existing = self._write(self._in_folder("x.hdr"), b"old")
        asked = []

        def decline(path):
            asked.append(path)
            return False

        result = FileUtils.import_files(
            [first, second], self.folder, on_conflict=decline
        )
        self.assertEqual(asked, [existing])
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["added"], [])
        self.assertEqual(self._read(existing), b"old")
        self.assertFalse(os.path.exists(self._in_folder("y.hdr")))

        result = FileUtils.import_files(
            [first], self.folder, on_conflict=lambda path: True
        )
        self.assertEqual(result["added"], [existing])
        self.assertFalse(result["cancelled"])
        self.assertEqual(self._read(existing), b"new")

    def test_strict_stops_at_the_first_refused_source(self):
        bad = self._write(os.path.join(self.src, "bad.hdr"), b"bad")
        good = self._write(os.path.join(self.src, "good.hdr"))
        result = FileUtils.import_files(
            [bad, good], self.folder, validate=self._validate, strict=True
        )
        self.assertEqual(
            result["failed"], {"path": bad, "reason": "truncated", "destination": None}
        )
        self.assertEqual(result["added"], [])
        self.assertEqual(result["skipped"], [])
        self.assertFalse(os.path.exists(self._in_folder("good.hdr")))

    def test_a_failed_transfer_is_left_out_or_under_strict_stops(self):
        from unittest import mock

        a = self._write(os.path.join(self.src, "a.hdr"))
        b = self._write(os.path.join(self.src, "b.hdr"))
        real_copy = FileUtils.copy_file

        def copy(path, *args, **kwargs):
            if path == a:
                raise OSError("disk full")
            return real_copy(path, *args, **kwargs)

        with mock.patch.object(FileUtils, "copy_file", side_effect=copy):
            result = FileUtils.import_files([a, b], self.folder)
            self.assertEqual(result["skipped"], [(a, "disk full")])
            self.assertEqual(result["added"], [self._in_folder("b.hdr")])
            os.remove(self._in_folder("b.hdr"))
            result = FileUtils.import_files([a, b], self.folder, strict=True)
        self.assertEqual(
            result["failed"],
            {"path": a, "reason": "disk full", "destination": self._in_folder("a.hdr")},
        )
        self.assertEqual(result["added"], [])

    def test_copy_or_move_needs_an_existing_folder(self):
        """Said for the user: a caller shows the message as it is."""
        good = self._write(os.path.join(self.src, "good.hdr"))
        missing = os.path.join(self.folder, "missing")
        for folder, says in (
            (None, "set a project first"),
            ("", "set a project first"),
            (missing, f"{missing} does not exist"),
        ):
            with self.subTest(folder=folder):
                with self.assertRaises(ValueError) as caught:
                    FileUtils.import_files([good], folder, mode="copy")
                self.assertIn(says, str(caught.exception))
        with self.assertRaises(ValueError):
            FileUtils.import_files([good], self.folder, mode="symlink")


if __name__ == "__main__":
    unittest.main(exit=False)
