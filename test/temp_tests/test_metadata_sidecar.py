import os
import json
import unittest
import tempfile
import shutil
from pythontk.file_utils.metadata import Metadata


class TestMetadataSidecar(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_file.txt")
        with open(self.test_file, "w") as f:
            f.write("test content")

        # Reset state
        Metadata.enable_sidecar = False
        Metadata.sidecar_only = False

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        Metadata.enable_sidecar = False
        Metadata.sidecar_only = False

    def test_sidecar_only_set_get(self):
        """Test that sidecar_only=True uses the sidecar file."""
        Metadata.enable_sidecar = True
        Metadata.sidecar_only = True  # This is the new flag

        test_comment = "Test Comment Sidecar Only"
        Metadata.set(self.test_file, Comments=test_comment)

        # Verify sidecar file exists
        sidecar_path = f"{self.test_file}.metadata.json"
        self.assertTrue(os.path.exists(sidecar_path), "Sidecar file should exist")

        # Verify content of sidecar
        with open(sidecar_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data.get("Comments"), test_comment)

        # Verify get works
        result = Metadata.get(self.test_file, "Comments")
        self.assertEqual(result.get("Comments"), test_comment)

    def test_sidecar_only_ignores_propsys(self):
        """
        Indirectly verify we skip propsys by ensuring sidecar is created
        even if we don't mock propsys. Logic flow check.
        """
        Metadata.enable_sidecar = True
        Metadata.sidecar_only = True

        Metadata.set(self.test_file, Title="Sidecar Title")

        # Verify sidecar has it
        result = Metadata.get(self.test_file, "Title")
        self.assertEqual(result.get("Title"), "Sidecar Title")


if __name__ == "__main__":
    unittest.main()
