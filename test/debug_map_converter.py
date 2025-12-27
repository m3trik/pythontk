import unittest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from pythontk.img_utils.map_converter import MapConverterSlots
from pythontk import ImgUtils


class TestMapConverterDebug(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_sb = Mock()
        self.mock_widget = Mock()
        self.converter = MapConverterSlots(self.mock_sb)

        # Setup source dir
        self.converter.source_dir = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tb001_workflow_config_passthrough(self):
        print("\n--- Starting Debug Test ---")
        spec_file = os.path.join(self.test_dir, "test_Specular.png")
        # Create dummy image
        from PIL import Image

        img = Image.new("RGB", (64, 64), (128, 128, 128))
        img.save(spec_file)

        self.mock_sb.file_dialog.return_value = [spec_file]
        self.mock_widget.menu.chk000.isChecked.return_value = False

        print(f"Mock file_dialog return: {self.mock_sb.file_dialog.return_value}")

        with patch(
            "pythontk.img_utils.texture_map_factory.TextureMapFactory.prepare_maps"
        ) as mock_prepare:
            print(f"Patch active: {mock_prepare}")

            # Run the method
            self.converter.tb001(self.mock_widget)

            print(f"Mock called: {mock_prepare.called}")
            if mock_prepare.called:
                print(f"Call args: {mock_prepare.call_args}")
            else:
                print("Mock NOT called.")
                # Check if file_dialog was called
                print(f"File dialog called: {self.mock_sb.file_dialog.called}")
                if self.mock_sb.file_dialog.called:
                    print(
                        f"File dialog return: {self.mock_sb.file_dialog.return_value}"
                    )


if __name__ == "__main__":
    unittest.main()
