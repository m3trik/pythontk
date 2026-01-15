import unittest
import logging
from pythontk.core_utils.logging_mixin import LoggerExt


class TestHtmlPresets(unittest.TestCase):
    def test_presets(self):
        # Test default presets
        msg = "Test Message"

        # Success (should use default style)
        formatted = LoggerExt.format_message_as_html(msg, "SUCCESS")
        color = LoggerExt.get_color("SUCCESS")
        self.assertIn(f'style="color:{color}"', formatted)
        self.assertIn(msg, formatted)

        # Highlight
        formatted = LoggerExt.format_message_as_html(msg, "INFO", preset="highlight")
        color = LoggerExt.get_color("INFO")
        self.assertIn(f'style="color:{color}"', formatted)
        self.assertIn("<hl", formatted)

    def test_style_presets(self):
        msg = "Test Message"

        # Bold (Critical default)
        formatted = LoggerExt.format_message_as_html(msg, "CRITICAL")
        self.assertIn("font-weight:bold", formatted)

        # Italic (Debug default)
        formatted = LoggerExt.format_message_as_html(msg, "DEBUG")
        self.assertIn("font-style:italic", formatted)

        # Explicit Bold on Info
        formatted = LoggerExt.format_message_as_html(msg, "INFO", preset="bold")
        self.assertIn("font-weight:bold", formatted)
        self.assertIn("color:#FFFFFF", formatted)

    def test_highlight_preset(self):
        msg = "Test Message"
        formatted = LoggerExt.format_message_as_html(msg, "INFO", preset="highlight")
        self.assertIn("<hl", formatted)


if __name__ == "__main__":
    unittest.main()
