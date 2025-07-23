import unittest
import logging
from logging.handlers import MemoryHandler
from io import StringIO
from pythontk.core_utils.logging_mixin import LoggingMixin, DefaultTextLogHandler


class TestLoggingMixin(unittest.TestCase):
    def setUp(self):
        class TestClass(LoggingMixin):
            pass

        self.test_instance = TestClass()

    def test_logger_initialization(self):
        logger = self.test_instance.logger
        self.assertIsInstance(logger, logging.Logger)
        # Accept NOTSET or WARNING as valid defaults
        self.assertIn(logger.getEffectiveLevel(), [logging.NOTSET, logging.WARNING])
        expected_name = (
            f"{self.test_instance.__class__.__module__}."
            f"{self.test_instance.__class__.__qualname__}"
        )
        self.assertEqual(logger.name, expected_name)

    def test_logging_property(self):
        self.assertIs(self.test_instance.logging, logging)

    def test_add_stream_handler(self):
        self.test_instance.logger.add_stream_handler()
        stream_handlers = [
            h
            for h in self.test_instance.logger.handlers
            if isinstance(h, logging.StreamHandler)
        ]
        self.assertTrue(stream_handlers, "Stream handler not added")

    def test_add_file_handler(self):
        self.test_instance.logger.add_file_handler("test.log")
        file_handlers = [
            h
            for h in self.test_instance.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        self.assertTrue(file_handlers, "File handler not added")

    def test_add_text_widget_handler(self):
        class MockWidget:
            def append(self, text):
                self.last = text

        widget = MockWidget()
        self.test_instance.logger.add_text_widget_handler(widget)
        widget_handlers = [
            h
            for h in self.test_instance.logger.handlers
            if isinstance(h, DefaultTextLogHandler)
        ]
        self.assertTrue(widget_handlers, "Text widget handler not added")

    def test_log_message_capture_with_memory_handler(self):
        logger = self.test_instance.logger
        memory_handler = MemoryHandler(capacity=10000, flushLevel=logging.DEBUG)
        logger.addHandler(memory_handler)
        logger.debug("Test log message to verify capture")
        memory_handler.flush()
        captured = any(
            "Test log message to verify capture" in record.getMessage()
            for record in memory_handler.buffer
        )
        logger.removeHandler(memory_handler)
        self.assertTrue(captured, "Expected log message not captured")

    def test_log_divider_output(self):
        logger = self.test_instance.logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.log_divider(width=10, char="*")
        handler.flush()
        output = stream.getvalue()
        self.assertIn("*" * 10, output)
        logger.removeHandler(handler)

    def test_log_box_output(self):
        logger = self.test_instance.logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        title = "Title"
        items = ["Item1", "Item2"]
        width = logger.log_box(title, items, align="center")
        handler.flush()
        output = stream.getvalue()
        self.assertIn("╔", output)
        self.assertIn("║", output)
        self.assertIn("╚", output)
        self.assertIsInstance(width, int)
        logger.removeHandler(handler)

    def test_custom_log_levels(self):
        logger = self.test_instance.logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.success("Success message")
        logger.result("Result message")
        logger.notice("Notice message")
        handler.flush()
        output = stream.getvalue()
        self.assertIn("Success message", output)
        self.assertIn("Result message", output)
        self.assertIn("Notice message", output)
        logger.removeHandler(handler)

    def test_set_log_prefix_and_suffix(self):
        logger = self.test_instance.logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.set_log_prefix("[PREFIX] ")
        logger.set_log_suffix(" [SUFFIX]")
        logger.info("Test with prefix and suffix")
        handler.flush()
        output = stream.getvalue()
        self.assertIn("[PREFIX]", output)
        self.assertIn("[SUFFIX]", output)
        logger.removeHandler(handler)

    def test_logger_and_handler_level_sync(self):
        logger = self.test_instance.logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.debug("This debug message should NOT appear")
        logger.info("This info message SHOULD appear")
        handler.flush()
        output = stream.getvalue()
        self.assertNotIn("debug message", output)
        self.assertIn("info message SHOULD appear", output)
        logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
