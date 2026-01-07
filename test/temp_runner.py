import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the test modules
from test import test_execution_monitor
from test import test_map_converter

# Create a test suite
suite = unittest.TestSuite()

# Add tests from modules
loader = unittest.TestLoader()
suite.addTests(loader.loadTestsFromModule(test_execution_monitor))
suite.addTests(loader.loadTestsFromModule(test_map_converter))

# Run the suite
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
