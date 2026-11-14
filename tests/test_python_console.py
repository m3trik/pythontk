# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.python_console: PythonConsole, a namespace
that lasts from snippet to snippet, as a console's does."""

import unittest

from pythontk import Operation, PythonConsole


class TestPythonConsole(unittest.TestCase):
    def setUp(self):
        PythonConsole._namespace.clear()
        PythonConsole._namespace["__name__"] = "__main__"

    def test_the_last_expression_is_the_value_and_names_last(self):
        self.assertEqual(
            PythonConsole.run("x = 40\nx + 1"), {"value": 41, "output": ""}
        )
        self.assertEqual(PythonConsole.run("x + 2")["value"], 42)
        self.assertIsNone(
            PythonConsole.run("y = x")["value"]
        )  # a statement has no value

    def test_what_it_prints_comes_back(self):
        result = PythonConsole.run(
            "import sys\nprint('out')\nprint('err', file=sys.stderr)"
        )
        self.assertEqual(result["output"], "out\nerr\n")

    def test_a_long_output_keeps_its_end(self):
        output = PythonConsole.run(f"print('a' * {PythonConsole.OUTPUT_MAX} + 'end')")[
            "output"
        ]
        self.assertTrue(output.startswith("[the first "))
        self.assertTrue(output.rstrip().endswith("aend"))

    def test_a_huge_value_comes_back_as_the_start_of_its_repr(self):
        value = PythonConsole.run(f"list(range({PythonConsole.VALUE_MAX}))")["value"]
        self.assertIsInstance(value, str)
        self.assertTrue(value.startswith("[0, 1, 2"))
        self.assertIn("[list cut:", value)
        self.assertEqual(PythonConsole.run("[1, 2]")["value"], [1, 2])

    def test_an_error_says_where_in_the_snippet_and_what_it_printed_first(self):
        with self.assertRaises(RuntimeError) as caught:
            PythonConsole.run("print('before')\n\ndef f():\n    return 1 / 0\nf()")
        message = str(caught.exception)
        self.assertIn('File "<console>", line 4, in f', message)
        self.assertIn("ZeroDivisionError", message)
        self.assertNotIn("python_console.py", message)  # its own frames are left out
        self.assertTrue(message.endswith("--- it printed first ---\nbefore"))

    def test_a_syntax_error_or_an_exit_never_ends_the_process(self):
        with self.assertRaises(SyntaxError) as caught:  # none of it ran
            PythonConsole.run("def (:")
        self.assertIn("SyntaxError", str(caught.exception))
        with self.assertRaises(SyntaxError):  # nor of code that cannot be parsed at all
            PythonConsole.run("x = 1\0")
        with self.assertRaises(RuntimeError) as caught:
            PythonConsole.run("import sys\nsys.exit(3)")
        self.assertIn("SystemExit: 3", str(caught.exception))

    def test_code_has_no_dry_run_so_its_operation_applies_only(self):
        # Echoing the code back as a "plan" told nobody anything: code is run
        # where running it was allowed (the change server), never planned.
        entry = Operation("py.python.exec", "pythontk:PythonConsole.run", effect="code").describe()
        self.assertFalse(entry["dry_run"])
        self.assertEqual(list(entry["args"]["properties"]), ["code"])


if __name__ == "__main__":
    unittest.main()
