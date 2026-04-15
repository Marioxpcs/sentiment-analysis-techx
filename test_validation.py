"""
Unit tests for input validation in sentiment.analyze.
Run with:  python -m unittest test_validation.py
"""

import unittest

from sentiment import analyze, InvalidInputError


class ValidationTests(unittest.TestCase):

    # --- non-string input ---
    def test_none_raises(self):
        with self.assertRaises(InvalidInputError):
            analyze(None)  # type: ignore[arg-type]

    def test_int_raises(self):
        with self.assertRaises(InvalidInputError):
            analyze(42)  # type: ignore[arg-type]

    def test_list_raises(self):
        with self.assertRaises(InvalidInputError):
            analyze(["I love this"])  # type: ignore[arg-type]

    # --- empty / whitespace input ---
    def test_empty_string_raises(self):
        with self.assertRaises(InvalidInputError):
            analyze("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(InvalidInputError):
            analyze("   \t\n  ")

    # --- valid input still works ---
    def test_valid_string_returns_result(self):
        result = analyze("This is wonderful!")
        self.assertEqual(result.label, "Positive")
        self.assertGreater(result.polarity, 0)


if __name__ == "__main__":
    unittest.main()
