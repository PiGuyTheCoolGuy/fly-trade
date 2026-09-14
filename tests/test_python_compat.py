"""Prevent accidental use of a newer syntax/API than the declared minimum."""

import ast
from pathlib import Path
import unittest


class PythonCompatibilityTests(unittest.TestCase):
    def test_all_project_python_parses_as_python_310(self):
        root = Path(__file__).parents[1]
        paths = [root / "run.py", *sorted((root / "flytrade").glob("*.py")),
                 *sorted((root / "tests").glob("*.py"))]
        for path in paths:
            with self.subTest(path=path.relative_to(root)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
