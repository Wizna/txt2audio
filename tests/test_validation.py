import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from validation import build_path_validation_entries  # noqa: E402


class ValidationTests(unittest.TestCase):
    def test_build_path_validation_entries_sorts_and_joins_output_stems(self):
        entries = build_path_validation_entries(
            Path('/tmp/output'),
            {
                2: '书名/第二章',
                0: '书名/序',
            },
            {
                2: Path('书名/第二章'),
                0: Path('书名/序'),
            },
        )

        self.assertEqual(entries, [
            {
                'index': 0,
                'display_path': '书名/序',
                'output_stem': '/tmp/output/书名/序',
            },
            {
                'index': 2,
                'display_path': '书名/第二章',
                'output_stem': '/tmp/output/书名/第二章',
            },
        ])


if __name__ == '__main__':
    unittest.main()
