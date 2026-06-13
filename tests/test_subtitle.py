import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import subtitle  # noqa: E402


class SubtitleTests(unittest.TestCase):
    def test_format_srt_time(self):
        self.assertEqual(subtitle.format_srt_time(0), '00:00:00,000')
        self.assertEqual(subtitle.format_srt_time(3661.234), '01:01:01,233')

    def test_split_subtitle_entries_splits_long_text_at_clause_punctuation(self):
        entries = [(0.0, 4.0, '这是第一句，这是第二句；这是第三句：这是第四句')]
        result = subtitle.split_subtitle_entries(entries)

        self.assertGreater(len(result), 1)
        self.assertEqual(result[0][0], 0.0)
        self.assertAlmostEqual(result[-1][1], 4.0)
        self.assertTrue(all(text.strip('，；：') == text for _, _, text in result))


if __name__ == '__main__':
    unittest.main()
