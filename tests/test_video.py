import sys
import unittest
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from ffmpeg_filters import (  # noqa: E402
    escape_subtitles_filter_value,
    normalize_subtitles_path,
    build_subtitles_filter,
)
from video import _wrap_text_to_width  # noqa: E402


class VideoFilterTests(unittest.TestCase):
    def test_normalize_subtitles_path_uses_forward_slashes_for_windows_paths(self):
        path = PureWindowsPath(r"C:\Books\三体\第1章.srt")
        self.assertEqual(normalize_subtitles_path(path), 'C:/Books/三体/第1章.srt')

    def test_escape_subtitles_filter_value_handles_filtergraph_special_characters(self):
        value = "C:/Books/quote's [end],semi;colon.srt"
        expected = "C\\\\:/Books/quote\\\\\\'s \\[end\\]\\,semi\\;colon.srt"
        self.assertEqual(escape_subtitles_filter_value(value), expected)

    def test_build_subtitles_filter_escapes_filename_and_style(self):
        filter_value = build_subtitles_filter(
            PureWindowsPath(r"C:\Books\三体 [终].srt"),
            "FontSize=22,PrimaryColour=&H00ffff,MarginV=95",
        )
        self.assertEqual(
            filter_value,
            "subtitles=filename=C\\\\:/Books/三体 \\[终\\].srt:"
            "force_style=FontSize=22\\,PrimaryColour=&H00ffff\\,MarginV=95:"
            "wrap_unicode=1",
        )

    def test_wrap_text_to_width_splits_long_unspaced_text(self):
        class FakeDraw:
            @staticmethod
            def textbbox(xy, text, font):
                return (0, 0, len(text) * 10, 10)

        lines = _wrap_text_to_width(FakeDraw(), '这是一个非常非常长的章节标题', object(), 60)

        self.assertGreater(len(lines), 1)
        self.assertEqual(''.join(lines), '这是一个非常非常长的章节标题')
        self.assertTrue(all(len(line) <= 6 for line in lines))


if __name__ == '__main__':
    unittest.main()
