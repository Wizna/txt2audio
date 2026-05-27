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


if __name__ == '__main__':
    unittest.main()
