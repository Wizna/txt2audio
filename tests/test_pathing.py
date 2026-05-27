import os
import tempfile
import unittest
from pathlib import Path

from src.pathing import (
    build_output_relpath,
    resolve_runtime_path,
    sanitize_path_component,
    tmp_output_path,
)


class PathingTests(unittest.TestCase):
    def test_sanitize_path_component_replaces_windows_invalid_characters(self):
        self.assertEqual(sanitize_path_component('第1章: 开始?*'), '第1章_ 开始__')

    def test_sanitize_path_component_avoids_reserved_names(self):
        self.assertEqual(sanitize_path_component('CON'), '_CON')

    def test_build_output_relpath_deduplicates_collisions(self):
        used_paths = set()
        first = build_output_relpath(['书名', '第1章:开始'], used_paths)
        second = build_output_relpath(['书名', '第1章?开始'], used_paths)

        self.assertEqual(first, Path('书名', '第1章_开始'))
        self.assertEqual(second, Path('书名', '第1章_开始-2'))

    def test_tmp_output_path_preserves_parent_and_suffix(self):
        path = Path('/tmp/output/chapter-1.mp3')
        self.assertEqual(tmp_output_path(path), Path('/tmp/output/chapter-1.tmp.mp3'))

    def test_resolve_runtime_path_handles_relative_and_home(self):
        original_home = os.environ.get('HOME')
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            os.environ['HOME'] = temp_dir
            base_dir = temp_path / 'project'
            base_dir.mkdir()

            self.assertEqual(resolve_runtime_path('resources', base_dir), (base_dir / 'resources').resolve())
            self.assertEqual(resolve_runtime_path('~/speaker.wav', base_dir), (temp_path / 'speaker.wav').resolve())

        if original_home is None:
            os.environ.pop('HOME', None)
        else:
            os.environ['HOME'] = original_home


if __name__ == '__main__':
    unittest.main()
