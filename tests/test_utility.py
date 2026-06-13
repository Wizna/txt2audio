import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import utility  # noqa: E402
import video  # noqa: E402


class FakeTensor:
    def __init__(self, samples=1000):
        self.shape = (1, samples)

    def numel(self):
        return self.shape[-1]

    def dim(self):
        return len(self.shape)

    def unsqueeze(self, dim):
        return self


class FakeTorch:
    @staticmethod
    def zeros(channels, samples):
        return FakeTensor(samples)

    @staticmethod
    def cat(chunks, dim=-1):
        return FakeTensor(sum(chunk.shape[-1] for chunk in chunks))


class FakeTts:
    sample_rate = 1000

    def inference_zero_shot(self, *args, **kwargs):
        yield {'tts_speech': FakeTensor(1000)}


class UtilityTests(unittest.TestCase):
    def test_parse_range_string_accepts_existing_forms(self):
        self.assertEqual(list(utility.parse_range_string('all', total=8)), list(range(9)))
        self.assertEqual(list(utility.parse_range_string('5', total=8)), [5])
        self.assertEqual(list(utility.parse_range_string('0~2', total=8)), [0, 1, 2])
        self.assertEqual(list(utility.parse_range_string('0-2', total=8)), [0, 1, 2])

    def test_parse_range_string_rejects_invalid_values(self):
        invalid_values = ['abc', '-1', '8~0', '0~9']
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    utility.parse_range_string(value, total=8)

    def test_convert_wav_to_mp3_raises_on_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / 'chapter-1.wav'
            wav_path.write_bytes(b'wav')
            failed = SimpleNamespace(returncode=1, stderr=b'bad codec')

            with patch.object(utility._subprocess, 'run', return_value=failed):
                with self.assertRaises(RuntimeError):
                    utility.convert_wav_to_mp3(wav_path)

            self.assertTrue(wav_path.is_file())
            self.assertFalse(wav_path.with_suffix('.mp3').exists())

    def test_convert_wav_to_mp3_returns_generated_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / 'chapter-1.wav'
            wav_path.write_bytes(b'wav')
            mp3_path = wav_path.with_suffix('.mp3')
            tmp_mp3_path = Path(temp_dir) / 'chapter-1.tmp.mp3'

            def fake_run(command, capture_output=True):
                tmp_mp3_path.write_bytes(b'mp3')
                return SimpleNamespace(returncode=0, stderr=b'')

            with patch.object(utility._subprocess, 'run', side_effect=fake_run):
                result = utility.convert_wav_to_mp3(wav_path)

            self.assertEqual(result, str(mp3_path))
            self.assertFalse(wav_path.exists())
            self.assertTrue(mp3_path.is_file())

    def test_transform_wav_to_video_raises_on_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / 'chapter-1.wav'
            audio_path.write_bytes(b'wav')
            image_path = Path(temp_dir) / 'cover.jpg'
            image_path.write_bytes(b'jpg')
            failed = SimpleNamespace(returncode=1, stdout=b'', stderr=b'ffmpeg failed')

            with patch.object(video, 'create_image_from_text', return_value=str(image_path)):
                with patch.object(video.subprocess, 'run', return_value=failed):
                    with self.assertRaises(RuntimeError):
                        video.transform_wav_to_video(0, str(audio_path), '书/章', Path(temp_dir))

            self.assertTrue(audio_path.is_file())
            self.assertFalse(audio_path.with_suffix('.mp4').exists())

    def test_transform_wav_to_video_keeps_subtitle_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / 'chapter-1.wav'
            audio_path.write_bytes(b'wav')
            srt_path = audio_path.with_suffix('.srt')
            srt_path.write_text('1\n00:00:00,000 --> 00:00:01,000\n你好\n', encoding='utf-8')
            image_path = Path(temp_dir) / 'cover.jpg'
            image_path.write_bytes(b'jpg')
            tmp_mp4 = Path(temp_dir) / 'chapter-1.tmp.mp4'

            def fake_run(command, capture_output=True, text=False, timeout=None):
                if '-filters' in command:
                    return SimpleNamespace(returncode=0, stdout=' subtitles ', stderr=b'')
                tmp_mp4.write_bytes(b'mp4')
                return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

            video._ffmpeg_has_subtitles_filter.cache_clear()
            with patch.object(video, 'create_image_from_text', return_value=str(image_path)):
                with patch.object(video.subprocess, 'run', side_effect=fake_run):
                    result = video.transform_wav_to_video(
                        0,
                        str(audio_path),
                        '书/章',
                        Path(temp_dir),
                        keep_subtitles=True,
                    )
            video._ffmpeg_has_subtitles_filter.cache_clear()

            self.assertTrue(srt_path.is_file())
            self.assertFalse(audio_path.exists())
            self.assertTrue(audio_path.with_suffix('.mp4').is_file())
            self.assertEqual(result, str(audio_path.with_suffix('.mp4')))

    def test_generate_audio_clip_can_skip_subtitle_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_stem = Path(temp_dir) / 'chapter'
            saved_indices = []

            def fake_save_audio_file(wav_tensor, sample_rate, output_path, video_clip_index, export_indices):
                saved_indices.append(video_clip_index)
                export_indices.append(video_clip_index)

            with patch.object(utility, '_load_torch_modules', return_value=(FakeTorch, None)):
                with patch.object(utility, 'get_tts', return_value=FakeTts()):
                    with patch.object(utility, 'check_export_file_exists', return_value=True):
                        with patch.object(utility, 'save_audio_file',
                                          side_effect=fake_save_audio_file):
                            result = utility.generate_audio_clip(
                                '你好。',
                                str(output_stem),
                                sample_rate=1000,
                                generate_subtitles=False,
                            )

            self.assertEqual(result, [1])
            self.assertEqual(saved_indices, [1])
            self.assertFalse((Path(temp_dir) / 'chapter-1.srt').exists())

    def test_construct_text_and_name_preserves_current_chapter_parsing(self):
        raw_data = '\n'.join([
            '开篇文字',
            '第一卷',
            '第一章',
            '第一章内容。',
            '第一章',
            '重复标题不应开新章。',
            '第二章',
            '第二章内容。',
            '后记',
            '后记内容。',
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(utility, 'OUTPUT_DIR', output_dir):
                toc, output_targets, contents = utility.construct_text_and_name(raw_data, '书名')

        self.assertEqual(toc, {
            0: '书名/引言',
            1: '书名/第一卷/第一章',
            2: '书名/第一卷/第二章',
            3: '书名/第一卷/第二章/后记',
        })
        self.assertEqual(output_targets[1], Path('书名/第一卷/第一章'))
        self.assertEqual(contents[1], ['第一章内容。', '第一章', '重复标题不应开新章。'])

    def test_construct_text_and_name_sanitizes_output_without_changing_display_path(self):
        raw_data = '\n'.join([
            '第一章',
            '正文。',
            '第二章',
            '继续。',
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(utility, 'OUTPUT_DIR', output_dir):
                toc, output_targets, _ = utility.construct_text_and_name(raw_data, '书:名?')

        self.assertEqual(toc[0], '书:名?/第一章')
        self.assertEqual(output_targets[0], Path('书_名_', '第一章'))

    def test_construct_text_and_name_preserves_current_special_sections(self):
        raw_data = '\n'.join([
            '序章',
            '序章内容。',
            '第一卷',
            '第一章',
            '正文。',
            '楔子',
            '楔子内容。',
            '第二章',
            '第二章内容。',
            '终章',
            '终章内容。',
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(utility, 'OUTPUT_DIR', output_dir):
                toc, output_targets, contents = utility.construct_text_and_name(raw_data, '书名')

        self.assertEqual(toc, {
            0: '书名/序章',
            1: '书名/第一卷/第一章',
            2: '书名/第一卷/第一章/楔子',
            3: '书名/第一卷/第二章',
            4: '书名/第一卷/第二章/终章',
        })
        self.assertEqual(output_targets[2], Path('书名/第一卷/第一章/楔子'))
        self.assertEqual(contents[4], ['终章内容。'])

    def test_mask_punctuations_removes_urls_and_normalizes_sentence_end(self):
        self.assertEqual(utility.mask_punctuations('他说——你好 https://example.com'), '他说，你好。')
        self.assertEqual(utility.mask_punctuations('※※※'), '')

    def test_annotate_polyphones_marks_current_supported_character(self):
        self.assertEqual(utility.annotate_polyphones('校对文本'), '[j][iào]对文本')
        self.assertNotIn('[', utility.annotate_polyphones('普通文本'))

    def test_build_run_plan_reports_existing_outputs_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_stem = output_dir / '书名' / '第一章'
            output_stem.parent.mkdir(parents=True)
            existing_mp3 = output_stem.parent / '第一章-1.mp3'
            existing_mp3.write_bytes(b'mp3')
            tmp_mp3 = output_stem.parent / '第一章-1.tmp.mp3'
            tmp_mp3.write_bytes(b'tmp')

            args = SimpleNamespace(video=False)
            with patch.object(utility, 'OUTPUT_DIR', output_dir):
                plan = utility._build_run_plan(
                    args,
                    {0: '书名/第一章'},
                    {0: Path('书名', '第一章')},
                    [0],
                    '书名',
                    output_dir / '书名',
                    Path('/tmp/source.txt'),
                )

            self.assertEqual(plan['mode'], 'plan')
            self.assertEqual(plan['output_format'], 'mp3')
            self.assertEqual(plan['chapter_count'], 1)
            self.assertTrue(plan['chapters'][0]['will_skip_existing'])
            self.assertEqual(plan['chapters'][0]['existing_outputs'][0]['path'], str(existing_mp3))
            self.assertTrue(tmp_mp3.exists())

    def test_build_run_plan_reports_explicit_subtitle_export(self):
        args = SimpleNamespace(video=False, srt=True, keep_srt=False, chapter_manifest=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(utility, 'OUTPUT_DIR', output_dir):
                plan = utility._build_run_plan(
                    args,
                    {0: '书名/第一章'},
                    {0: Path('书名', '第一章')},
                    [0],
                    '书名',
                    output_dir / '书名',
                    Path('/tmp/source.txt'),
                )

        self.assertTrue(plan['generate_subtitles'])
        self.assertTrue(plan['keep_subtitles'])
        self.assertEqual(plan['output_format'], 'mp3')
        self.assertTrue(plan['write_chapter_manifest'])
        self.assertEqual(
            plan['chapter_manifest_path'],
            str(Path(temp_dir) / '书名' / 'chapter_manifest.json'),
        )

    def test_save_chapter_manifest_writes_json_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / 'chapter_manifest.json'

            result = utility.save_chapter_manifest(
                manifest_path,
                {
                    'schema_version': 1,
                    'book_name': '书名',
                    'chapters': [{'index': 0, 'display_path': '书名/第一章'}],
                },
            )

            self.assertEqual(result, str(manifest_path))
            self.assertFalse((Path(temp_dir) / 'chapter_manifest.tmp.json').exists())
            self.assertIn('"book_name": "书名"', manifest_path.read_text(encoding='utf-8'))

    def test_build_chapter_manifest_records_chapter_results(self):
        manifest = utility._build_chapter_manifest(
            book_name='书名',
            source_text_path=Path('/tmp/source.txt'),
            book_output_dir=Path('/tmp/output/书名'),
            output_format='mp3',
            elapsed=1.24,
            chapter_results=[
                {
                    'index': 0,
                    'display_path': '书名/第一章',
                    'status': 'generated',
                    'clip_count': 1,
                    'existing_outputs': [],
                    'generated_outputs': ['/tmp/output/书名/第一章-1.mp3'],
                    'failures': [],
                }
            ],
        )

        self.assertEqual(manifest['schema_version'], 1)
        self.assertEqual(manifest['chapter_count'], 1)
        self.assertEqual(manifest['elapsed_seconds'], 1.2)
        self.assertEqual(manifest['chapters'][0]['status'], 'generated')

    def test_output_policy_suppresses_non_interactive_noise(self):
        interactive = SimpleNamespace(json=False, plan_json=False, quiet=False, range=None)
        explicit_range = SimpleNamespace(json=False, plan_json=False, quiet=False, range='all')
        quiet = SimpleNamespace(json=False, plan_json=False, quiet=True, range='all')

        self.assertTrue(utility._should_show_catalog(interactive))
        self.assertFalse(utility._should_show_catalog(explicit_range))
        self.assertFalse(utility._should_show_catalog(quiet))
        self.assertFalse(utility._should_show_chapter_progress(explicit_range, [0]))
        self.assertTrue(utility._should_show_chapter_progress(explicit_range, [0, 1]))
        self.assertFalse(utility._should_print_summary(quiet, []))
        self.assertTrue(utility._should_print_summary(quiet, [{'message': 'failed'}]))

    def test_subprocess_failure_message_uses_last_stderr_line(self):
        message = utility._subprocess_failure_message(
            'MP3 conversion failed',
            1,
            'ffmpeg version 7.0\nbad codec\nConversion failed!',
        )

        self.assertEqual(message, 'MP3 conversion failed (code 1): Conversion failed!')


if __name__ == '__main__':
    unittest.main()
