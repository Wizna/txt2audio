import contextlib
import io
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import utility  # noqa: E402


class AgentContractTests(unittest.TestCase):
    def test_public_json_schemas_load(self):
        schema_dir = PROJECT_ROOT / 'schemas'
        expected = {
            'error.schema.json',
            'run-result.schema.json',
            'plan-result.schema.json',
            'validate-paths-result.schema.json',
            'chapter-manifest.schema.json',
        }

        actual = {path.name for path in schema_dir.glob('*.schema.json')}
        self.assertEqual(actual, expected)
        for schema_name in expected:
            with self.subTest(schema=schema_name):
                schema = json.loads((schema_dir / schema_name).read_text(encoding='utf-8'))
                self.assertEqual(schema['$schema'], 'https://json-schema.org/draft/2020-12/schema')
                self.assertIn('schema_version', schema['required'])
                self.assertEqual(schema['properties']['schema_version']['const'], 1)

    def test_json_error_envelope_has_stable_agent_fields(self):
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            utility._json_error(
                'ffmpeg_not_found',
                'missing ffmpeg',
                retryable=False,
                details={'dependency': 'ffmpeg'},
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['schema_version'], 1)
        self.assertEqual(payload['status'], 'error')
        self.assertEqual(payload['error_code'], 'ffmpeg_not_found')
        self.assertEqual(payload['error'], 'ffmpeg_not_found')
        self.assertEqual(payload['message'], 'missing ffmpeg')
        self.assertFalse(payload['retryable'])
        self.assertEqual(payload['details'], {'dependency': 'ffmpeg'})


if __name__ == '__main__':
    unittest.main()
