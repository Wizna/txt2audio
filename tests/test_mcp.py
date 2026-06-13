import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import txt2audio_mcp  # noqa: E402


class McpTests(unittest.TestCase):
    def test_tool_list_exposes_expected_tools_and_annotations(self):
        tools = {tool['name']: tool for tool in txt2audio_mcp.TOOLS}

        self.assertEqual(set(tools), {
            'txt2audio_validate_book',
            'txt2audio_plan_conversion',
            'txt2audio_convert_book',
            'txt2audio_get_manifest',
        })
        self.assertTrue(tools['txt2audio_validate_book']['annotations']['readOnlyHint'])
        self.assertFalse(tools['txt2audio_convert_book']['annotations']['readOnlyHint'])
        self.assertTrue(tools['txt2audio_convert_book']['annotations']['destructiveHint'])
        self.assertIn('outputSchema', tools['txt2audio_plan_conversion'])

    def test_build_cli_args_for_plan_conversion(self):
        args = txt2audio_mcp._build_cli_args(
            'txt2audio_plan_conversion',
            {
                'input_file_path': '/tmp/book.txt',
                'range': '0~2',
                'video': True,
                'chapter_manifest': True,
                'output_format': 'wav',
            },
        )

        self.assertEqual(args, [
            '/tmp/book.txt',
            '--plan-json',
            '--range',
            '0~2',
            '--video',
            '--chapter-manifest',
            '--output-format',
            'wav',
        ])

    def test_build_cli_args_for_convert_events(self):
        args = txt2audio_mcp._build_cli_args(
            'txt2audio_convert_book',
            {
                'input_file_path': '/tmp/book.txt',
                'events_jsonl': '-',
                'speed': 0.95,
            },
        )

        self.assertEqual(args, [
            '/tmp/book.txt',
            '--json',
            '--range',
            'all',
            '--events-jsonl',
            '-',
            '--speed',
            '0.95',
        ])

    def test_json_rpc_initialize_and_tools_list(self):
        request_stream = io.StringIO(
            json.dumps({
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {'protocolVersion': '2025-06-18'},
            }) + '\n' +
            json.dumps({
                'jsonrpc': '2.0',
                'id': 2,
                'method': 'tools/list',
            }) + '\n'
        )
        response_stream = io.StringIO()

        txt2audio_mcp.serve(request_stream, response_stream)

        responses = [json.loads(line) for line in response_stream.getvalue().splitlines()]
        self.assertEqual(responses[0]['result']['serverInfo']['name'], 'txt2audio')
        self.assertEqual(responses[1]['result']['tools'][0]['name'], 'txt2audio_validate_book')

    def test_json_rpc_parse_error_does_not_stop_server(self):
        request_stream = io.StringIO(
            '{bad json}\n' +
            '[]\n' +
            json.dumps({
                'jsonrpc': '2.0',
                'id': 2,
                'method': 'tools/list',
            }) + '\n'
        )
        response_stream = io.StringIO()

        txt2audio_mcp.serve(request_stream, response_stream)

        responses = [json.loads(line) for line in response_stream.getvalue().splitlines()]
        self.assertEqual(responses[0]['error']['code'], -32700)
        self.assertEqual(responses[0]['id'], None)
        self.assertEqual(responses[1]['error']['code'], -32600)
        self.assertEqual(responses[1]['id'], None)
        self.assertEqual(responses[2]['result']['tools'][0]['name'], 'txt2audio_validate_book')

    def test_tool_call_wraps_structured_content(self):
        payload = {'schema_version': 1, 'status': 'success', 'mode': 'plan'}
        with patch.object(txt2audio_mcp, '_run_cli_tool', return_value=(payload, 0)):
            result = txt2audio_mcp._handle_request({
                'method': 'tools/call',
                'params': {
                    'name': 'txt2audio_plan_conversion',
                    'arguments': {'input_file_path': '/tmp/book.txt'},
                },
            })

        self.assertFalse(result['isError'])
        self.assertEqual(result['structuredContent'], payload)
        self.assertEqual(json.loads(result['content'][0]['text']), payload)


if __name__ == '__main__':
    unittest.main()
