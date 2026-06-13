import json
import shutil
import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT


PROTOCOL_VERSION = "2025-06-18"
SCHEMA_DIR = PROJECT_ROOT / "schemas"


def _load_schema(name):
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _input_schema(required, properties):
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


BOOK_INPUT = {
    "input_file_path": {
        "type": "string",
        "description": "Path to the source .txt, .epub, or .mobi book file.",
    },
    "range": {
        "type": "string",
        "default": "all",
        "description": "Chapter range: all, N, start~end, or start-end.",
    },
    "output_dir": {
        "type": "string",
        "description": "Optional output directory override.",
    },
}


CONVERSION_OPTIONS = {
    **BOOK_INPUT,
    "video": {"type": "boolean", "default": False},
    "landscape": {"type": "boolean", "default": False},
    "srt": {"type": "boolean", "default": False},
    "keep_srt": {"type": "boolean", "default": False},
    "chapter_manifest": {"type": "boolean", "default": False},
    "output_format": {"type": "string", "enum": ["mp3", "wav"]},
    "speed": {"type": "number"},
    "events_jsonl": {
        "type": "string",
        "description": "Optional JSONL event output path. Use '-' for stderr.",
    },
}


TOOLS = [
    {
        "name": "txt2audio_validate_book",
        "description": "Parse a source book and validate generated output paths without loading the TTS model.",
        "inputSchema": _input_schema(["input_file_path"], {
            "input_file_path": BOOK_INPUT["input_file_path"],
            "output_dir": BOOK_INPUT["output_dir"],
        }),
        "outputSchema": _load_schema("validate-paths-result.schema.json"),
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "txt2audio_plan_conversion",
        "description": "Return a model-free conversion plan, including existing outputs and skip decisions.",
        "inputSchema": _input_schema(["input_file_path"], {
            **BOOK_INPUT,
            "video": CONVERSION_OPTIONS["video"],
            "srt": CONVERSION_OPTIONS["srt"],
            "keep_srt": CONVERSION_OPTIONS["keep_srt"],
            "chapter_manifest": CONVERSION_OPTIONS["chapter_manifest"],
            "output_format": CONVERSION_OPTIONS["output_format"],
        }),
        "outputSchema": _load_schema("plan-result.schema.json"),
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "txt2audio_convert_book",
        "description": "Convert a source book into audio or video artifacts. Writes files under the output directory.",
        "inputSchema": _input_schema(["input_file_path"], CONVERSION_OPTIONS),
        "outputSchema": _load_schema("run-result.schema.json"),
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "txt2audio_get_manifest",
        "description": "Read a generated chapter_manifest.json file.",
        "inputSchema": _input_schema(["manifest_path"], {
            "manifest_path": {
                "type": "string",
                "description": "Path to a txt2audio chapter_manifest.json file.",
            },
        }),
        "outputSchema": _load_schema("chapter-manifest.schema.json"),
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
]


def _txt2audio_command():
    executable = shutil.which("txt2audio")
    if executable:
        return [executable]
    return [sys.executable, str(PROJECT_ROOT / "src" / "transform_to_audio.py")]


def _append_common_options(command, arguments):
    if arguments.get("output_dir"):
        command += ["--output-dir", arguments["output_dir"]]
    if arguments.get("output_format"):
        command += ["--output-format", arguments["output_format"]]
    if arguments.get("speed") is not None:
        command += ["--speed", str(arguments["speed"])]
    return command


def _build_cli_args(tool_name, arguments):
    input_file_path = arguments["input_file_path"]
    command = [input_file_path]

    if tool_name == "txt2audio_validate_book":
        command += ["--validate-paths", "--json"]
        return _append_common_options(command, arguments)

    if tool_name == "txt2audio_plan_conversion":
        command += ["--plan-json", "--range", arguments.get("range", "all")]
    elif tool_name == "txt2audio_convert_book":
        command += ["--json", "--range", arguments.get("range", "all")]
    else:
        raise ValueError(f"unknown CLI-backed tool: {tool_name}")

    if arguments.get("video"):
        command.append("--video")
    if arguments.get("landscape"):
        command.append("--landscape")
    if arguments.get("srt"):
        command.append("--srt")
    if arguments.get("keep_srt"):
        command.append("--keep-srt")
    if arguments.get("chapter_manifest"):
        command.append("--chapter-manifest")
    if arguments.get("events_jsonl"):
        command += ["--events-jsonl", arguments["events_jsonl"]]

    return _append_common_options(command, arguments)


def _run_cli_tool(tool_name, arguments):
    command = _txt2audio_command() + _build_cli_args(tool_name, arguments)
    completed = subprocess.run(command, capture_output=True, text=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "schema_version": 1,
            "status": "error",
            "error_code": "invalid_cli_json",
            "error": "invalid_cli_json",
            "message": completed.stdout.strip() or completed.stderr.strip() or "txt2audio returned no JSON",
            "retryable": False,
            "details": {"returncode": completed.returncode},
        }
    return payload, completed.returncode


def _read_manifest(arguments):
    manifest_path = Path(arguments["manifest_path"])
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")), 0
    except (OSError, json.JSONDecodeError) as e:
        return {
            "schema_version": 1,
            "status": "error",
            "error_code": "manifest_read_failed",
            "error": "manifest_read_failed",
            "message": str(e),
            "retryable": False,
            "details": {"manifest_path": str(manifest_path)},
        }, 1


def _call_tool(name, arguments):
    if name == "txt2audio_get_manifest":
        return _read_manifest(arguments)
    return _run_cli_tool(name, arguments)


def _tool_result(payload, returncode):
    is_error = returncode != 0 or payload.get("status") == "error"
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _handle_request(request):
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        protocol_version = params.get("protocolVersion", PROTOCOL_VERSION)
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "txt2audio", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        payload, returncode = _call_tool(params["name"], params.get("arguments") or {})
        return _tool_result(payload, returncode)
    raise ValueError(f"unsupported MCP method: {method}")


def _response(request, result=None, error=None):
    response = {"jsonrpc": "2.0", "id": request.get("id")}
    if error is None:
        response["result"] = result
    else:
        response["error"] = error
    return response


def _error_response(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(input_stream=sys.stdin, output_stream=sys.stdout):
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = _error_response(None, -32700, f"parse error: {e.msg}")
            print(json.dumps(response, ensure_ascii=False), file=output_stream, flush=True)
            continue
        if not isinstance(request, dict):
            response = _error_response(None, -32600, "invalid request: expected JSON object")
            print(json.dumps(response, ensure_ascii=False), file=output_stream, flush=True)
            continue
        if "id" not in request:
            continue
        try:
            result = _handle_request(request)
            response = _response(request, result=result)
        except Exception as e:
            response = _response(request, error={"code": -32603, "message": str(e)})
        print(json.dumps(response, ensure_ascii=False), file=output_stream, flush=True)


def main():
    serve()


if __name__ == "__main__":
    main()
